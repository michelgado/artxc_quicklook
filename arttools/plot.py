from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import *
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec, vec_to_offset_pairs, get_shadowed_pix_mask_for_urddata
from .time import gti_union, gti_intersection, get_gti, get_filtered_table, gti_difference
from .atthist import make_small_steps_quats, hist_orientation_for_attdata, hist_orientation, make_wcs_for_attdata
from .telescope import URDNS, OPAX
from .caldb import get_energycal, get_shadowmask
from .energy import get_events_energy
from astropy.io import fits
from math import pi, cos, sin
from multiprocessing import Pool, cpu_count, Queue, Process, Pipe
from threading import Thread
import copy
import time
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import os, sys
from scipy.optimize import minimize
from .expmap import make_expmap_for_wcs
from .lightcurve import get_overall_countrate
from .background import make_bkgmap_for_wcs
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm


class NoDATA(Exception):
    pass

def make_events_mask(minrawx = 0, minrawy=0, maxrawx=47, maxrawy=47,
                     mingrade=-1, maxgrade=10, minenergy=4., maxenergy=16.):
    def mask_events(urddata, grade, energy):
        eventsmask = np.all([grade > mingrade, grade < maxgrade,
                            urddata["RAW_X"] > minrawx, urddata["RAW_X"] < maxrawx,
                            urddata["RAW_Y"] > minrawy, urddata["RAW_Y"] < maxrawy,
                            energy > minenergy, energy < maxenergy], axis=0)
        return eventsmask
    return mask_events

standard_events_mask = make_events_mask(minenergy=4., maxenergy=16.)

def make_image(urdfile, attdata, locwcs, gti=None, maskevents=standard_events_mask):
    urddata = np.copy(urdfile["EVENTS"].data)
    URDN = urdfile["EVENTS"].header["URDN"]
    caldbfile = get_energycal(urdfile)
    shadow = get_shadowmask(urdfile)

    attgti = np.array([attdata["TIME"][[0, -1]]])
    gti = attgti if gti is None else gti_intersection(gti, attgti)
    gti = gti_intersection(gti, get_gti(urdfile))

    idx = np.searchsorted(urddata['TIME'], gti)
    masktime = np.zeros(urddata.size, np.bool)
    for s, e in idx:
        masktime[s:e] = True

    mask = np.copy(masktime)
    urddata = urddata[masktime]

    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    urddata = urddata[maskshadow]
    mask[mask] = maskshadow

    energy, xc, yc, grade = get_events_energy(urddata, np.copy(urdfile["HK"].data), caldbfile)
    emask = maskevents(urddata, grade, energy)
    mask[mask] = emask

    if not np.any(emask):
        raise NoDATA("empty event list, after e filter")

    urddata = urddata[emask]
    print("events on image", urddata.size)

    r, d = get_photons_sky_coord(urddata, urdfile[1].header["URDN"], attdata, 10)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    return img


def get_image(attname, urdname, locwcs, gti=None, maskevents=standard_events_mask):
    """
    filter out events outsied of gti, tripple events, events with energy < 5 or > 11
    estimates its coordinates, split each photon on 100  subphotons and spreaad uniformiliy in the pixel
    return computed photons on the coordinates grid defined by locwcs

    """
    if not os.path.exists(attname):
        raise NoDATA("no attitude file")

    attfile = fits.open(attname)
    attdata = np.copy(attfile["ORIENTATION"].data)
    attdata = clear_att(attdata)
    urdfile = fits.open(urdname)
    if urdfile["HK"].data.size < 5:
        raise NoDATA("not enough HK data")
    return make_image(attdata, urdfile, locwcs, gti, maskevents)


def get_rawxy_hist(urddata):
    img = np.histogram2d(urddata['RAW_X'], urddata['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    agttdata = filter_gyrodata(attdata)
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return np.histogram2d(dec, ra, [ye, xe])[0]

def make_vec_to_sky_hist_fun(vecs, effarea, locwcs, img):
    def hist_vec_to_sky(quat, weight):
        vec_icrs = quat.apply(vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1).T
        locweight = weight*effarea
        return np.add.at(img, [i, j], locweight)
    return hist_vec_to_sky

def make_inverce_vign(vecsky, qval, exp, vignmap, hist):
    vecdet = qval.apply(vecsky, inverse=True)
    xy = vec_to_offset_pairs(vecdet)
    idx = np.where(np.all([xy[:,0] > -14.2, xy[:,0] < 14.2, xy[:,1] > -14.2, xy[:,1] < 14.2], axis=0))[0]
    hist[idx] += vignmap(xy[idx])*exp

class VignInt(object):
    def __init__(self, vecsky, qval, exptime, rg):
        self.vecsky = vecsky
        self.qval = qval
        self.exptime = exptime
        self.rg = rg
        self.iter_dimension = self.iter_along_time

    def __call__(self, i):
        return self.iter_dimension(i)

    def iter_along_time(self, i):
        vecdet = self.qval.apply(self.vecsky[i], inverse=True)
        xy = vec_to_offset_pairs(vecdet)
        mask = np.all([xy[:,0] > -14.2, xy[:,0] < 14.2, xy[:,1] > -14.2, xy[:,1] < 14.2], axis=0)
        return np.sum(self.exptime[mask]*self.rg(xy[mask]))

    def iter_along_coordinate(self, i):
        vecdet = self.qval[i].apply(self.vecsky, inverse=True)
        xy = vec_to_offset_pairs(vecdet)
        mask = np.all([xy[:,0] > -14.2, xy[:,0] < 14.2, xy[:,1] > -14.2, xy[:,1] < 14.2], axis=0)
        return np.sum(self.exptime[mask]*self.rg(xy[mask]))

class SkyPix(object):
    def __init__(self, rg, qval, exptime, locwcs):
        self.rg = rg
        self.qval = qval
        self.exptime = exptime
        self.locwcs = locwcs

    def __call__(self, idx):
        x, y = idx
        ra, dec = self.locwcs.all_pix2world([[x, y],], 1).T
        vec = pol_to_vec(ra*pi/180., dec*pi/180.,)
        vax = self.qval.apply([1., 0., 0.])
        vprod = np.sum(vax*vec[0][np.newaxis,:], axis=1)
        m1 = np.arccos(vprod) < 1170./3600.*pi/180.
        xyoffset = vec_to_offset_pairs(self.qval[m1].apply(vec, inverse=True))
        efscale = np.sum(self.rg(xyoffset)*self.exptime[m1])
        return x - 1, y - 1, efscale


class SphProj(object):

    @staticmethod
    def worker(rg, qval, exptime, locwcs, qin, qout):
        while True:
            i, j = qin.get()
            if i == -1:
                break
            ra, dec = locwcs.all_pix2world([[i, j],], 1).T
            vec = pol_to_vec(ra*pi/180., dec*pi/180.,)
            """
            vax = self.qval.apply([1., 0., 0.])
            vprod = np.sum(vax*vec[0][np.newaxis,:], axis=1)
            m1 = np.arccos(vprod) < 1170./3600.*pi/180.
            xyoffset = vec_to_offset_pairs(self.qval[m1].apply(vec, inverse=True))
            efscale = np.sum(self.rg(xyoffset)*self.exptime[m1])
            """
            xyoffset = vec_to_offset_pairs(qval.apply(vec[0], inverse=True))
            efscale = np.sum(rg(xyoffset)*exptime)
            qout.put([i - 1, j - 1, efscale])

    @staticmethod
    def collect(qout, img, size):
        for k in range(size):
            i, j, exp = qout.get()
            img[j, i] = exp

    def __init__(self, rg, qval, exptime, locwcs, mpnum=40):
        self.qin = Queue(100)
        self.qout = Queue()
        self.pool = [Process(target=self.worker, args=(rg, qval, exptime, locwcs, self.qin, self.qout)) for i in range(mpnum)]

    def start(self, xsize, ysize):
        img = np.zeros((ysize, xsize), np.double)
        for proc in self.pool: proc.start()

        #pix = np.array([np.repeat(np.arange(1, xsize + 1), ysize), np.tile(np.arange(1, ysize + 1), xsize)]).T
        ctr = 0
        th = Thread(target=self.collect, args=(self.qout, img, xsize*ysize))
        th.start()
        for i in range(1, xsize + 1):
            for j in range(1, ysize + 1):
                self.qin.put([i, j])
                sys.stderr.write('\rdone {0:%}'.format((i*ysize + j)/xsize/ysize))
        for i in range(len(self.pool) + 5):
            self.qin.put([-1, None])
        for p in self.pool: p.join()
        th.join()
        return img

def make_vignmap_for_quat2(locwcs, xsize, ysize, qval, exptime, vignmapfilename, energy=6.):
    vignmapfile = fits.open(vignmapfilename)
    img = np.zeros((ysize, xsize), np.double)
    effarea = vignmapfile["Vign_EA"].data["EFFAREA"][
        np.searchsorted(vignmapfile["Vign_EA"].data["E"], energy)]
    rg = RegularGridInterpolator([vignmapfile["Coord"].data["X"],
                                  vignmapfile["Coord"].data["Y"]],
                                 effarea/effarea.max(), bounds_error=False, fill_value=0.)
    proc = SphProj(rg, qval, exptime, locwcs)
    img = proc.start(xsize, ysize)
    return img

def make_vignmap_for_quat(locwcs, xsize, ysize, qval, exptime, vignmapfilename, energy=6., ssize=64000):
    """
    to do: implement mpi reduce
    """

    """
    you should be very cautiont about the memory usage here,
    we expect, that array is stored in c order -
    effarea[i, j](in 2d representation) = effarea[i*ysize + j]
    """
    vignmapfile = fits.open(vignmapfilename)
    effarea = vignmapfile["Vign_EA"].data["EFFAREA"][
        np.searchsorted(vignmapfile["Vign_EA"].data["E"], energy)]
    rg = RegularGridInterpolator([vignmapfile["Coord"].data["X"],
                                  vignmapfile["Coord"].data["Y"]],
                                 effarea/effarea.max(),
                                 bounds_error = False,
                                 fill_value=0.)
    x = np.tile((np.arange(-115, 115) + 0.5)*0.595/5., 230)
    y = np.repeat((np.arange(-115, 115) + 0.5)*0.595/5., 230)
    mxy = (x**2. + y**2.) < (25.*0.595)**2.
    x, y = x[mxy], y[mxy]

    vmap = rg(np.array([x, y]).T)
    vecs = offset_to_vec(x, y)
    img = np.zeros((ysize, xsize), np.double)

    for q, exp in zip(qval ,exptime):
        vec_icrs = q.apply(vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = (locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1) + 0.5).T.astype(np.int)
        u, idx = np.unique(np.array([x, y]), return_index=True, axis=1)
        mask = np.all([u[0] > -1, u[1] > -1, u[0] < img.shape[1], u[1] < img.shape[0]], axis=0)
        u, idx = u[:, mask], idx[mask]
        np.add.at(img, (u[1], u[0]), vmap[idx]*exp)
    return img

def make_vignmap_mp(args):
    return make_vignmap_for_quat(*args)

def make_mosaic_for_urdset_by_region(urdflist, attflist, ra, dec, deltara, deltadec):
    gti = {}
    urdflist = {}
    for urdfname, attfname in zip(rdflist, attflist):
        urdn = fits.getheader(urdfname, "EVENTS")["URDN"]
        lgti = make_orientation_gti(attdata, urdn, ra, dec, deltara, deltadec)
        gti[urdn] = [lgti,] if urdn not in gti else gti[urdn] + [lgti,]

    for urdn in gti:
        gti[urdn] = np.concatenate(gti[urdn])

    qvtot, dttot, locwcs = make_wcs_for_urd_sets(urdflist, attflist, gti)
    imgdata = np.empty((int(locwcs.wcs.crpix[0])*2 + 1, int(locwcs.wcs.crpix[1])*2 + 1), np.double)
    for urdfname, attfname in zip(urdflist, attflist):
        try:
            urdn = fits.getheader(urdfname, "EVENTS")["URDN"]
            timg = get_image(attfname, urdfname, locwcs, gti.det(urdn))
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            lhdu.writeto("tmpctmap.fits.gz", overwrite=True)

    qval, exptime = hist_orientation(qvtot, dttot)
    xsize = int(locwcs.wcs.crpix[0]*2 + 1)
    ysize = int(locwcs.wcs.crpix[1]*2 + 1)

    pool = Pool(24)
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    emaps = pool.map(make_vignmap_mp, [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)])
    emap = sum(emaps)

    emap = fits.ImageHDU(data=emap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, emap])
    ehdu.writeto("tmpemap.fits.gz", overwrite=True)

def make_mosaic_for_urdset_by_gti(urdflist, attflist, gti):
    """
    given two sets with paths to the urdfiles and corresponding attfiles,
    and gti as a dictionary, each key contains gti for particular urd
    the program produces overall count map and exposition map for this urdfiles set
    the wcs is produced automatically to cover nonzero exposition area with some margin
    """
    ottgti = np.empty((0, 2), np.double)
    attgti = []
    attall = []
    for attname in set(attflist):
        attdata = np.copy(fits.getdata(attname, "ORIENTATION"))
        attdata = clear_att(attdata)
        attgti = gti_intersection(gti, np.array([attdata["TIME"][[0, -1]],]))
        if attgti.size == 0:
            continue
        print(attname)
        print(attgti)
        attgti = gti_difference(ottgti, attgti)
        print(attgti)
        if attgti.size == 0:
            continue

        attall.append(get_filtered_table(attdata, attgti))
        ottgti = gti_union(np.concatenate([ottgti, attgti]))

    attall = np.concatenate(attall)
    attall = attall[np.argsort(attall["TIME"])]
    gti = ottgti


    locwcs = make_wcs_for_attdata(attall, gti)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    urdgti = {}
    bkgrate = {}
    bkgts = {}
    k = 0

    for urdfname in urdflist[:]:
        try:
            urdfile = fits.open(urdfname)
            urdn = urdfile["EVENTS"].header["URDN"]
            ts, rate = get_overall_countrate(urdfile, 40., 100.)
            bkgrate[urdn] = bkgrate.get(urdn, []) + [rate,]
            bkgts[urdn] = bkgts.get(urdn, []) + [ts,]
            locgti = gti_difference(urdgti.get(urdn, np.empty((0, 2), np.double)), get_gti(urdfile))
            locgti = gti_intersection(gti, locgti)
            print(urdfname, locgti)
            print(np.any(np.isnan(rate)))
            print(np.where(np.isnan(rate)))
            timg = make_image(urdfile, attall, locwcs, locgti)
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            """
            plt.clf()
            plt.imshow(timg, interpolation="nearest", norm=LogNorm())
            plt.title(urdfname)
            plt.savefig("plots/%04d.png" % k)
            k += 1
            """
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            lhdu.writeto("tmpctmap.fits.gz", overwrite=True)
            urdgti[urdn] = np.concatenate([urdgti.get(urdn, np.empty((0, 2), np.double)), locgti])
    #urdgti = {urdn:np.concatenate(urdgti[urdn]) for urdn in urdgti}
    #urdgti = {urdn:gti_intersection(gti, urdgti[urdn]) for urdn in urdgti}
    for urdn in bkgrate:
        ts = np.concatenate(bkgts[urdn])
        rt = np.concatenate(bkgrate[urdn])
        idx = np.argsort(ts)
        bkgrate[urdn] = interp1d(ts[idx], rt[idx], bounds_error=False, fill_value=np.median(rt))
    import pickle
    pickle.dump(bkgrate, open("bkgrate.pickle", "wb"))
    pickle.dump(urdgti, open("urdgti.pickle", "wb"))

    emap = make_expmap_for_wcs(locwcs, attall, urdgti)
    emap = fits.ImageHDU(data=emap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, emap])
    ehdu.writeto("tmpemap.fits.gz", overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attall, urdgti, time_corr=bkgrate)
    bmap = fits.ImageHDU(data=bmap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, bmap])
    ehdu.writeto("tmpbmap.fits.gz", overwrite=True)


def make_expmap_for_urd(urdfile, attfile, locwcs, agti=None):
    """
    given the urdfile, attfile, wcs and gti produces exposition map in the wcs coordinates.
    gti is generated from as an intesection of agti and urdfile gti
    it is assumed implicitly, that locwcs crpix is a precise center of the produced exposition map
    therefore the size of produced image is locwcs.wcs.crpix*2 + 1 (assuming crpix is odd)
    """
    gti = np.array([urdfile["GTI"].data["START"], urdfile["GTI"].data["STOP"]]).T
    gtiatt = np.array([attfile["ORIENTATION"].data["TIME"][[0, -1]]])
    if not agti is None: gtiatt = gti_intersection(gtiatt, agti)
    gti = gti_intersection(gtiatt, gti)
    exptime, qval = hist_orientation_for_attdata(attfile["ORIENTATION"].data, gti)
    qval = qval*ART_det_QUAT[urdfile["EVENTS"].header["URDN"]]
    """
    to do: implement vignmap in caldb
    """
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    xsize = int(locwcs.wcs.crpix[0]*2 - 1)
    ysize = int(locwcs.wcs.crpix[1]*2 - 1)

    pool = Pool(24)
    emap = sum(pool.imap_unordered(make_vignmap_mp,
                [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)]))
    return emap


if __name__ == "__main__":
    pass
