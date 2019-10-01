import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import *
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec, vec_to_offset_pairs, get_shadowed_pix_mask_for_urddata
from .time import gti_union, gti_intersection, get_gti, make_hv_gti
from .atthist import make_small_steps_quats
from .atthist  import hist_orientation_for_attfile, hist_orientation
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
import  matplotlib.pyplot as plt
import os, sys
from scipy.optimize import minimize

ART_det_mean_QUAT = Rotation([-0.0170407534937525, -0.0013956210322403, -0.0011146951027288, 0.9998532004335073])


class NoDATA(Exception):
    pass

def make_events_mask(minrawx = 0, minrawy=0, maxrawx=47, maxrawy=47,
                     mingrade=-1, maxgrade=10, minenergy=4., maxenergy=12.):
    def mask_events(urddata, grade, energy):
        eventsmask = np.all([grade > mingrade, grade < maxgrade,
                            urddata["RAW_X"] > minrawx, urddata["RAW_X"] < maxrawx,
                            urddata["RAW_Y"] > minrawy, urddata["RAW_Y"] < maxrawy,
                            energy > minenergy, energy < maxenergy], axis=0)
        return eventsmask
    return mask_events

standard_events_mask = make_events_mask()

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

    urddata = np.copy(urdfile["EVENTS"].data)
    URDN = urdfile["EVENTS"].header["URDN"]
    caldbfile = get_energycal(urdfile)
    shadow = get_shadowmask(urdfile)

    attgti = np.array([attdata["TIME"][[0, -1]]])
    gti = attgti if gti is None else gti_intersection(gti, attgti)
    gti = gti_intersection(gti, get_gti(urdfile))
    gti = gti_intersection(gti, make_hv_gti(urdfile["HK"].data))

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
    maskevents = standard_events_mask(urddata, grade, energy)
    mask[mask] = maskevents

    if not np.any(maskevents):
        raise NoDATA("empty event list, after e filter")

    urddata = urddata[maskevents]

    r, d = get_photons_sky_coord(urddata, urdfile[1].header["URDN"], attdata, 10)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    return img

def get_rawxy_hist(urddata):
    img = np.histogram2d(urddata['RAW_X'], urddata['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    agttdata = filter_gyrodata(attdata)
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return np.histogram2d(dec, ra, [ye, xe])[0]

def make_vec_to_sky_hist_fun(vecs, effarea, locwcs, img):
    def hist_vec_to_sky(quat, weight):
        vec_icrs = quat.apply(vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = (locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1) + 0.5).T.astype(np.int)
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

def make_wcs_for_radecs(ra, dec, pixsize=20./3600.):
    radec = np.array([ra, dec]).T
    ch = ConvexHull(radec)
    r, d = radec[ch.vertices].T
    def find_bbox(alpha):
        x = r*cos(alpha) - d*sin(alpha)
        y = r*sin(alpha) + d*cos(alpha)
        return (x.max() - x.min())*(y.max() - y.min())
    res = minimize(find_bbox, [pi/4., ], method="Nelder-Mead")
    alpha = res.x
    x, y = r*cos(alpha) - d*sin(alpha), r*sin(alpha) + d*cos(alpha)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xc = (xmax + xmin)/2.
    yc = (ymax + ymin)/2.

    locwcs = WCS(naxis=2)
    locwcs.wcs.cdelt = [pixsize, pixsize]
    cdmat = np.array([[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]])
    locwcs.wcs.cd = cdmat*pixsize
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.crval = [xc*cos(alpha) + yc*sin(alpha), -xc*sin(alpha) + yc*cos(alpha)]
    desize = int((ymax - ymin + 0.7)/pixsize)//2
    desize = desize + 1 - desize%2
    rasize = int((xmax - xmin + 0.7)/pixsize)//2
    rasize = rasize + 1 - rasize%2
    locwcs.wcs.crpix = [rasize, desize]
    print(locwcs)
    """
    #inspect obtained wcs region
    plt.scatter(ra, dec, color="g")
    r1, d1 = locwcs.all_pix2world(np.array([np.arange(1, rasize*2 + 1), np.ones(rasize*2)]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.arange(1, rasize*2 + 1), np.ones(rasize*2)*desize*2]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.ones(desize*2), np.arange(1, desize*2 + 1)]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.ones(desize*2)*rasize*2, np.arange(1, desize*2 + 1)]).T, 1).T
    plt.scatter(r1, d1, color="b")

    plt.scatter(*locwcs.all_pix2world([[1., 1.], [0., locwcs.wcs.crpix[1]*2 + 1],
                                      [locwcs.wcs.crpix[0]*2 + 1, locwcs.wcs.crpix[1]*2 + 1],
                                      [1., locwcs.wcs.crpix[1]*2 + 1]], 1).T, color="r")
    plt.scatter([locwcs.wcs.crval[0],], [locwcs.wcs.crval[1],], color="k")
    plt.show()
    """
    return locwcs

def make_wcs_for_quats(quats, pixsize=20./3600.):
    opaxis = quats.apply(OPAX)
    ra, dec = vec_to_pol(opaxis)
    ra, dec = ra*180/pi, dec*180/pi
    return make_wcs_for_radecs(ra, dec, pixsize)

def make_wcs_for_attsets(attflist, gti=None):
    print("wcs gti", gti)
    qvtot = []
    for attname in attflist:
        attdata = np.copy(fits.getdata(attname, 1))
        attdata = clear_att(attdata)
        if not gti is None:
            lgti = gti_intersection(gti, np.array([attdata["TIME"][[0, -1]],]))
            idx = np.searchsorted(attdata["TIME"], lgti)
            mask = np.zeros(attdata.size, np.bool)
            for s, e in idx:
                mask[s:e] = True
            attdata = attdata[mask]
        quats = get_gyro_quat(attdata)*qrot0*ART_det_mean_QUAT
        qvtot.append(quats) #qvals)

    print("attname ok")
    qvtot = Rotation(np.concatenate([q.as_quat() for q in qvtot]))
    print(len(qvtot))
    return make_wcs_for_quats(qvtot)

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

def make_mosaic_for_urdset_by_gti(urdflist, attflist, gti={}):
    """
    given two sets with paths to the urdfiles and corresponding attfiles,
    and gti as a dictionary, each key contains gti for particular urd
    the program produces overall count map and exposition map for this urdfiles set
    the wcs is produced automatically to cover nonzero exposition area with some margin
    """
    print(gti)

    attgti = None if gti == {} else \
        gti_union(np.concatenate([gti[k] for k in gti]))

    locwcs = make_wcs_for_attsets(set(attflist), attgti)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)

    dttot = []
    qvtot = []

    for urdfname, attfname in zip(urdflist, attflist):
        try:
            urdn = fits.getheader(urdfname, "EVENTS")["URDN"]
            timg = get_image(attfname, urdfname, locwcs, gti.get(urdn, None))
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            lhdu.writeto("tmpctmap.fits.gz", overwrite=True)


            attdata = np.copy(fits.getdata(attfname, 1))
            attdata = clear_att(attdata)

            urdfile = fits.open(urdfname)
            attgti = np.array([attdata["TIME"][[0, -1]]])
            tgti = attgti if urdn not in gti else gti_intersection(gti[urdn], attgti)
            tgti = gti_intersection(tgti, get_gti(urdfile))
            tgti = gti_intersection(tgti, make_hv_gti(urdfile["HK"].data))

            quats = get_gyro_quat(attdata)*qrot0*ART_det_QUAT[urdn]
            qvals, dt = make_small_steps_quats(attdata["TIME"], quats, tgti) #, fits.getdata(urdfname, "HK"))
            dttot.append(dt)
            qvtot.append(qvals)

    dttot = np.concatenate(dttot)
    qvtot = Rotation(np.concatenate([q.as_quat() for q in qvtot]))

    exptime, qval = hist_orientation(qvtot, dttot)
    pool = Pool(40)
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    stepsize = len(qval)//100
    emap = sum(pool.imap_unordered(make_vignmap_mp, [(locwcs, xsize, ysize, qval[i::stepsize], exptime[i::stepsize], vignfilename) for i in range(stepsize)]))

    emap = fits.ImageHDU(data=emap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, emap])
    ehdu.writeto("tmpexmap.fits.gz", overwrite=True)


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
    exptime, qval = hist_orientation_for_attfile(attfile["ORIENTATION"].data, gti)
    qval = qval*qrot0*ART_det_QUAT[urdfile["EVENTS"].header["URDN"]]
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
