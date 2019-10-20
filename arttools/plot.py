import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, \
        get_gyro_quat, nonzero_quaternions, vec_to_pol, pol_to_vec, get_gyro_quat_as_arr, \
        get_photons_sky_coord
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec, vec_to_offset_pairs, get_shadowed_pix_mask_for_urddata
from .time import hist_orientation, gti_intersection, get_gti, make_small_steps_quats, hist_orientation_for_attfile
from .telescope import URDNS
from .caldb import get_energycal, get_shadowmask
from .energy import get_events_energy
from astropy.io import fits
from math import pi, cos, sin
from multiprocessing import Pool, cpu_count
import copy
import time
from scipy.interpolate import RegularGridInterpolator
import  matplotlib.pyplot as plt
import os


class NoDATA(Exception):
    pass

def get_image(attname, urdname, locwcs, gti=None):
    print("image gti", gti)
    """
    filter out events outsied of gti, tripple events, events with energy < 5 or > 11
    estimates its coordinates, split each photon on 100  subphotons and spreaad uniformiliy in the pixel
    return computed photons on the coordinates grid defined by locwcs

    """
    if not os.path.exists(attname):
        raise NoDATA("no attitude file")

    attfile = fits.open(attname)
    attdata = np.copy(attfile["ORIENTATION"].data)
    quatarr = get_gyro_quat_as_arr(attdata)
    attdata = attdata[nonzero_quaternions(quatarr)]
    urdfile = fits.open(urdname)
    if urdfile["HK"].data.size < 5:
        raise NoDATA("not enough HK data")

    urddata = np.copy(urdfile["EVENTS"].data)
    URDN = urdfile["EVENTS"].header["URDN"]
    caldbfile = get_energycal(urdfile)
    shadow = get_shadowmask(urdfile)

    if not gti is None:
        gti = gti_intersection(gti, np.array([attdata["TIME"][[0, -1]]]))
        idx = np.searchsorted(urddata['TIME'], gti)
        masktime = np.zeros(urddata.size, np.bool)
        for s, e in idx:
            masktime[s:e] = True
    else:
        masktime = np.ones(urddata.size, np.bool)

    mask = np.copy(masktime)
    urddata = urddata[masktime]

    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    urddata = urddata[maskshadow]
    mask[mask] = maskshadow


    ENERGY, xc, yc, grades = get_events_energy(urddata, np.copy(urdfile["HK"].data), caldbfile)
    maskenergy = (grades >=0) & (grades <= 9)
    urddata = urddata[maskenergy]
    ENERGY = ENERGY[maskenergy]
    mask[mask] = maskenergy

    maskenergy2 = (ENERGY > 4.) & (ENERGY < 11.)
    if not np.any(maskenergy2):
        raise NoDATA("empty event list, after e filter")
    urddata = urddata[maskenergy2]
    mask[mask] = maskenergy2

    r, d = get_photons_sky_coord(urddata, urdfile[1].header["URDN"], attdata, 10)
    print("\n\n\nfinal size after filtering!!!! ", r.size)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T

    return img

def get_rawxy_hist(data):
    img = np.histogram2d(data['RAW_X'], data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    attdata = filter_gyrodata(attdata)
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return np.histogram2d(dec, ra, [ye, xe])[0]

def make_vec_to_sky_hist_fun(vecs, effarea, locwcs, xsize, ysize):
    def hist_vec_to_sky(quat, weight):
        vec_icrs = quat.apply(vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1).T
        locweight = weight*effarea
        return np.histogram2d(x, y, [np.arange(xsize + 1) + 0.5, np.arange(ysize + 1) + 0.5], weights=locweight)[0].T
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

def make_vignmap_for_quat2(locwcs, xsize, ysize, qval, exptime, vignmapfilename, energy=6.):
    vignmapfile = fits.open(vignmapfilename)
    effarea = vignmapfile["Vign_EA"].data["EFFAREA"][
        np.searchsorted(vignmapfile["Vign_EA"].data["E"], energy)]
    rg = RegularGridInterpolator([vignmapfile["Coord"].data["X"],
                                  vignmapfile["Coord"].data["Y"]],
                                 effarea/effarea.max(), bounds_error=False, fill_value=0.)
    x = np.repeat(np.arange(ysize) + 1., xsize)
    y = np.tile(np.arange(xsize) + 1., ysize)
    print(x.size, y.size)
    hist = np.zeros(x.size, np.double)
    xy = np.array([x, y]).T
    r, d = locwcs.all_pix2world(xy, 1).T
    vecsky = pol_to_vec(r, d)
    pool = Pool(24)
    hist[:] = pool.map(VignInt(vecsky, qval, exptime, rg), range(x.size))
    return hist.reshape((ysize, xsize)).T

def make_vignmap_for_quat(locwcs, xsize, ysize, qval, exptime, vignmapfilename, energy=6.):
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
                                 effarea/effarea.max())
    x = (np.arange(-230, 230) + 0.5)*0.0595
    y = np.copy(x)
    x = np.repeat(x, y.size)
    y = np.tile(y, y.size)
    vmap = rg(np.array([x, y]).T)
    vecs = offset_to_vec(x, y)
    worker = make_vec_to_sky_hist_fun(vecs, vmap, locwcs, xsize, ysize)
    hist = sum(worker(q, exp) for q, exp in zip(qval, exptime))
    return hist/100.

def make_vignmap_mp(args):
    return make_vignmap_for_quat(*args)

def make_wcs_for_urd_sets(urdflist, attflist, gti = {}):
    if type(gti) is np.ndarray:
        gti = {urd:gti for urd in URDNS}
    qvtot, dttot = [], []
    for urdfname, attfname in zip(urdflist, attflist):
        attfile = fits.open(attfname)
        attdata = np.copy(attfile["ORIENTATION"].data)
        locgti = get_gti(fits.open(urdfname))
        locgti = gti_intersection(locgti, np.array([attdata["TIME"][[0, -1]]]))
        urdn = fits.getheader(urdfname, 1)["URDN"]
        locgti = locgti if not urdn in gti else gti_intersection(locgti, gti.get(urdn))
        if locgti.size == 0:
            continue
        quats = get_gyro_quat(attdata)*qrot0*ART_det_QUAT[urdn]
        qvals, dt = make_small_steps_quats(attdata["TIME"], quats, locgti)
        qvtot.append(qvals)
        dttot.append(dt)
    dttot = np.concatenate(dttot)
    qvtot = Rotation(np.concatenate([q.as_quat() for q in qvtot]))
    opaxis = qvtot.apply([1, 0, 0])
    ra, dec = vec_to_pol(opaxis)
    ra, dec = ra*180/pi, dec*180/pi
    ramin, ramax = ra.min() - 0.5, ra.max() + 0.5
    decmin, decmax = dec.min() - 0.5, dec.max() + 0.5
    locwcs = WCS(naxis=2)
    locwcs.wcs.cdelt = [20./3600., 20./3600.]
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    desize = int((decmax - decmin)/locwcs.wcs.cdelt[1])//2
    desize = desize + 1 - desize%2
    rasize = int((ramax - ramin)/locwcs.wcs.cdelt[0])//2
    rasize = rasize + 1 - rasize%2
    locwcs.wcs.crpix = np.array([rasize, desize], np.int)
    locwcs.wcs.crval = [(ramin + ramax)/2., (decmin + decmax)/2.]
    return qvtot, dttot, locwcs

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

    qvtot, dttot, locwcs = make_wcs_for_urd_sets(urdflist, attflist, gti)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    print(imgdata.shape)
    for urdfname, attfname in zip(urdflist, attflist):
        try:
            urdn = fits.getheader(urdfname, "EVENTS")["URDN"]
            timg = get_image(attfname, urdfname, locwcs, gti.get(urdn))
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            lhdu.writeto("tmpctmap.fits.gz", overwrite=True)
    exptime, qval = hist_orientation(qvtot, dttot)

    pool = Pool(24)
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    emaps = pool.map(make_vignmap_mp, [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)])
    emap = sum(emaps)

    emap = fits.ImageHDU(data=emap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, emap])
    ehdu.writeto("tmpemap.fits.gz", overwrite=True)


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
