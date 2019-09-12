import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, \
        get_gyro_quat, filter_gyrodata, vec_to_pol, pol_to_vec
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec, vec_to_offset_pairs
from .time import hist_orientation, gti_intersection
from astropy.io import fits
from math import pi, cos, sin
from multiprocessing import Pool, cpu_count
import copy
import time
from scipy.interpolate import RegularGridInterpolator
import  matplotlib.pyplot as plt

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
        x, y = np.round(locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1)).T
        locweight = weight*effarea
        plt.scatter(x, y, c=locweight)
        plt.show()
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
    return hist/10.

def make_vignmap_mp(args):
    return make_vignmap_for_quat(*args)

def make_expmap_in_wcspix(qval, wcs, rect):
    x, y = np.arange(segment[0, 0], segment[0, 1]), np.arange(segment[1, 0], segment[1, 1])
    x, y = np.repeat(x, y.size), np.tile(y, x.size)


def make_expmap_for_urd(urdfile, attfile, locwcs, segment, agti=None):
    gti = np.array([urdfile["GTI"].data["START"], urdfile["GTI"].data["STOP"]]).T
    gtiatt = np.array([attfile["ORIENTATION"].data["TIME"][[0, -1]]])
    if not agti is None: gtiatt = gti_intersection(gtiatt, agti)
    gti = gti_intersection(gtiatt, gti)
    exptime, qval = hist_orientation(attfile["ORIENTATION"].data, gti)
    qval = qval*qrot0*ART_det_QUAT[urdfile["EVENTS"].header["URDN"]]
    """
    to do: implement vignmap in caldb
    """
    #vignfile = fits.open("/home/andrey/auxiliary/artxc_quicklook/arttools/art-xc_vignea.fits")
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    xsize = int(locwcs.wcs.crpix[0]*2 - 1)
    ysize = int(locwcs.wcs.crpix[1]*2 - 1)

    #emap = make_vignmap_mp([locwcs, xsize, ysize, qval, exptime, vignfilename])
    pool = Pool(24)
    emaps = pool.map(make_vignmap_mp, [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)])
    return sum(emaps)



if __name__ == "__main__":
    pass
