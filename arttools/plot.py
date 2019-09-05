import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, \
        get_gyro_quat, filter_gyrodata, vec_to_pol, pol_to_vec
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec, vec_to_offset_pairs
from .time import hist_orientation
from astropy.io import fits
from math import pi, cos, sin
import sigproc
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

def mktimegrid(gti):
    ts = np.arange(gti[0], gti[1] + 0.9, 1.)
    ts[-1] = gti[1]
    tc = (ts[1:] + ts[:-1])/2.
    dt = ts[1:] - ts[:-1]
    return tc, dt

def make_vec_to_sky_hist_fun(vecs, effarea, locwcs, xsize, ysize):
    def hist_vec_to_sky(quat, weight):
        vec_icrs = quat.apply(vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = locwcs.all_world2pix(np.array([r, d]).T, 1).T
        locweight = weight*effarea
        return np.histogram2d(x, y, [np.arange(xsize), np.arange(ysize)], weights=locweight)[0].T
    return hist_vec_to_sky

def make_inverce_vign(vecsky, qval, exp, vignmap, hist):
    vecdet = qval.apply(vecsky, inverse=True)
    xy = vec_to_offset_pairs(vecdet)
    idx = np.where(np.all([xy[:,0] > -14.2, xy[:,0] < 14.2, xy[:,1] > -14.2, xy[:,1] < 14.2], axis=0))[0]
    hist[idx] += vignmap(xy[idx])*exp

def make_vignmap_for_quat2(locwcs, xsize, ysize, qval, exptime, vignmapfilename, energy=6.):
    vignmapfile = fits.open(vignmapfilename)
    effarea = vignmapfile["Vign_EA"].data["EFFAREA"][
        np.searchsorted(vignmapfile["Vign_EA"].data["E"], energy)]
    rg = RegularGridInterpolator([vignmapfile["Coord"].data["X"],
                                  vignmapfile["Coord"].data["Y"]],
                                 effarea/effarea.max(), bounds_error=False, fill_value=0.)
    x = np.repeat(np.arange(xsize - 1) + 1., ysize - 1)
    y = np.tile(np.arange(ysize - 1) + 1., xsize - 1)
    hist = np.zeros(x.size, np.double)
    xy = np.array([x, y]).T
    r, d = locwcs.all_pix2world(xy, 1).T
    #maskclose = qval.apply([1, 0, 0])
    """
    plt.scatter(r, d)
    plt.show()
    """
    vecsky = pol_to_vec(r/180.*pi, d/180.*pi)

    i = 0
    tstart = time.time()
    for q, e in zip(qval, exptime):
        i += 1
        make_inverce_vign(vecsky, q, e, rg, hist)
        if i == 100:
            tc = time.time()
            print("time per 100 computations", tc - tstart)
            tstart = tc
            i = 0

    return hist.reshape((xsize - 1, ysize - 1), order="F")


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
    x = np.arange(-128, 129)*27.37/257
    y = np.copy(x)
    x = np.repeat(x, y.size)
    y = np.tile(y, y.size)
    vmap = rg(np.array([x, y]).T)
    vecs = offset_to_vec(x, y)
    worker = make_vec_to_sky_hist_fun(vecs, vmap, locwcs, xsize, ysize)
    hist = sum(worker(q, exp) for q, exp in zip(qval, exptime))
    return hist

def make_vignmap_mp(args):
    return make_vignmap_for_quat2(*args)

def make_expmap_for_urd(urdfile, attfile, locwcs, agti=None):
    gti = np.array([urdfile["GTI"].data["START"], urdfile["GTI"].data["STOP"]]).T
    gtiatt = np.array([attfile["ORIENTATION"].data["TIME"][[0, -1]]])
    if not agti is None: gtiatt = sigproc.overall_gti(gtiatt, agti)
    gti = sigproc.overall_gti(gtiatt, gti)
    exptime, qval = hist_orientation(attfile["ORIENTATION"].data, gti)
    qval = qval*qrot0*ART_det_QUAT[urdfile["EVENTS"].header["URDN"]]
    """
    to do: implement vignmap in caldb
    """
    #vignfile = fits.open("/home/andrey/auxiliary/artxc_quicklook/arttools/art-xc_vignea.fits")
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    xsize = int(locwcs.wcs.crpix[0]*2) + 2
    ysize = int(locwcs.wcs.crpix[1]*2) + 2
    pool = Pool(24)

    emaps = pool.map(make_vignmap_mp, [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)])
    emap = np.sum(emaps, axis=0)
    #plt.imshow(emap, interpolation="nearest")
    #plt.show()
    #emap = make_vignmap_for_quat(locwcs, xsize, ysize, qval, exptime, vignfilename)
    return emap


if __name__ == "__main__":
    pass
