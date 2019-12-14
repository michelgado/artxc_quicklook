from .caldb import ARTQUATS
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttWCSHistmean, AttWCSHistinteg, convolve_profile
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from .caldb import get_backprofile_by_urdn, get_shadowmask_by_urd
from ._det_spatial import DL, offset_to_vec, vec_to_offset, vec_to_offset_pairs
from .telescope import URDNS
from functools import reduce
from multiprocessing import cpu_count
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt

MPNUM = cpu_count()

def make_background_det_map_for_urdn(urdn, useshadowmask=True, ignoreedgestrips=True):
    bkgprofile = get_backprofile_by_urdn(urdn)
    shmask = get_shadowmask_by_urd(urdn)
    if ignoreedgestrips:
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False
    bkgmap = RegularGridInterpolator(((np.arange(-24, 24) + 0.5)*DL,
                                      (np.arange(-24, 24) + 0.5)*DL),
                                        bkgprofile*shmask,
                                        method="nearest", bounds_error=False, fill_value=0)
    return bkgmap

def make_overall_background_map(subgrid=10, useshadowmask=True):
    xmin, xmax = -24.5*DL, 24.5*DL
    ymin, ymax = -24.5*DL, 24.5*DL

    vecs = offset_to_vec(np.array([xmin, xmax, xmax, xmin]),
                         np.array([ymin, ymin, ymax, ymax]))
    vmaps = {}
    for urdn in URDNS:
        quat = ARTQUATS[urdn]
        xlim, ylim = vec_to_offset(quat.apply(vecs))
        xmin, xmax = min(xmin, xlim.min()), max(xmax, xlim.max())
        ymin, ymax = min(ymin, ylim.min()), max(ymax, ylim.max())

    dd = DL/subgrid
    dx = dd - (xmax - xmin)%dd
    xmin, xmax = xmin - dx/2., xmax + dx
    dy = dd - (ymax - ymin)%dd
    ymin, ymax = ymin - dy/2., ymax + dy

    x, y = np.mgrid[xmin:xmax:dd, ymin:ymax:dd]
    print(x.shape)
    shape = x.shape
    newvmap = np.zeros(shape, np.double)
    vecs = offset_to_vec(np.ravel(x), np.ravel(y))

    for urdn in URDNS:
        vmap = make_background_det_map_for_urdn(urdn, useshadowmask)
        quat = ARTQUATS[urdn]
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs))).reshape(shape)

    bkgmap = RegularGridInterpolator((x[:, 0], y[0]), newvmap, bounds_error=False, fill_value=0)
    return bkgmap

def make_bkgmap_for_wcs(wcs, attdata, urdgtis, mpnum=MPNUM, time_corr={}):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    bkg = 0
    overall_gti = emptyGTI
    """
    if time_corr:
        overall_gti = reduce(lambda a, b: a & b, urdgtis.values())
        tcorr = np.sort(np.concatenate([t.x for t in time_corr.values()]))
        ts = np.sum([time_corr.get(urdn, lambda x: np.ones(x.size))(tcorr) for urdn in URDNS], axis=0)
        tcorrf = interp1d(tcorr, ts, bounds_error=False, fill_value = np.median(ts))
        bkgmap = make_overall_background_map()
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti, tcorrf)
        print(exptime.sum(), locgti.exposure, overall_gti.exposure)
        bkg = AttWCSHistinteg.make_mp(bkgmap, exptime, qval, wcs, mpnum, subscale=10)
    """

    for urd in urdgtis:
        gti = urdgtis[urd] & -overall_gti
        if gti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*ARTQUATS[urd], gti, \
                                                     time_corr.get(urd, lambda x: 1.))
        print("processed exposure", gti.exposure, exptime.sum())
        bkgmap = make_background_det_map_for_urdn(urd)
        bkg = AttWCSHistmean.make_mp(bkgmap, exptime, qval, wcs, mpnum, subscale=6) + bkg
        print("done!")

    if wcs.wcs.has_cd():
        scale = np.linalg.det(wcs.wcs.cd)/(45./3600.)**2.
    else:
        scale = wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1]/(45./3600.)**2.
    return bkg*scale

def make_quick_bkgmap_for_wcs(wcs, attdata, urdgtis, time_corr={}):
    pixsize = sqrt(np.linalg.det(wcs.wcs.pc))
    bkgimg = 0.
    for urd in urdgtis:
        bkgmap = make_background_det_map_for_urdn(urd)
        bkg = bkgmap(tuple(np.mgrid[-24.*DL: 24.0001*DL: pixsize*DL/(45./3600.),
                                    -24.*DL: 24.0001*DL: pixsize*DL/(45./3600.)]))
        bkgimg = convolve_profile(attdata, wcs, bkg, urdgtis[urd], time_corr.get(urd, lambda x: 1.)) + bkgimg
    return bkgimg

