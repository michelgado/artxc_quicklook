from .caldb import ARTQUATS
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttWCSHistmean, AttWCSHistinteg, convolve_profile, AttInvHist
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from .caldb import get_backprofile_by_urdn, get_shadowmask_by_urd
from ._det_spatial import DL, dxya, offset_to_vec, vec_to_offset, vec_to_offset_pairs
from .telescope import URDNS
from functools import reduce
from multiprocessing import cpu_count
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt

MPNUM = cpu_count()

def make_background_det_map_for_urdn(urdn, useshadowmask=True, ignoreedgestrips=True):
    """
    for provided urdn provides RegularGridInterpolator of backgroud profile for corresponding urdn
    """
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

def make_bkgmap_for_wcs(wcs, attdata, urdgtis, mpnum=MPNUM, time_corr={}, subscale=10):
    """
    produce background map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image


    -------------
    parameters:
        wcs - astropy.wcs.WCS
        attdata - attitude data container defined by arttools.orientation.AttDATA
        urdgtis - a dict of the form {urdn: arttools.time.GTI ...}
        mpnum - num of the processort to use in multiprocessing computation
        time_corr - a dict containing functions {urdn: urdnbkgrate(time) ...}
        subscale - defined a number of subpixels (under detecto pixels) to interpolate bkgmap
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
        #bkg = AttWCSHistmean.make_mp(bkgmap, exptime, qval, wcs, mpnum, subscale=subscale) + bkg
        bkg = AttInvHist.make_mp(wcs, bkgmap, exptime, qval,  mpnum) + bkg
        print("done!")

    if wcs.wcs.has_cd():
        scale = np.linalg.det(wcs.wcs.cd)/dxya**2.
    else:
        scale = wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1]/dxya**2.
    return bkg*scale

def make_quick_bkgmap_for_wcs(wcs, attdata, urdgtis, time_corr={}):
    pixsize = wcs.wcs.cdelt[0] #sqrt(np.linalg.det(wcs.wcs.pc))
    bkgimg = 0.

    dp = pixsize/dxya
    grid = np.arange(dp/2., 24 + dp*0.9, dp)
    grid = np.repeat(grid, 5) + np.tile(np.arange(-4, 1)*dp/5., grid.size)
    grid = grid[2:]
    grid = np.concatenate([-grid[::-1], grid])*DL
    grid = (grid[1:] + grid[:-1])/2.
    mgrid = np.meshgrid(grid, grid)

    for urd in urdgtis:
        bkgmap = make_background_det_map_for_urdn(urd)
        bkg = bkgmap((mgrid[1], mgrid[0]))
        print(bkg.shape)
        bkg = sum(bkg[i%5::5,i//5::5] for i in range(25))/25.
        print(bkg.shape)
        print("run convolve")
        bkgimg = convolve_profile(attdata, wcs, bkg, urdgtis[urd], time_corr.get(urd, lambda x: 1.)) + bkgimg
    return bkgimg
