from .background import get_background_surface_brigtnress, get_background_spectrum
from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAX, OPAXOFFSET, get_boresight_by_device
from ._det_spatial import offset_to_vec, vec_to_offset, DL, F
from .telescope import URDNS
from .lightcurve import sum_lcs, Bkgrate
from .time import emptyGTI
from .atthist import hist_orientation_for_attdata
from .orientation import vec_to_pol, pol_to_vec
from .filters import get_shadowmask_filter
from .energy  import get_arf_energy_function
from .caldb import  get_optical_axis_offset_by_device, get_arf, get_crabspec
from .planwcs import ConvexHullonSphere, convexhull_to_wcs
from .psf import xy_to_opaxoffset, unpack_inverse_psf_ayut
from .mosaic import SkyImage
from .spectr import get_filtered_crab_spectrum, Spec

from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.special import erf
from astropy.wcs import WCS
from multiprocessing import cpu_count, Pool, Process, Queue, RawArray
import numpy as np
from functools import reduce
from math import pi, sqrt, sin, cos

MPNUM = cpu_count()

def make_detstat_estimation(urdn, imgfilter, phot_index=2., egrid=None, cspec=None, app=None):
    arf = get_arf_energy_function(get_arf())
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    emin, emax = imgfilter["ENERGY"].arr[0]
    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    if not cspec is None:
        egloc = np.unique(np.concatenate([egrid, ee]))
        ec = (egloc[1:] + egloc[:-1])/2.
        cspec = np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        w = cspec[np.searchsorted(egrid, ec) - 1]*np.diff(egloc)
        eidx = np.searchsorted(ee, ec) - 1
    else:
        ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
        eidx = np.searchsorted(ee, [emin, emax]) - [1, -1]
        eidx[0] = max(eidx[0], 0)
        eidx[1] = min(eidx[1], ee.size)
        eel = np.copy(ee[eidx[0]: eidx[1]])
        eel[0] = max(emin, eel[0])
        eel[-1] = min(emax, eel[-1])

        w = [quad(lambda e: arf(e)*e**-phot_index, emin, emax)[0] for emin, emax in zip(eel[:-1], eel[1:])]
        w = np.array(w)
        w = w/w.sum()

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)

    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter, fill_value=0., normalize=True)

    if app is None:
        psfmask = None
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. > app**2./25.

    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        lipsf = np.sum(unpack_inverse_psf_ayut(xo, yo)[eidx[0]:eidx[1] - 1]*w[:, np.newaxis, np.newaxis], axis=0)
        if not app is None:
            lipsf[psfmask] = 0.
        sl += lipsf**2./bkgprofile[xl, yl]/2.

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.)


def make_overall_detstat_estimation(imgfilters, scales = {}, rebinres=2./3600., **kwargs):
    maps = {urdn:make_detstat_estimation(urdn, filt, **kwargs) for urdn, filt in imgfilters.items()}

    chull = reduce(lambda a, b: a + b, [ConvexHullonSphere(get_boresight_by_device(urdn).apply(
                    offset_to_vec(m.grid[0][[0, -1, -1, 0]], m.grid[1][[0, 0, -1, -1]]))) for urdn, m in maps.items()])
    locwcs = convexhull_to_wcs(chull, alpha=pi, cax=OPAX, pixsize=rebinres, maxsize=False)
    sky = SkyImage(locwcs, vmap=maps[28])
    for urdn in imgfilters:
        sky.vmap.values = maps[urdn].values
        sky.interpolate_vmap_for_qval(get_boresight_by_device(urdn), scales.get(urdn, 1.), sky.img)
    rd1 = np.rad2deg(vec_to_pol(OPAX))
    i, j = (locwcs.all_world2pix(rd1.reshape((-1, 2)), 1) - 0.5).astype(np.int).T
    grid = [vec_to_offset(sky.vecs[j, :, :])[0][0], vec_to_offset(sky.vecs[:, i, :])[1][::-1, 0]]
    return RegularGridInterpolator(grid, sky.img.T[:, ::-1])


def make_sensitivity_map(wcs, attdata, urdgtis, imgfilters, urdbkg, shape=None, mpnum=MPNUM, kind="direct", **kwargs):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    if shape is None:
        ysize, xsize = int(wcs.wcs.crpix[0]*2 + 1), int(wcs.wcs.crpix[1]*2 + 1)
        shape = [(0, xsize), (0, ysize)]

    if kind not in ["direct", "convolve"]:
        raise ValueError("only  convolve and direct option for exposure mosiac is available")

    sky = SkyImage(wcs, shape=shape)
    overall_gti = emptyGTI

    overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
    print(overall_gti.exposure)
    if overall_gti.exposure > 0:
        te, lc = sum_lcs([urdbkg[urdn].te for urdn in URDNS], [urdbkg[urdn].crate for urdn in URDNS])
        tbkg = Bkgrate(te, lc)
        scales =  {}
        for urdn in urdbkg:
            t1, g1 = (urdgtis[urdn] & overall_gti).arange(np.max(overall_gti.arr[:, 1] - overall_gti.arr[:, 0]))
            scales[urdn] = np.sum(urdbkg[urdn].integrate_in_timebins(t1)[g1])/np.sum(tbkg.integrate_in_timebins(t1)[g1])
        print("background telescope scales", scales)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti, timecorrection=Bkgrate(tbkg.te, 1./tbkg.crate))
        vmap = make_overall_detstat_estimation(imgfilters, scales=scales, **kwargs)
        print("exptime sum", exptime.sum())
        print("produce overall urds expmap")
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        print("\ndone!")

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        bscale = Bkgrate(urdbkg[urdn].te, 1./urdbkg[urdn].crate)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti, timecorrection=bscale)
        vmap = make_detstat_estimation(urdn, imgfilters[urdn], **kwargs)
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        print(" done!")
    return sky.img


def make_direct_estimation(urdn, rate, brate, imgfilter, phot_index=2., powi=1, egrid=None, cspec=None, app=None):
    arf = get_arf_energy_function(get_arf())
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    emin, emax = imgfilter["ENERGY"].arr[0]
    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.]) #TODO put iPSF in caldb!!!!
    gridb, bkgspec = get_background_spectrum(imgfilter)
    bkgspec = bkgspec.sum(axis=1)/bkgspec.sum()
    s1 = Spec(gridb["ENERGY"][:-1], gridb["ENERGY"][1:], bkgspec)
    grids, crbspec = get_filtered_crab_spectrum(imgfilter, collapsegrades=True)
    s2 = Spec(grids["ENERGY"][:-1], grids["ENERGY"][1:], crbspec/crbspec.sum())

    wb = s1.integrate_in_bins(np.array([ee[:-1], ee[1:]]).T)
    wb = wb/wb.sum()

    if not cspec is None:
        s3 = Spec(egrid[:-1], egrid[1:], cspec)
        w = s3.integrate_in_bins(np.array([ee[:-1], ee[1:]]).T)
        w = w/w.sum()
        """
        egloc = np.unique(np.concatenate([egrid, ee]))
        ec = (egloc[1:] + egloc[:-1])/2.
        cspec = np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        w = cspec[np.searchsorted(egrid, ec) - 1]*np.diff(egloc)
        eidx = np.searchsorted(ee, ec) - 1
        """
    else:
        eidx = np.searchsorted(ee, [emin, emax]) - [1, -1]
        eidx[0] = max(eidx[0], 0)
        eidx[1] = min(eidx[1], ee.size)
        eel = np.copy(ee[eidx[0]: eidx[1]])
        eel[0] = max(emin, eel[0])
        eel[-1] = min(emax, eel[-1])

        #w = [quad(lambda e: arf(e)*e**-phot_index, emin, emax)[0] for emin, emax in zip(eel[:-1], eel[1:])]
        w = [quad(lambda e: arf(e)*e**-phot_index, min(el, emax), min(eh, emax))[0] for el, eh in zip(ee[:-1], ee[1:])]
        w = np.array(w)
        w = w/w.sum()

    ms = w > 0.

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)

    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter, fill_value=0., normalize=True)

    if app is None:
        psfmask = None
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. > app**2./25.

    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        """
        lipsf = np.sum(unpack_inverse_psf_ayut(xo, yo)[eidx[0]:eidx[1] - 1]*w[:, np.newaxis, np.newaxis], axis=0)
        if not app is None:
            lipsf[psfmask] = 0.
        sl += (-lipsf*rate/(lipsf*rate + bkgprofile[xl, yl]*brate) + np.log((lipsf*rate + bkgprofile[xl, yl]*brate)/bkgprofile[xl, yl]/brate))**powi*(lipsf*rate + bkgprofile[xl, yl]*brate)
        """
        usf = unpack_inverse_psf_ayut(xo, yo)
        if not app is None:
            usf = usf*psfmask[np.newaxis, :, :]
        u1 = usf[ms, :, :]*rate*w[ms, np.newaxis, np.newaxis]
        b1 = bkgprofile[xl, yl]*brate*wb[ms]
        b2 = u1 + b1[:, np.newaxis, np.newaxis]
        sl += np.sum((-u1/b2 + np.log(b2/b1[:, np.newaxis, np.newaxis]))**powi*b2, axis=0)

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.)

def compute_mean_threshold_for_rate(wcs, rate, attdata, urdgtis, imgfilters, urdbkg, shape=None, mpnum=MPNUM, kind="direct", **kwargs):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    if shape is None:
        ysize, xsize = int(wcs.wcs.crpix[0]*2 + 1), int(wcs.wcs.crpix[1]*2 + 1)
        shape = [(0, xsize), (0, ysize)]

    if kind not in ["direct", "convolve"]:
        raise ValueError("only  convolve and direct option for exposure mosiac is available")

    sky = SkyImage(wcs, shape=shape)
    overall_gti = emptyGTI

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti)
        t1, g1 = gti.arange(np.max(gti.arr[:, 1] - gti.arr[:, 0]))
        bmean = urdbkg[urdn].integrate_in_timebins(t1)[g1].sum()/gti.exposure
        print("mean background rate", bmean)
        vmap = make_direct_estimation(urdn, rate, bmean, imgfilters[urdn], **kwargs)
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        print(" done!")
    return sky.img


def compute_completness_forconstbkg_and_srcrate(wcs, thlim, rate, attdata, urdgtis, imgfilters, urdbkg, shape=None, mpnum=MPNUM, kind="direct", **kwargs):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    if shape is None:
        ysize, xsize = int(wcs.wcs.crpix[0]*2 + 1), int(wcs.wcs.crpix[1]*2 + 1)
        shape = [(0, xsize), (0, ysize)]

    if kind not in ["direct", "convolve"]:
        raise ValueError("only  convolve and direct option for exposure mosiac is available")

    skym1 = SkyImage(wcs, shape=shape)
    skym2 = SkyImage(wcs, shape=shape)
    overall_gti = emptyGTI

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti)
        t1, g1 = gti.arange(np.max(gti.arr[:, 1] - gti.arr[:, 0]))
        bmean = urdbkg[urdn].integrate_in_timebins(t1)[g1].sum()/gti.exposure
        print("mean background rate", bmean)
        vmap = make_direct_estimation(urdn, rate, bmean, imgfilters[urdn], **kwargs)
        skym1._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            skym1.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            skym1.convolve(qval, exptime, mpnum)
            print("urdn", urdn, skym1.img.sum())
        vmap = make_direct_estimation(urdn, rate, bmean, imgfilters[urdn], powi=2, **kwargs)
        skym2._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            skym2.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            skym2.convolve(qval, exptime, mpnum)
        print(" done!")
    return (1. + erf((skym1.img - thlim)/np.sqrt(2.*skym2.img)))*0.5


def make_smaps(urdn, imgfilter, ls=np.logspace(-4, 0, 45), phot_index=2., app=None):
    arf = get_arf_energy_function(get_arf())

    if not urdn is None:
        shmask = imgfilter.meshgrid(["RAW_X", "RAW_Y"], [np.arange(48), np.arange(48)])
        #shmask = get_shadowmask_filter(imgfilter)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    ed, gaps = imgfilter["ENERGY"].make_tedges(ee)
    w = np.zeros(ee.size - 1, np.double)
    for i, j in enumerate(np.searchsorted(ee, (ed[1:] + ed[:-1])/2.) - 1):
        if ~gaps[i]:
            continue
        w[j] += quad(lambda e: arf(e)*e**-phot_index, ed[i], ed[i+1])[0]

    w = w/w.sum()

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)
    print(x1.size)

    if app is None:
        psfmask = None
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. > app**2./25.

    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter, fill_value=0., normalize=True)
    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    idx = np.arange(img.size).reshape(img.shape)
    idxs = []
    smap = []
    bmap = []

    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        i = idx[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        lipsf = np.sum(unpack_inverse_psf_ayut(xo, yo)*w[:, np.newaxis, np.newaxis], axis=0)
        if not app is None:
            lipsf[psfmask] = 0.
            smap.append(lipsf[psfmask])
            idxs.append(i[psfmask])
        else:
            idxs.append(i.ravel())
            smap.append(lipsf.ravel())
        bmap.append(np.ones(smap[-1].size)*bkgprofile[xl, yl])


    idxs = np.concatenate(idxs)
    smap = np.concatenate(smap)
    bmap = np.concatenate(bmap)

    sval = smap/bmap
    sidx = np.argsort(sval)
    idxs = idxs[sidx]
    smap = smap[sidx]
    bmap = bmap[sidx]
    sval = sval[sidx]
    smax = sval[-1]

    bmaps = [np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double) for i in range(ls.size - 1)]
    smaps = [np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double) for i in range(ls.size - 1)]
    emaps = [np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double) for i in range(ls.size - 1)]
    i = 0
    posidx = np.searchsorted(sval, ls*smax)
    print(posidx, sval.size)
    for s, e in zip(posidx[:-1], posidx[1:]):
        np.add.at(bmaps[i].ravel(), idxs[s:e], bmap[s:e])
        np.add.at(smaps[i].ravel(), idxs[s:e], sval[s:e])
        np.add.at(emaps[i].ravel(), idxs[s:e], smap[s:e])
        i += 1

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    emaps = [RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.) for img in emaps]
    bmaps = [RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.) for img in bmaps]
    smaps = [RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.) for img in smaps]
    return ls, emaps, bmaps, smaps

def make_constbkg_sest(wcs, rate, attdata, urdgtis, imgfilters, urdbkg, **kwargs):
    for urdn in urdgtis:
        gti = urdgti[urdn]
        ls, e, b, s = make_smaps(urdn, imgfilter, **kwargs)
        print("urd %d progress:" % urdn)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti)
        vmap = make_detstat_estimation(urdn, imgfilters[urdn], **kwargs)
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        print(" done!")
    return sky.img

