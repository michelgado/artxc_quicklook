from .background import get_background_surface_brigtnress, get_background_spectrum, get_particle_and_photon_templates
from copy import copy
from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAX, OPAXOFFSET, get_boresight_by_device
from ._det_spatial import offset_to_vec, vec_to_offset, DL, F, raw_xy_to_offset, raw_xy_to_vec, vec_to_offset_pairs
from .telescope import URDNS
from .lightcurve import sum_lcs, Bkgrate
from .time import emptyGTI
from .atthist import hist_orientation_for_attdata, make_wcs_steps_quats, make_small_steps_quats
from .vector import vec_to_pol, pol_to_vec
from .filters import get_shadowmask_filter, IndependentFilters
from .energy  import get_arf_energy_function
from .caldb import  get_optical_axis_offset_by_device, get_arf, get_crabspec
from .planwcs import ConvexHullonSphere, convexhull_to_wcs
from .psf import xy_to_opaxoffset, unpack_inverse_psf_ayut, unpack_inverse_psf_ayut, unpack_inverse_psf_with_weights, get_ipsf_interpolation_func, naive_bispline_interpolation, photbkg_pix_coeff
from .mosaic2 import SkyImage
from .spectr import get_filtered_crab_spectrum, Spec
from .vignetting import get_blank_vignetting_interpolation_func
from .illumination import DataDistributer
from scipy.optimize import minimize, root

from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.spatial.transform import Rotation
from scipy.special import erf
from astropy.wcs import WCS
from multiprocessing import cpu_count, Pool, Process, Queue, RawArray
import numpy as np
from functools import reduce
from math import pi, sqrt, sin, cos, log10
import pickle

MPNUM = cpu_count()



def get_sb_distribution(ax, att, filters, urdbkg, wcs=None, photbkgrate=0., urdweights={}, dtcorr={}, illum_filters=None, cspec=None):
    te, gaps, locgti = make_small_steps_quats(att, att.circ_gti(ax, 25.*60.))
    tc = (te[1:] + te[:-1])[gaps]/2.
    dt = np.diff(te)[gaps]
    iifun = get_ipsf_interpolation_func()
    x, y = np.mgrid[0:48:1, 0:48:1]
    nmask = np.zeros((82, 82), bool)
    nxof, nyof = raw_xy_to_offset(np.arange(-17.5, 65.6), np.arange(-17.5, 65.6))
    imap, jmap = np.arange(-17, 65), np.arange(-10, 65)
    ishift, jshift = np.mgrid[-7:8:1, -7:8:1]

    v = raw_xy_to_vec(x.ravel(), y.ravel()).reshape(list(x.shape) + [3,])
    cfilters = IndependentFilters({"TIME": locgti}) & filters
    bmin = 1e5
    bmax = 0.
    smin = 1e5
    smax = 0.
    for urdn in filters:
        pgrid, photrate, partrate  = get_particle_and_photon_templates(filters[urdn], cspec=cspec)
        bkgprofile = get_background_surface_brigtnress(urdn, filters[urdn].filters, fill_value=0.)
        bmax = max(partrate.max()*urdbkg[urdn].crate.max()*bkgprofile.max(), bmax)
        brmin = urdbkg[urdn].crate[urdbkg[urdn].crate > 0.].min()
        bmin = min(bmin, partrate.min()*brmin*bkgprofile[bkgprofile > 0.].min())
        smax = max(smax, photrate.max()*urdweights.get(urdn, 1/7.))
        smin = min(smin, photrate.min()*urdweights.get(urdn, 1/7.))

    print("grid smax bmax", smax, bmax, smin, bmin)

    sbin = np.linspace(0., smax*129/128.5, 129)
    bbin = np.linspace(0., bmax*128/128.5, 129)
    sbin = np.logspace(log10(smin) - 3, log10(smax), 129)
    bbin = np.logspace(log10(bmin), log10(bmax), 129)
    h = 0.
    stsum = 0.
    for urdn in filters:
        nmask[:, :] = False
        shmask = filters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        nmask[17:-17, 17:-17] = shmask
        gtimask = filters[urdn]["TIME"].mask_external(tc)
        dtl = dt[gtimask]
        tcl = tc[gtimask]
        srcvec = att.for_urdn(urdn)(tcl).apply(ax, inverse=True)
        xax, yax = vec_to_offset(srcvec)
        ic, jc = np.searchsorted(nxof, xax) - 1, np.searchsorted(nxof, yax) - 1
        pixmask = nmask[np.repeat(ic, ishift.size) + np.tile(ishift.ravel(), ic.size), np.repeat(jc, jshift.size) + np.tile(jshift.ravel(), jc.size)]
        rsize = pixmask.reshape((-1, ishift.size)).sum(axis=1)
        i, j = np.repeat(ic, rsize) + np.tile(ishift.ravel(), ic.size)[pixmask] - 17, np.repeat(jc, rsize) + np.tile(jshift.ravel(), jc.size)[pixmask] - 17
        #return i[:200], j[:200], np.repeat(srcvec, rsize, axis=0)[:200]
        #print("resulted separate pix events", i.size)
        dtl = np.repeat(dtl, rsize)
        pgrid, photrate, partrate  = get_particle_and_photon_templates(filters[urdn], cspec=cspec)
        egrid = (pgrid["ENERGY"][1:] + pgrid["ENERGY"][:-1])/2.
        lbrate = urdbkg[urdn](np.repeat(tcl, rsize))
        bkgprofile = get_background_surface_brigtnress(urdn, filters[urdn].filters, fill_value=0.)
        pbkgprofile = photbkg_pix_coeff(urdn, filters[urdn], cspec=cspec)*urdweights.get(urdn, 1/7.)

        bkgpixrate = bkgprofile[i, j]*lbrate
        pbkgpixrate = pbkgprofile[i, j]*photbkgrate

        for energy, gradphotrate, gradpartrate in zip(egrid, photrate, partrate): #[4.5,]: #egrid:
            mask, s = naive_bispline_interpolation(i, j, np.repeat(srcvec, rsize, axis=0), energy=energy, urdn=urdn)
            if not illum_filters is None:
                imask = ~illum_filters.check_pixel_in_illumination(urdn, i[mask], j[mask], att.for_urdn(urdn)(np.repeat(tcl, rsize)[mask]))
                s = s[imask]
                mask[mask] = imask
            dtw = dtl[mask]
            stsum += np.sum(s*gradphotrate.sum()*dtw)*urdweights.get(urdn, 1/7.)
            sl = (s[:, np.newaxis]*gradphotrate[np.newaxis, :]).ravel()*urdweights.get(urdn, 1/7.)
            bl = (bkgpixrate[mask, np.newaxis]*gradpartrate[np.newaxis, :] + pbkgpixrate[mask, np.newaxis]*gradphotrate[np.newaxis,:]).ravel()
            h += np.histogram2d(sl, bl, [sbin, bbin], weights = np.repeat(dtw, gradphotrate.size))[0]
    return sbin, bbin, h


import matplotlib.pyplot as plt

def get_detection_quantiles_fluxes(sbin, bbin, dthist, thlim=11.3, quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.98,]):
    ss = np.sqrt(sbin[1:]*sbin[:-1])
    bs = np.sqrt(bbin[1:]*bbin[:-1])
    B, S = np.meshgrid(bs, ss)
    sopt, bopt = S[dthist>0.], B[dthist > 0.]
    dt = dthist[dthist > 0.]
    expt = np.sum(dt*sopt)   # vignetting corrected exposure
    nbkg = np.sum(dt*bopt)   # overall expected background
    rsig = sqrt(nbkg)/expt # 1 sigma backgorund events noise
    rs = np.logspace(log10(rsig) - 1., log10(rsig) + 1., 50) # vicinity of the backgroudn noise
    th = np.array([np.sum(np.log(r*sopt/bopt + 1)*(sopt*r + bopt)*dt) - expt*r for r in rs]) # math expecteation of Theta for r
    plt.plot(rs, th)
    rthreshold = interp1d(th, rs)(thlim) # sensitivity is approximately proportional to
    rs = np.logspace(log10(rthreshold) - 1., log10(rthreshold) + 1., 60) # 0.1 -- 10 vicinity of 50% quntile
    frac = []
    #brut forse poisson simulation here :(
    for rl in rs:
        rdist = np.empty(2048, float)
        thdist = np.empty(2048, float)
        for i in range(2048):
            nevts = np.random.poisson((sopt*rl + bopt)*dt)
            sens, bens = np.repeat(sopt, nevts), np.repeat(bopt, nevts)
            rres = root(lambda r: np.sum(sens/(sens*r + bens)) - expt, rl)
            rdist[i] = max(rres.x[0], 0.)
            thdist[i] = np.sum(np.log(sens*rdist[i]/bens + 1.)) - expt*rdist[i]
        frac.append(np.sum(thdist > thlim)/2048.)
    return interp1d(frac, rs)(quantiles)



def make_detstat_psf_weigthtfun(rate, brate, imgfilter, powi=1, cspec=None, app=None):
    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    w = np.zeros(ee.size - 1, np.double)
    wb = np.zeros(ee.size - 1, np.double)

    gridb, bkgspec = get_background_spectrum(imgfilter)
    bkgspec = bkgspec.sum(axis=1)/bkgspec.sum()
    s1 = Spec(gridb["ENERGY"][:-1], gridb["ENERGY"][1:], bkgspec)

    if not cspec is None:
        arf = get_arf_energy_function(get_arf())
        egloc, egaps = imgfilter["ENERGY"].make_tedges(ee)
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        cspec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(egloc[:-1][egaps], egloc[1:][egaps])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        bspec = s1.integrate_in_bins(np.array([egloc[:-1], egloc[1:]]).T[egaps])
        np.add.at(w, np.searchsorted(ee, ec) - 1, cspec)
        np.add.at(wb, np.searchsorted(ee, ec) - 1, bspec)
    else:
        rgrid, cspec = get_filtered_crab_spectrum(imgfilter, collapsegrades=True)
        crabspec = Spec(rgrid["ENERGY"][:-1], rgrid["ENERGY"][1:], cspec)
        egloc, egaps = imgfilter["ENERGY"].make_tedges(np.unique(np.concatenate([ee, rgrid["ENERGY"]])))
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        cspec = crabspec.integrate_in_bins(np.array([egloc[:-1], egloc[1:]]).T[egaps])
        bspec = s1.integrate_in_bins(np.array([egloc[:-1], egloc[1:]]).T[egaps])
        np.add.at(w, np.searchsorted(ee, ec) - 1, cspec)
        np.add.at(wb, np.searchsorted(ee, ec) - 1, bspec)
    w = w/w.sum()
    wb = wb/wb.sum()

    ms = w > 0.

    if app is None:
        psfmask = np.ones((121, 121), np.double)
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. < app**2./25.

    def weightfunc(ipsf):
        u1 = ipsf[ms, :, :]*rate*w[ms, np.newaxis, np.newaxis]
        b1 = brate*wb[ms]
        b2 = u1 + b1[:, np.newaxis, np.newaxis]
        return np.sum((-u1/b2 + np.log(b2/b1[:, np.newaxis, np.newaxis]))**powi*b2, axis=0)*psfmask

    return weightfunc


def make_detstat_estimation(urdn, rate, brate, imgfilter, powi=1, cspec=None, app=None):
    shmask = imgfilter.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])

    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter)
    bkgprofile = bkgprofile[~np.isnan(bkgprofile)]
    bkgprofile = np.median(bkgprofile)/bkgprofile.sum()
    ipsf = unpack_inverse_psf_with_weights(make_detstat_psf_weigthtfun(rate, bkgprofile*brate, imgfilter, powi=powi, cspec=cspec, app=app))

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)

    x0, y0 = get_optical_axis_offset_by_device(urdn)
    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x1, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        sl += ipsf(xo, yo)

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


def estimate_theta_mean_val(wcs, attdata, urdgtis, imgfilters, srcrates, urdbkg, urdcrates={}, dtcorr={}, illum_filters=None, mpnum=10, **kwargs):
    def prepare_task(srcrate, powi, **kwargs):
        for urdn in urdgtis:
            vmap = make_detstat_estimation(urdn, srcrate*urdcrates.get(urdn, 1./7.), urdbkg[urdn].median(urdgtis[urdn]), imgfilters[urdn], powi=powi, **kwargs)
            tc, qlist, dtq, gloc = make_wcs_steps_quats(wcs, attdata*get_boresight_by_device(urdn), urdgtis[urdn], timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
            #print(dtq.sum(), qlist, vmap.values)
            yield qlist, dtq, vmap.values

    sky = SkyImage(wcs, get_blank_vignetting_interpolation_func(), mpnum=mpnum)

    bkgprofile = [get_background_surface_brigtnress(urdn, imgfilters[urdn]) for urdn in urdgtis if urdgtis[urdn].exposure > 0.]
    bkgprofile = np.mean([np.median(b[~np.isnan(b)])/np.sum(b[~np.isnan(b)]) for b in bkgprofile])
    mbrate = np.mean([urdbkg[urdn].median(urdgtis[urdn]) for urdn in urdgtis if urdgtis[urdn].exposure > 0.])
    imgfilter = [imgfilters[urdn] for urdn in urdgtis if urdgtis[urdn].exposure > 0.][0]

    res1 = []
    res2 = []
    res3 = []
    res4 = []
    if not illum_filters is None:
        #idata = illum_filters.prepare_data_for_computation(wcs, attdata, urdgtis, imgfilters, urdweights=urdcrates, dtcorr=dtcorr, **kwargs)
        #idata = illum_filters.prepare_data_for_computation(wcs, attdata, urdgtis, imgfilters, urdweights=urdcrates, dtcorr=dtcorr, **kwargs)
        #idata.ipsffuncs = []
        #pickle.dump(idata, open("/srg/a1/work/andrey/ART-XC/GC/idata.pkl", "wb"))
        idata = pickle.load(open("/srg/a1/work/andrey/ART-XC/GC/idata.pkl", "rb"))
        idata.set_stack_pixels(True)
        idata.dtqt = [dt/urdcrates[urdn] for urdn, dt in zip(URDNS, idata.dtqt)]
        """
        idata2 = DataDistributer(True)
        idata2.it = idata.it
        idata2.jt = idata.jt
        idata2.maskt = idata.maskt
        idata2.qlist = idata.qlist
        idata2.dtqt = idata.dtqt
        idata2.qct = idata.qct
        idata2.ipsffuncs = []
        idata = idata2
        """


    for k, srcrate in enumerate(srcrates):
        res = [srcrate, ]
        sky.set_core(get_blank_vignetting_interpolation_func())
        sky.clean_image(join=True)
        sky.img[:, :] = 0.
        sky.fft_convolve_multiple(prepare_task(srcrate, 1,**kwargs), total=len(imgfilters))
        sky.accumulate_img()
        res.append(np.copy(sky.img))
        #res1.append(np.copy(sky.img))
        sky.clean_image(join=True)
        sky.img[:, :] = 0.
        sky.fft_convolve_multiple(prepare_task(srcrate, 2,**kwargs), total=len(imgfilters))
        sky.accumulate_img()
        #res2.append(np.copy(sky.img))
        res.append(np.copy(sky.img))
        if not illum_filters is None:
            sky.set_core(get_ipsf_interpolation_func())
            sky.clean_image(join=True)
            sky.img[:, :] = 0.
            idata.ipsffuncs = [unpack_inverse_psf_with_weights(make_detstat_psf_weigthtfun(srcrate/7., mbrate*bkgprofile, imgfilter, powi=1, **kwargs)),]
            sky.fft_convolve_multiple(idata, total=idata.get_size())
            sky.accumulate_img()
            #res3.append(np.copy(sky.img))
            res.append(np.copy(sky.img))
            sky.clean_image(join=True)
            sky.img[:, :] = 0.
            idata.ipsffuncs = [unpack_inverse_psf_with_weights(make_detstat_psf_weigthtfun(srcrate/7., mbrate*bkgprofile, imgfilter, powi=2, **kwargs)),]
            sky.fft_convolve_multiple(idata, totatl=idata.get_size())
            sky.accumulate_img()
            #res4.append(np.copy(sky.img))
            res.append(np.copy(sky.img))
        pickle.dump(res, open("sensitivity/%02d.pkl" % k, "wb"))
    return res1, res2, res3, res4
