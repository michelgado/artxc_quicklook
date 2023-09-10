from .caldb import get_boresight_by_device, get_shadowmask_by_urd, \
                    make_background_brightnes_profile, get_background_for_urdn, get_overall_background, \
                    get_crabspec_for_filters, get_optical_axis_offset_by_device, get_arf
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttWCSHistmean, AttWCSHistinteg, convolve_profile, AttInvHist, make_small_steps_quats, make_wcs_steps_quats
from .energy  import get_arf_energy_function
from .filters import Intervals, IndependentFilters, RationalSet
from .orientation import get_photons_sky_coord, quat_to_pol_and_roll, ra_dec_roll_to_quat
from .containers import Urddata
from .vector import vec_to_pol, pol_to_vec
from .aux import interp1d
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from .vignetting import make_vignetting_for_urdn
from ._det_spatial import DL, dxya, offset_to_vec, vec_to_offset, vec_to_offset_pairs, raw_xy_to_vec, vec_to_offset_pairs, offset_to_raw_xy, rawxy_to_qcorr
from .psf import get_pix_overall_countrate_constbkg_ayut, urddata_to_opaxoffset, photbkg_pix_coeff, xy_to_opaxoffset
from .mosaic2 import SkyImage
from .telescope import URDNS
from functools import reduce
from scipy.spatial.transform import Rotation
from multiprocessing import cpu_count, Pool
from multiprocessing.pool import ThreadPool
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
import matplotlib.pyplot as plt
from copy import copy
from math import pi, sin, cos, sqrt
from tqdm import tqdm

MPNUM = cpu_count()

def make_background_det_map_for_urdn(urdn, imgfilter=None):
    """
    for specified urdn and eventfilter provides RegularGridInterpolator of backgroud profile
    """
    bkgprofile = np.ones((48, 48), np.double)/48.**2. if imgfilter is None else get_background_surface_brigtnress(urdn, imgfilter, normalize=True, fill_value=0.)
    bkgmap = RegularGridInterpolator(((np.arange(-24, 24) + 0.5)*DL,
                                      (np.arange(-24, 24) + 0.5)*DL),
                                        bkgprofile,
                                        method="nearest", bounds_error=False, fill_value=0)
    return bkgmap

def make_overall_background_map(subgrid=10, useshadowmask=True):
    """
    produces 2d interpolator for the backgroudn with subresolution
    """
    xmin, xmax = -24.5*DL, 24.5*DL
    ymin, ymax = -24.5*DL, 24.5*DL

    vecs = offset_to_vec(np.array([xmin, xmax, xmax, xmin]),
                         np.array([ymin, ymin, ymax, ymax]))
    vmaps = {}
    for urdn in URDNS:
        quat = get_boresight_by_device(urdn)
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
        quat = get_boresight_by_device(urdn)
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs))).reshape(shape)

    bkgmap = RegularGridInterpolator((x[:, 0], y[0]), newvmap, bounds_error=False, fill_value=0)
    return bkgmap

def make_bkgmap_for_wcs(wcs, attdata, urdbkg, imgfilters, shape=None, illuminations=None, mpnum=MPNUM, time_corr={}, kind="direct"):
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
        urdgtis - a dict of the form {urdn: arttools.time.GTI ...} UPDATE urdgti is now a part of imgfilters, stored in TIME extention
        mpnum - num of the processort to use in multiprocessing computation
        time_corr - a dict containing functions {urdn: urdnbkgrate(time) ...}
        subscale - defined a number of subpixels (under detecto pixels) to interpolate bkgmap
    """
    bkgmap = make_background_det_map_for_urdn(None)
    sky = SkyImage(wcs, bkgmap, mpnum=mpnum)
    overall_gti = emptyGTI

    for urdn in imgfilters:
        gti = imgfilters[urdn].filters["TIME"] & ~overall_gti
        if gti.size == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        #tc, qval, exptime, ugti = make_wcs_steps_quats(wcs, attdata*get_boresight_by_device(urdn), gti, urdbkg[urdn])
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti, wcs=wcs, timecorrection=urdbkg[urdn])
        bkgprofile = get_background_surface_brigtnress(urdn, imgfilters[urdn].filters, fill_value=0., normalize=True)
        bkgmap = make_background_det_map_for_urdn(urdn, imgfilters[urdn].filters)
        sky.set_vmap(bkgmap)
        if kind == "direct":
            sky.direct_convolve(qval, exptime)
        elif kind == "convolve":
            sky.fft_convolve(qval, exptime)
        print("done!")

    if wcs.wcs.has_cd():
        scale = np.linalg.det(wcs.wcs.cd)/dxya**2.
    else:
        scale = wcs.wcs.cdelt[0]*wcs.wcs.cdelt[1]/dxya**2.
    return np.copy(sky.img)*scale


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

def get_background_surface_brigtnress(urdn, filters, fill_value=np.nan, normalize=False):
    """
    provides count rate in each pixel of48x48 detector for specified by the urdn detector and events selection filter
    signature:
        urdn, fitler, fill_value[nan], normalize[False]

    returns 48x48 2d array, which contains count rates in pixel if normalize is False,
    if normalize is True then koeficcient is applied in order to sum of array became 1
    """
    grid, datacube = get_background_for_urdn(urdn)
    menergy = filters["ENERGY"].apply(grid["ENERGY"])
    menergys = np.logical_and(menergy[1:], menergy[:-1])
    mgrade = filters["GRADE"].apply(grid["GRADE"])
    profile = datacube[:, :, menergys, :][:, :, :, mgrade].sum(axis=(2, 3))
    y, x = np.meshgrid(grid["RAW_Y"], grid["RAW_X"])
    #shmask = filters.apply(np.column_stack([x.ravel(), y.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
    shmask = filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    profile[~shmask] = fill_value
    return profile/np.sum(profile[~np.isnan(profile)]) if normalize else profile

def get_local_bkgrates(udata, bkglc):
    profile = get_background_surface_brigtnress(udata.urdn, udata.filters, 0., True)
    return bkglc(udata["TIME"])*profile[udata["RAW_X"], udata["RAW_Y"]]

def get_background_spectrum(filters):
    grid, datacube = get_overall_background()
    menergy = filters.filters["ENERGY"].apply(grid["ENERGY"])
    menergys = np.logical_and(menergy[1:], menergy[:-1])
    mgrade = filters["GRADE"].apply(grid["GRADE"])
    rgrid = {"ENERGY": grid["ENERGY"][menergy], "GRADE": grid["GRADE"][mgrade]}
    y, x = np.meshgrid(grid["RAW_Y"], grid["RAW_X"])
    #shmask = filters.apply(np.column_stack([x.ravel(), y.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
    shmask = filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    return rgrid, (datacube[:, :, menergys, :][:, :, :, mgrade]*shmask[:, :, np.newaxis, np.newaxis]).sum(axis=(0, 1))

def get_background_events_weight(filters, udata):
    grid, spec = get_background_spectrum(filters)
    gidx = np.zeros(30, int)
    gidx[grid["GRADE"]] = np.arange(grid["GRADE"].size)
    idxe = np.searchsorted(grid["ENERGY"], udata["ENERGY"]) - 1
    idxg = gidx[udata["GRADE"]]
    return spec[idxe, idxg]/np.sum(spec)


def get_particle_and_photon_templates(filters, cspec=None):
    gridp, specp = get_crabspec_for_filters(filters)
    specp = (specp/np.diff(gridp["ENERGY"])[:, np.newaxis])/specp.sum()
    if not cspec is None:
        arf = get_arf_energy_function(get_arf())
        spec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(gridp["ENERGY"][:-1], gridp["ENERGY"][1:])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        spec = spec/spec.sum()
        specp = specp*(spec/specp.sum(axis=1))[:, np.newaxis]
        specp = specp/specp.sum()

    gridb, specb = get_background_spectrum(filters)
    specb = (specb/np.diff(gridb["ENERGY"])[:, np.newaxis])/specb.sum()
    return gridp, specp, specb


def get_full_photbkgrate(wcs, photbkgmap, imgfilters, attdata, te):
    from scipy.signal import convolve
    #import matplotlib.pyplot as plt
    ra, dec, roll = quat_to_pol_and_roll(attdata(attdata.times))
    roll = np.median(roll)

    qc = ra_dec_roll_to_quat(*np.array([wcs.wcs.crval[0], wcs.wcs.crval[1], roll*pi/180.]).reshape((3, 1)))
    lclist = {}
    for urdn in imgfilters:
        vmap = make_vignetting_for_urdn(urdn, imgfilters[urdn])
        vlist = offset_to_vec(vmap.grid[0][[0, 0, -1, -1]], vmap.grid[1][[0, -1, -1, 0]])
        xy = (wcs.all_world2pix(np.rad2deg(vec_to_pol(qc.apply(vlist))).T, 0) + 0.5).astype(int)
        xmin, xmax, ymin, ymax = np.sort(xy, axis=0)[[0, -1, 0, -1], [0, 0, 1, 1]]
        xm, ym = np.mgrid[xmin:xmax:1, ymin:ymax:1]
        rdm = wcs.all_pix2world(np.array([xm.ravel(), ym.ravel()]).T, 0)
        rdm = pol_to_vec(*np.deg2rad(rdm).T)
        core = vmap(vec_to_offset(qc.apply(rdm, inverse=True))).reshape((xmax - xmin, ymax - ymin))

        cmap = convolve(photbkgmap, core)
        #return cmap
        tel, gaps, locgti = make_small_steps_quats(attdata, imgfilters[urdn].filters['TIME'] & GTI(te[[0, -1]]))
        dte = np.diff(tel)[gaps]
        tcl = (tel[1:] + tel[:-1])[gaps]/2.
        xy = (wcs.all_world2pix(np.rad2deg(vec_to_pol(attdata.for_urdn(urdn)(tcl).apply([1, 0, 0]))).T, 0) + 0.5).astype(int)[:, ::-1]
        cs = np.zeros(te.size - 1, float)
        dts = np.zeros(te.size - 1, float)
        idx = np.searchsorted(te, tcl) - 1
        np.add.at(cs, idx, cmap[xy[:, 0], xy[:, 1]]*dte)
        np.add.at(dts, idx, dte)
        lclist[urdn] = cs/dts
    return lclist


def get_photon_and_particles_rates(urddata, cspec=None):
    """
    if cspec is not provided returns a ratio between particle and crab shaped spectrum for each grade and energy
    """

    gridp, specp = get_crabspec_for_filters(urddata.filters)
    specp = (specp/np.diff(gridp["ENERGY"])[:, np.newaxis])/specp.sum()
    if not cspec is None:
        arf = get_arf_energy_function(get_arf())
        spec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(gridp["ENERGY"][:-1], gridp["ENERGY"][1:])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        spec = spec/spec.sum()
        specp = specp*(spec/specp.sum(axis=1))[:, np.newaxis]
        specp = specp/specp.sum()

    gridb, specb = get_background_spectrum(urddata.filters)
    specb = (specb/np.diff(gridb["ENERGY"])[:, np.newaxis])/specb.sum()

    eidx = np.searchsorted(gridb["ENERGY"], urddata["ENERGY"]) - 1
    gidx = np.zeros(30, int)
    gidx[gridb["GRADE"]] = np.arange(gridb["GRADE"].size)
    bweights = specb[eidx, gidx[urddata["GRADE"]]]

    eidx = np.searchsorted(gridp["ENERGY"], urddata["ENERGY"]) - 1
    gidx = np.zeros(30, int)
    gidx[gridp["GRADE"]] = np.arange(gridb["GRADE"].size)
    return specp[eidx, gidx[urddata["GRADE"]]], bweights

def get_photon_to_particle_rate_ratio(urddata, cspec=None):
    prate, brate = get_photon_and_particles_rates(urddata, cspec)
    return prate/brate

def get_photon_vs_particle_prob(udata, urdweights={}, cspec=None):
    pweights = {}
    for urdn in udata:
        pweights[urdn] = get_photon_to_particle_rate_ratio(udata[urdn], cspec)*urdweights.get(urdn, 1/7.)
    return pweights

def get_background_lightcurve(tevts, bkgfilters, timebin, imgfilters=None, dtcorr={}):
    """
    returns an approximation to the background lightcurve for the specified data filters
    """
    bkgprofiles = {urdn: get_background_surface_brigtnress(urdn, bkgfilters[urdn].filters, fill_value=0.) for urdn in bkgfilters}
    urdgti = {urdn: f.filters["TIME"] for urdn, f in bkgfilters.items()}

    tgti = reduce(lambda a, b: a | b, urdgti.values())

    te, gaps = tgti.arange(timebin)
    ii = np.array([te[:-1], te[1:]]).T[gaps]
    ffun = tgti.get_interpolation_function()

    tweights = np.sum(list(bkgprofiles.values()))/len(bkgprofiles)
    bkgscales = {urdn: np.sum(d)/tweights for urdn, d in bkgprofiles.items()}

    tlive = sum((dtcorr.get(urdn, ffun)*urdgti[urdn].get_interpolation_function(bkgscales[urdn])).integrate_in_intervals(ii) for urdn in bkgfilters)
    tt = np.concatenate([te[:1], te[1:][gaps]])
    brate = np.diff(np.searchsorted(tevts, te))[gaps]/tlive

    urdbkg = {urdn: interp1d(tt, np.concatenate([[0,], brate*bkgscales[urdn]]), kind="next", bounds_error=False, fill_value=tuple(brate[[0, -1]]))*dtcorr.get(urdn, ffun) for urdn in bkgfilters}
    """
    estimated mean background rate for background in the desired parameters space (energy, grade, coordinates)
    """
    if not imgfilters is None:
        urdbkg = {urdn: lc._scale(get_background_bands_ratio(imgfilters[urdn].filters, bkgfilters[urdn].filters)) for urdn, lc in urdbkg.items() if urdn in imgfilters}
    return urdbkg

def get_bkg_lightcurve_for_app(urdbkg, filters, att, ax, app=120., te=np.array([-np.inf, np.inf]), dtcorr={}, locphotbkg=0., urdweights={}, illum_filters=None, cspec=None):
    cgti = att.circ_gti(ax, 25.*60.)
    gti = reduce(lambda a, b: a | b, [f.filters["TIME"] & Intervals(te[[0, -1]]) & cgti for f in filters.values()])

    tel, gaps, locgti = make_small_steps_quats(att, gti, tedges=te)

    cosa = cos(app*pi/180./3600.)
    lcs = np.zeros(te.size - 1, np.double)
    xd, yd = np.mgrid[0:48:1, 0:48:1]
    vecs = raw_xy_to_vec(xd.ravel(), yd.ravel()).reshape((48, 48, 3))

    shiftsize = int(min(app, 300)//45 + 1)
    print("shiftsize", shiftsize)
    xc, yc = np.mgrid[-shiftsize: shiftsize + 1: 1, -shiftsize: shiftsize + 1: 1] # size of the pixel is 45 arcsec

    for urdn in filters:
        if filters[urdn].filters["TIME"].arr.size == 0:
            continue

        lgti = filters[urdn].filters["TIME"] & locgti
        teu, gaps = lgti.make_tedges(tel)
        tc = (teu[1:] + teu[:-1])[gaps]/2.
        dtu = np.diff(teu)[gaps]
        qlist = att.for_urdn(urdn)(tc)
        idx = np.searchsorted(te, tc) - 1
        shmask = filters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        x, y = xd[shmask], yd[shmask]

        cbkg = get_background_surface_brigtnress(urdn, filters[urdn].filters, fill_value=0., normalize=True)
        rl = (urdbkg[urdn]*lgti.get_interpolation_function()).integrate_in_intervals(np.array([teu[:-1], teu[1:]]).T[gaps])
        pbkg = photbkg_pix_coeff(urdn, filters[urdn].filters, cspec)*urdweights.get(urdn, 1/7.)*locphotbkg

        vsrc = qlist.apply(ax, inverse=True)

        xs, ys = offset_to_raw_xy(*vec_to_offset(vsrc))
        xs, ys = xs[:, np.newaxis] + xc.ravel()[np.newaxis, :], ys[:, np.newaxis] + yc.ravel()[np.newaxis, :]
        mask = np.logical_and.reduce([xs > -1, xs < 48, ys > -1, ys < 48])
        mask[mask] = shmask[xs[mask], ys[mask]]
        cvals = mask.sum(axis=1)
        xs, ys = xs[mask], ys[mask]
        mask = np.sum(vecs.reshape((48, 48, 3))[xs, ys]*np.repeat(vsrc, cvals, axis=0), axis=1) > cosa
        xs, ys = xs[mask], ys[mask]
        cu, cc = np.unique(np.repeat(np.arange(cvals.size), cvals)[mask], return_counts=True)
        cvals[:] = 0
        cvals[cu] = cc

        qlist = Rotation(np.repeat(qlist.as_quat(), cvals, axis=0))
        if not illum_filters is None:
            mask = ~illum_filters.check_pixel_in_illumination(urdn, xs, ys, qlist, vecs.reshape((48, 48, 3))[xs, ys])
            cu, cc = np.unique(np.repeat(np.arange(cvals.size), cvals)[mask], return_counts=True)
            cvals[:] = 0
            cvals[cu] = cc
            xs, ys = xs[mask], ys[mask]

        if np.sum(cvals) > 0:
            idx = np.searchsorted(te, np.repeat(tc, cvals)) - 1
            np.add.at(lcs, idx, np.repeat(dtu, cvals)*pbkg[xs, ys] + np.repeat(rl, cvals)*cbkg[xs, ys])
    return lcs


def get_photbkg_lightcurve_for_app(locphotbkg, att, ax, app, te, filters, dtcorr={}, urdweights={}, cspec=None, illum_filters=None):

    cgti = att.circ_gti(ax, 25.*60.)
    gti = reduce(lambda a, b: a | b, [f.filters["TIME"] & Intervals(te[[0, -1]]) & cgti for f in filters.values()])

    tel, gaps, locgti = make_small_steps_quats(att, gti, tedges=te)
    tc = (tel[1:] + tel[:-1])[gaps]/2.

    if app is None:
        app= 300.

    cosa = cos(app*pi/180./3600.)
    lcs = np.zeros(te.size - 1, np.double)
    xd, yd = np.mgrid[0:48:1, 0:48:1]
    vecs = raw_xy_to_vec(xd.ravel(), yd.ravel()).reshape((48, 48, 3))

    shiftsize = int(min(app, 300)//45 + 1)
    print("shiftsize", shiftsize)
    xc, yc = np.mgrid[-shiftsize: shiftsize + 1: 1, -shiftsize: shiftsize + 1: 1] # size of the pixel is 45 arcsec

    for urdn in filters:
        dtn = np.diff(te)*(dtcorr[urdn](tc) if urdn in dtcorr else 1.) #timecorrection(tc)
        if filters[urdn].filters["TIME"].arr.size == 0:
            continue

        teu, gaps = (filters[urdn].filters["TIME"] & locgti).make_tedges(tel)
        tc = (teu[1:] + teu[:-1])[gaps]/2.
        dtu = np.diff(teu)[gaps]
        idx = np.searchsorted(te, tc) - 1


        pr = photbkg_pix_coeff(urdn, filters[urdn].filters, cspec)*urdweights.get(urdn, 1/7.)

        shmask = filters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        vsrc = qlist.apply(ax, inverse=True)

        xs, ys = offset_to_raw_xy(*vec_to_offset(vsrc))
        xs, ys = xs[:, np.newaxis] + xc.ravel()[np.newaxis, :], ys[:, np.newaxis] + yc.ravel()[np.newaxis, :]
        mask = np.logical_and.reduce([xs > -1, xs < 48, ys > -1, ys < 48])
        mask[mask] = shmask[xs[mask], ys[mask]]
        cvals = mask.sum(axis=1)
        xs, ys = xs[mask], ys[mask]
        mask = np.sum(vecs.reshape((48, 48, 3))[xs, ys]*np.repeat(vsrc, cvals, axis=0), axis=1) > cosa
        xs, ys = xs[mask], ys[mask]
        cu, cc = np.unique(np.repeat(np.arange(cvals.size), cvals)[mask], return_counts=True)
        cvals[:] = 0
        cvals[cu] = cc

        qlist = Rotation(np.repeat(qlist.as_quat(), cvals, axis=0))
        if not illum_filters is None:
            mask = ~illum_filters.check_pixel_in_illumination(urdn, xs, ys, qlist, vecs.reshape((48, 48, 3))[xs, ys])
            cu, cc = np.unique(np.repeat(np.arange(cvals.size), cvals)[mask], return_counts=True)
            cvals[:] = 0
            cvals[cu] = cc
            xs, ys = xs[mask], ys[mask]

        if np.sum(cvals) > 0:
            idx = np.searchsorted(te, np.repeat(tc, cvals)) - 1
            dtu = np.repeat(dtu, cvals)
            np.add.at(lcs, idx, dtu*locphotbkg*pr[xs, ys])
    return lcs

def get_bkg_spec(urdbkg, filters, att, ax, appsize, dtcorr={}, illum_filters=None, mpnum=MPNUM):
    tfilt = reduce(lambda a, b: a|b, filters.values())
    filters = {urdn: copy(f) for urdn, f in filters.items()}

    gti = reduce(lambda a, b: a | b, [f.filters["TIME"] for f in filters.values()])

    tel, gaps, locgti = make_small_steps_quats(att, gti, tedges=te)
    tc = (tel[1:] + tel[:-1])[gaps]/2.
    qval = attdata(tc)
    dtn = np.diff(te)[gaps]*timecorrection(tc)

    cosa = cos(appsize*pi/180./3600.)
    bspec = 0.
    xd, yd = np.mgrid[0:48:1, 0:48:1]
    for urdn in filters:
        if filters[urdn]["TIME"].arr.size == 0:
            continue

        grid, bcube = get_background_for_urdn(urdn)
        mgrade = filters[urdn].get("GRADE", Intervals([-np.inf, np.inf])).apply(grid["GRADE"])
        bloc = bcube[:, :, :, mgrade].sum(axis=3)
        menerg = filters[urdn].get("ENERGY", Intervals([-np.inf, np.inf])).apply((grid["ENERGY"][1:] + grid["ENERGY"][:-1])/2.)
        bloc = bloc[:, :, menerg]
        shmask = filters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        bloc = bloc[shmask, :]
        bloc = bloc/bloc.sum()
        print("background pix specs formed", bloc.shape)

        qlist = qval*get_boresight_by_device(urdn)
        x, y = xd[shmask], yd[shmask]

        teu, gaps = filters[urdn].filters["TIME"].make_tedges(tel)
        tc = (teu[1:] + teu[:-1])[gaps]/2.

        rl = urdbkg[urdn].integrate_in_intervals(np.array([teu[:-1], teu[1:]]).T[gaps]) #teu, dtcorr.get(urdn, None))[gaps]
        vec = raw_xy_to_vec(x, y)

        if not illum_filters is None:
            raise NotImpementedError("this is not written yet")
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for source in illum_filters.sources:
                source.setup_for_quats(qlist, opax)
            m1 = ~np.any([source.mask_vecs_with_setup(vec, qlist) for source in illum_filters.sources], axis=0)
            m2 = np.array([(np.sum(vec*q.apply(ax, inverse=True), axis=1) > cosa) for q in qlist]).T
            lloc = np.sum(np.logical_and(m1, m2)*pr[:, np.newaxis], axis=0)*rl
        else:
            def get_pixels_exposures(v):
                return rl[np.sum(v*qlist.apply(ax, inverse=True), axis=1) > cosa].sum()
            pool = Pool(mpnum)
            bspec += sum((s*b for s, b in tqdm(zip(pool.imap(get_pixels_exposures, vec), bloc), total=bloc.shape[0])))
    return ee, gaps, bspec


def get_background_bands_ratio(filters1, filters2=IndependentFilters({"ENERGY": Intervals([0, np.inf]), "GRADE": RationalSet(range(16))})):
    grid1, spec1 = get_background_spectrum(filters1)
    grid2, spec2 = get_background_spectrum(filters2)
    return np.sum(spec1)/np.sum(spec2)

def make_mock_photbkg(urdn, filters, attdata, photbkg, randomize=True):
    pfun = get_pix_overall_countrate_constbkg_ayut(filters.filters) #, False)
    shmask = filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    x, y = np.mgrid[0:48:1, 0:48:1]
    x, y = x[shmask], y[shmask]
    te, gaps, lgti = make_small_steps_quats(attdata.for_urdn(urdn))
    tc = (te[1:] + te[:-1])[gaps]/2.
    dt = np.diff(te)[gaps]
    print(tc.min(), tc.max(), dt.min())
    qloc = attdata.for_urdn(urdn)(tc)

    gridp, specp = get_crabspec_for_filters(filters)
    rat, dect = [], []
    qcorr = rawxy_to_qcorr(x, y)

    for xp, yp, qpix in zip(x, y, qcorr):
        i, j = xy_to_opaxoffset(xp, yp, urdn)
        #print("ij", i, j)
        ploc = pfun(i, j)
        ra, dec = np.rad2deg(vec_to_pol((qloc*qpix).apply([1, 0, 0])))
        #print(ra, dec)
        prates = photbkg(ra, dec)
        #print(prates, prates*ploc*dt)
        #print(np.random.poisson(prates*ploc*dt))
        time = np.repeat(tc, np.random.poisson(prates*ploc*dt))
        #print(time.min(), time.max(), time.size)
        if time.size == 0:
            continue
        if randomize:
            xpix, ypix = np.random.uniform(-0.5, 0.5, (2, time.size)) + 23.5
            ra, dec = np.rad2deg(vec_to_pol((attdata.for_urdn(urdn)(time)*qpix).apply(raw_xy_to_vec(xpix, ypix))))
        else:
            ra, dec = np.rad2deg(vec_to_pol((attdata.for_urdn(urdn)(time)*qpix).apply([1, 0, 0])))
        rat.append(ra)
        dect.append(dec)
    ra = np.concatenate(rat)
    dec = np.concatenate(dect)
    return ra, dec


def make_mock_data(urdn, bkglc, imgfilter, cspec=None, photbkgrate=None):

    gti = imgfilter["TIME"]
    te, gaps = gti.make_tedges(bkglc.x)
    tc = (te[1:] + te[:-1])[gaps]/2.
    totcts = bkglc(tc)*np.diff(te)[gaps]
    grid, datacube = get_background_for_urdn(urdn)
    keys = ["RAW_Y", "RAW_X", "ENERGY", "GRADE"]
    phase = np.meshgrid(*[grid[k] if k != "ENERGY" else (grid[k][1:] + grid[k][:-1])/2. for k in keys])
    m = imgfilter.meshgrid(keys, [grid[k] if k != "ENERGY" else (grid[k][1:] + grid[k][:-1])/2. for k in keys])
    positions = {k:a[m[:, :, :, :]] for k, a in zip(keys, phase)}
    data = datacube[m[:, :, :, :]]
    dvol = data.cumsum()
    totevents = np.random.poisson(totcts.sum())
    position = np.random.uniform(0., dvol[-1], totevents)
    time = np.random.uniform(0., totcts.sum(), totevents)
    dt = np.random.uniform(0., 1., totevents)
    events = np.empty(totevents, [("TIME", float), ("RAW_X", int), ("RAW_Y", int), ("ENERGY", float), ("GRADE", int)])
    idx = np.searchsorted(dvol, position)
    events["ENERGY"] = positions["ENERGY"][idx]
    events["RAW_X"] = positions["RAW_X"][idx]
    events["RAW_Y"] = positions["RAW_Y"][idx]
    events["GRADE"] = positions["GRADE"][idx]
    tidx = np.searchsorted(totcts.cumsum(), time)
    events["TIME"] = te[:-1][gaps][tidx] + dt*np.diff(te)[gaps][tidx]
    return Urddata(events, urdn, imgfilter)


def make_bkgmock_img(locwcs, urdn, bkglc, imgfilter, gti, shape=None, scale=1, mpnum=10):
    shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
    tmpimg = [np.zeros(shape, int) for _ in range(mpnum)]
    X, Y = np.mgrid[0:48:1, 0:48:1]
    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter.filters) #get_backprofile_by_urdn(urdn)
    shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])



def photbkgrate(wcs, pbkgrmap, urddata, attdata, weight=1., app=None, cspec=None):
    """
    assuming a constant(in time) and spline(which is almost does not changes on the PSF scales) photon background
    provide with expected for the energy count rate ratio to background
    """
    pfun = get_pix_overall_countrate_constbkg_ayut(urddata.filters, cspec, app)
    i, j = urddata_to_opaxoffset(urddata, urddata.urdn)
    radec = np.rad2deg(get_photons_sky_coord(urddata, urddata.urdn, attdata))
    xy = (wcs.all_world2pix(radec.T, 1) - 0.5).astype(int)[:, ::-1]
    prates = pbkgrmap[xy[:, 0], xy[:, 1]]*weight*pfun(i, j)
    return prates
