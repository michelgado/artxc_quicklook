from .caldb import get_boresight_by_device, get_backprofile_by_urdn, get_shadowmask_by_urd, \
                    make_background_brightnes_profile, get_background_for_urdn, get_overall_background, \
                    get_crabspec_for_filters, get_optical_axis_offset_by_device, get_arf
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttWCSHistmean, AttWCSHistinteg, convolve_profile, AttInvHist, make_small_steps_quats, make_wcs_steps_quats
from .energy  import get_arf_energy_function
from .orientation import get_photons_sky_coord
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from ._det_spatial import DL, dxya, offset_to_vec, vec_to_offset, vec_to_offset_pairs, raw_xy_to_vec, vec_to_offset_pairs
from .psf import get_pix_overall_countrate_constbkg_ayut, select_psf_grups, urddata_to_opaxoffset
from .lightcurve import make_overall_lc, Bkgrate
from .mosaic import SkyImage
from .telescope import URDNS
from functools import reduce
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
from scipy.integrate import quad
import matplotlib.pyplot as plt
from math import pi, sin, cos, sqrt

MPNUM = cpu_count()

def make_background_det_map_for_urdn(urdn, imgfilter):
    """
    for specified urdn and eventfilter provides RegularGridInterpolator of backgroud profile
    """
    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter, normalize=True, fill_value=0.) #get_backprofile_by_urdn(urdn)
    bkgmap = RegularGridInterpolator(((np.arange(-24, 24) + 0.5)*DL,
                                      (np.arange(-24, 24) + 0.5)*DL),
                                        bkgprofile,
                                        method="nearest", bounds_error=False, fill_value=0)
    return bkgmap

def make_overall_background_map(subgrid=10, useshadowmask=True):
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

def make_bkgmap_for_wcs(wcs, attdata, urdgtis, urdbkg, imgfilters, shape=None, illuminations=None, mpnum=MPNUM, time_corr={}, kind="stright"):
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

    sky = SkyImage(wcs)
    overall_gti = emptyGTI
    #X, Y = np.mgrid[0:48:1, 0:48:1]
    #vecs = raw_xy_to_vec(X.ravel(), Y.ravel())
    #shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
    #imgpool = [np.zeros(shape) for i in range(mpnum)]

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.size == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        tc, qval, exptime, ugti = make_wcs_steps_quats(wcs, attdata*get_boresight_by_device(urdn), gti, urdbkg[urdn])
        #exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti, urdbkg[urdn])
        print("processed exposure", gti.exposure) #, exptime.sum())
        bkgprofile = get_background_surface_brigtnress(urdn, imgfilters[urdn], fill_value=0., normalize=True)
        #shmask = filters.apply(np.column_stack([x.ravel(), y.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
        shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        bkgmap = make_background_det_map_for_urdn(urdn, imgfilters[urdn])
        sky._set_core(bkgmap.grid[0], bkgmap.grid[1], bkgmap.values)
        #bkg = AttWCSHistmean.make_mp(bkgmap, exptime, qval, wcs, mpnum, subscale=subscale) + bkg
        #bkg = AttInvHist.make_mp(wcs, bkgmap, exptime, qval,  mpnum) + bkg
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
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
    grid, datacube = get_background_for_urdn(urdn)
    menergy = filters["ENERGY"].apply(grid["ENERGY"])
    menergys = np.logical_and(menergy[1:], menergy[:-1])
    mgrade = filters["GRADE"].apply(grid["GRADE"])
    profile = datacube[:, :, menergys, :][:, :, :, mgrade].sum(axis=(2, 3))
    y, x = np.meshgrid(grid["RAW_Y"], grid["RAW_X"])
    #shmask = filters.apply(np.column_stack([x.ravel(), y.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
    shmask = filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    profile[~shmask] = fill_value
    return profile/profile.sum() if normalize else profile

def get_local_bkgrates(urdn, bkglc, urdfilter, udata):
    profile = get_background_surface_brigtnress(urdn, urdfilter, 0., True)
    return bkglc(udata["TIME"])*profile[udata["RAW_X"], udata["RAW_Y"]]

def get_background_spectrum(filters):
    grid, datacube = get_overall_background()
    menergy = filters["ENERGY"].apply(grid["ENERGY"])
    menergys = np.logical_and(menergy[1:], menergy[:-1])
    mgrade = filters["GRADE"].apply(grid["GRADE"])
    rgrid = {"ENERGY": grid["ENERGY"][menergy], "GRADE": grid["GRADE"][mgrade]}
    y, x = np.meshgrid(grid["RAW_Y"], grid["RAW_X"])
    #shmask = filters.apply(np.column_stack([x.ravel(), y.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
    shmask = filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    return rgrid, (datacube[:, :, menergys, :][:, :, :, mgrade]*shmask[:, :, np.newaxis, np.newaxis]).sum(axis=(0, 1))

def get_background_events_weight(filters, udata):
    grid, spec = get_background_spectrum(filters)
    gidx = np.zeros(30, np.int)
    gidx[grid["GRADE"]] = np.arange(grid["GRADE"].size)
    idxe = np.searchsorted(grid["ENERGY"], udata["ENERGY"]) - 1
    idxg = gidx[udata["GRADE"]]
    return spec[idxe, idxg]/np.sum(spec)

def get_photon_vs_particle_prob(urdfilters, udata, urdweights={}, cspec=None):
    pweights = {}
    for urdn in udata:
        filters = urdfilters[urdn]

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

        eidx = np.searchsorted(gridb["ENERGY"], udata[urdn]["ENERGY"]) - 1
        gidx = np.zeros(30, np.int)
        gidx[gridb["GRADE"]] = np.arange(gridb["GRADE"].size)
        bweights = specb[eidx, gidx[udata[urdn]["GRADE"]]]

        eidx = np.searchsorted(gridp["ENERGY"], udata[urdn]["ENERGY"]) - 1
        gidx = np.zeros(30, np.int)
        gidx[gridp["GRADE"]] = np.arange(gridb["GRADE"].size)
        pweights[urdn] = specp[eidx, gidx[udata[urdn]["GRADE"]]]/bweights*urdweights.get(urdn, 1)
    return pweights


def get_background_lightcurve(tevts, urdgti, bkgfilters, timebin, imgfilters=None, dtcorr={}):
    """
    get surface brightness profiless
    """
    bkgprofiles = {urdn: get_background_surface_brigtnress(urdn, bkgfilters, fill_value=0.) for urdn in urdgti}
    """
    estimate background count rates ratio to the mean overall countrate
    """
    tweights = np.sum(list(bkgprofiles.values()))/len(bkgprofiles)
    bkgscales = {urdn: np.sum(d)/tweights for urdn, d in bkgprofiles.items()}

    """
    estimated mean background rate for background in the desired parameters space (energy, grade, coordinates)
    """
    tebkg, mgapsbkg, cratebkg, crerrbkg, bkgrate = make_overall_lc(tevts, urdgti, timebin, bkgscales, dtcorr)
    urdbkg = {urdn: bkgrate._scale(v) for urdn, v in bkgscales.items()}
    if not imgfilters is None:
        urdbkg = {urdn: lc._scale(get_background_bands_ratio(imgfilters[urdn], bkgfilters)) for urdn, lc in urdbkg.items()}
    for urdn in dtcorr:
        urdbkg[urdn].set_dtcorr(dtcorr[urdn])
    return urdbkg

def get_bkg_lightcurve_for_app(urdbkg, urdgti, filters, att, ax, appsize, te, dtcorr={}, illum_filters=None):
    bkgprofiles = {urdn: get_background_surface_brigtnress(urdn, filters[urdn], fill_value=0.) for urdn in urdgti}
    bkgprofiles = {urdn: profile/profile.sum() for urdn, profile in bkgprofiles.items()}
    gti = reduce(lambda a, b: a | b, urdgti.values())
    ts, qval, dtq, locgti = make_small_steps_quats(att, gti, tedges=te)
    tel = np.empty(ts.size*2, np.double)
    tel[::2] = ts - dtq/2.
    tel[1::2] = ts + dtq/2.
    tel = np.unique(tel)
    cosa = cos(appsize*pi/180./3600.)
    lcs = np.zeros(te.size - 1, np.double)
    xd, yd = np.mgrid[0:48:1, 0:48:1]
    for urdn in urdgti:
        if urdgti[urdn].arr.size == 0:
            continue

        teu, gaps = urdgti[urdn].make_tedges(tel)
        tc = (teu[1:] + teu[:-1])[gaps]/2.
        qlist = att(tc)*get_boresight_by_device(urdn)
        #shmask = get_shadowmask_by_urd(urdn)
        shmask = filters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        x, y = xd[shmask], yd[shmask]
        pr = bkgprofiles[urdn][shmask]
        rl = urdbkg[urdn].integrate_in_timebins(teu, dtcorr.get(urdn, None))[gaps]
        vec = raw_xy_to_vec(x, y)

        if not illum_filters is None:
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for source in illum_filters.sources:
                source.setup_for_quats(qlist, opax)
            m1 = ~np.any([source.mask_vecs_with_setup(vec, qlist) for source in illum_filters.sources], axis=0)
            m2 = np.array([(np.sum(vec*q.apply(ax, inverse=True), axis=1) > cosa) for q in qlist]).T
            print("illumination and aperture mask", m1.shape, m2.shape, m1.sum(), m2.sum(), np.logical_and(m1, m2).sum())
            """
            lloc = np.sum((~np.any([source.mask_vecs_with_setup(vec, qlist) for source in illum_filters.sources], axis=0) &
                    np.array([(np.sum(vec*q.apply(ax, inverse=True), axis=1) > cosa) for q in qlist]).T)*pr[:, np.newaxis], axis=0)*rl
            """
            print("illumination", np.sum(np.sum(m1*pr[:, np.newaxis], axis=0)*rl))
            print("aperture", np.sum(np.sum(m2*pr[:, np.newaxis], axis=0)*rl))
            lloc = np.sum(np.logical_and(m1, m2)*pr[:, np.newaxis], axis=0)*rl
            print("overall sum", lloc.sum())
        else:
            lloc = np.array([pr[np.sum(vec*q.apply(ax, inverse=True), axis=1) > cosa].sum() for q in qlist])*rl
        idx = np.searchsorted(te, tc) - 1
        mloc = (idx >= 0) & (idx < te.size - 1)
        np.add.at(lcs, idx[mloc], lloc[mloc])
    return lcs

def get_background_bands_ratio(filters1, filters2):
    grid1, spec1 = get_background_spectrum(filters1)
    grid2, spec2 = get_background_spectrum(filters2)
    return np.sum(spec1)/np.sum(spec2)


def make_mock_data(urdn, bkglc, imgfilter, gti, cspec=None):
    te, gaps = gti.make_tedges(bkglc.te)
    tc = (te[1:] + te[:-1])[gaps]/2.
    totcts = bkglc(tc)*np.diff(te)[gaps]
    grid, datacube = get_background_for_urdn(urdn)
    keys = ["RAW_Y", "RAW_X", "ENERGY", "GRADE"]
    phase = np.meshgrid(*[grid[k] if k != "ENERGY" else (grid[k][1:] + grid[k][:-1])/2. for k in keys])
    m = imgfilter.meshgrid(keys, [grid[k] if k != "ENERGY" else (grid[k][1:] + grid[k][:-1])/2. for k in keys])
    positions = {k:a[m[:, :, :, :]] for k, a in zip(keys, phase)}
    print({k: p.size for k, p in positions.items()})
    data = datacube[m[:, :, :, :]]
    dvol = data.cumsum()
    print(dvol.size)
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
    return events


def make_bkgmock_img(locwcs, urdn, bkglc, imgfilter, gti, shape=None, scale=1, mpnum=10):
    shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
    tmpimg = [np.zeros(shape, int) for _ in range(mpnum)]
    X, Y = np.mgrid[0:48:1, 0:48:1]
    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter) #get_backprofile_by_urdn(urdn)
    shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])



def photbkgrate(wcs, pbkgrmap, urdn, urddata, attdata, imgfilter, weight=1.):
    """
    assuming a constant(in time) and spline(which is almost does not changes on the PSF scales) photon background
    provide with expected for the energy count rate ratio to background
    """
    pfun = get_pix_overall_countrate_constbkg_ayut(imgfilter)
    i, j = urddata_to_opaxoffset(urddata, urdn)
    #vecs = get_photons_vectors(urddata, URDN, attdata)
    radec = np.rad2deg(get_photons_sky_coord(urddata, urdn, attdata))
    xy = (wcs.all_world2pix(radec.T, 1) - 0.5).astype(int)[:, ::-1]
    prates = pbkgrmap[xy[:, 0], xy[:, 1]]*weight*pfun(i, j)
    return prates
