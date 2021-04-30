from .background import get_background_surface_brigtnress
from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAXOFFSET
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from .energy  import get_arf_energy_function
from .caldb import  get_optical_axis_offset_by_device, get_arf
from .psf import xy_to_opaxoffset, unpack_inverse_psf_ayut
from scipy.spatial.transform import Rotation
from .mosaic import SkyImage
from astopy.wcs import WCS



@lru_cache(maxsize=7)
def make_firstorder_detstat(urdn, imgfilter, phot_index=2., egrid=None, cspec=None, app=None):
    arf = get_arf_energy_function(get_arf())
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False


    emin, emax = imgfilter["ENERGY"].arr
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

    bkgprofile = get_background_surface_brigtnress(urdn, imgfilter)

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
        sl += lipsf**2./bkgprofile[xl, yl]


def make_overall_detstat_estimation(immgfilter, rebinres, **kwargs):
    wcs = WCS(NAXIS=2)
    wcs
    sky

def make_sensitivity_map(wcs, attdata, urdgtis, shape=None, mpnum=MPNUM, imgfilters, urdbkg, kind="direct", **kwargs):
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
    print(wcs)
    print(shape)

    sky = SkyImage(wcs, shape=shape)

    if dtcorr:
        overall_gti = emptyGTI
    else:
        overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
        print(overall_gti.exposure)
        if overall_gti.exposure > 0:
            exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti)
            vmap = make_overall_vignetting(**kwargs)
            print("exptime sum", exptime.sum())
            print("produce overall urds expmap")
            sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
            if kind == "direct":
                sky.interpolate_mp(qval[:], exptime[:], mpnum)
            elif kind == "convolve":
                sky.convolve(qval, exptime, mpnum)
            #emap = AttInvHist.make_mp(wcs, vmap, exptime, qval, mpnum)
            #emap = make_mosaic_expmap_mp_executor(shape, wcs, vmap, qval, exptime, mpnum)
            print("\ndone!")

    for urd in urdgtis:
        gti = urdgtis[urd] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urd), gti, \
                                                             dtcorr.get(urd, lambda x: 1))
        vmap = make_vignetting_for_urdn(urd, **kwargs)
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        #emap = make_mosaic_expmap_mp_executor(shape, wcs, vmap, qvals, exptime, mpnum) + emap
        #emap = AttInvHist.make_mp(wcs, vmap, exptime, qval, mpnum) + emap
        print(" done!")
    return sky.img

