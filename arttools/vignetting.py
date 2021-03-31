from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAXOFFSET
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from .energy  import get_arf_energy_function
from ._det_spatial import offset_to_raw_xy, DL, F, \
    offset_to_vec, vec_to_offset_pairs, vec_to_offset, offset_to_qcorr
from .telescope import URDNS
from .interval import Intervals
from .caldb import get_boresight_by_device, get_inverse_psf, get_optical_axis_offset_by_device, get_arf
from .psf import xy_to_opaxoffset, unpack_inverse_psf, unpack_inverse_psf_ayut
import numpy as np
from math import log10, pi, sin, cos
from functools import lru_cache

TINY = 1e-15

rawvignfun = None

def load_raw_wignetting_function():
    """
    read original vignetting function which defines Effective area depending on offset  from optical axis and energy of incident photon

    return: scipy.itnerpolate.RegularGridInterpolator instance awaiting energy, xoffset and yoffset as arguments
    """
    global rawvignfun
    if rawvignfun is None:
        vignfile = get_vigneting_by_urd(28)
        x = 23.5 + np.tan(vignfile["Offset angles"].data["X"]*pi/180/60.)*F/DL
        y = 23.5 + np.tan(vignfile["Offset angles"].data["Y"]*pi/180/60.)*F/DL
        rawvignfun = RegularGridInterpolator((vignfile["7 arcmin PSF"].data["E"], x, y), vignfile["5 arcmin PSF"].data["EFFAREA"])
    return rawvignfun

def cutmask(mask):
    mx = mask.any(axis=0)
    my = mask.any(axis=1)
    def newfunc(val2d):
        val2d = val2d[:, mx]
        val2d = val2d[my, :]
        return val2d
    return newfunc

class PixInterpolator(object):
    def __init__(self, shadow_mask):
        self.values = shadow_mask

    def __call__(self, xy_offset_pair):
        rawx, rawy = offset_to_raw_xy(xy_offset_pair[0], xy_offset_pair[1])
        mask = np.all([rawx > -1, rawy > -1, rawx < 48, rawy < 48], axis=0)
        result = np.zeros(rawx.shape[0] if rawx.ndim == 1 else rawx.shape[:1])
        result[mask] = self.values[rawx[mask], rawy[mask]]
        return result

@lru_cache(maxsize=7)
def make_vignetting_from_inverse_psf(urdn):
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
        print(x0, y0)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False
    """
    ipsf = get_inverse_psf()
    nax1 = ipsf[1].header["NAXIS1"]
    nax2 = ipsf[1].header["NAXIS2"]
    """
    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)
    q = offset_to_qcorr(x1, y1)
    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    #xi, yi = np.mgrid[
    #for


    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        if (xl - x0 + 26) < 0 or (xl - x0 + 26) > 52 or (yl - y0 + 26) < 0 or (yl - y0 + 26) > 52:
            shmask[xl, yl] = False
            continue
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        """
        sl = img[int((xl - x0 + 23.)*9) + 60 - 60: int((xl - x0 + 23.)*9) + 60 + 61, int((yl - y0 + 23.)*9) + 60 - 60: int((yl - y0 + 23.)*9) + 60 + 61]
        sl += ipsf[1].data[int(np.round(xl + 0.5 - x0)) + 26, int(np.round(yl + 0.5 - y0)) + 26, : sl.shape[0], :sl.shape[1]]
        """
        sl += np.copy(unpack_inverse_psf(xo, yo))

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img/img.max(), bounds_error=False, fill_value=0.)

@lru_cache(maxsize=7)
def make_vignetting_from_inverse_psf_ayut(urdn, emin=4., emax=12., phot_index=2., app=None):
    arf = get_arf_energy_function(get_arf())
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
        print(x0, y0)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    eidx = np.searchsorted(ee, [emin, emax]) - [1, -1]
    eidx[0] = max(eidx[0], 0)
    eidx[1] = min(eidx[1], ee.size)
    print(eidx)
    eel = np.copy(ee[eidx[0]: eidx[1]])
    eel[0] = max(emin, eel[0])
    eel[-1] = min(emax, eel[-1])

    w = [quad(lambda e: arf(e)*e**-phot_index, emin, emax)[0] for emin, emax in zip(eel[:-1], eel[1:])]
    w = np.array(w)
    w = w/w.sum()
    print(w.size, w, eel, eidx)

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)

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
        sl += lipsf

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img/img.max(), bounds_error=False, fill_value=0.)

def make_vignetting_from_inverse_psf_ayut_cspec(urdn, egrid, cspec, app=None):
    arf = get_arf_energy_function(get_arf())
    if not urdn is None:
        shmask = get_shadowmask_by_urd(urdn)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
        print(x0, y0)
    else:
        x0, y0 = 23.5, 23.5
        shmask = np.ones((48, 48), np.bool)
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    print(cspec, type(cspec))
    if app is None:
        psfmask = None
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. > app**2./25.

    ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    egloc = np.unique(np.concatenate([egrid, ee]))
    ec = (egloc[1:] + egloc[:-1])/2.
    cspec = np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
    w = cspec[np.searchsorted(egrid, ec) - 1]*np.diff(egloc)
    eidx = np.searchsorted(ee, ec) - 1

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)

    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        lipsf = np.sum(unpack_inverse_psf_ayut(xo, yo)[eidx]*w[:, np.newaxis, np.newaxis], axis=0)
        if not app is None:
            lipsf[psfmask] = 0.
        sl += lipsf

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img/img.max(), bounds_error=False, fill_value=0.)


def make_vignetting_for_urdn(urdn, energy=None, phot_index=None,
                             emin=0, emax=np.inf, grid=None, cspec=None, app=None):
    """
    for provided urd number  energy or photon index provided 2d interpolation function (RegularGridInterpolator) defining the profile of the effective area depending on offset

    --------------
    Parameters:
        urdn:  num of urd (28, 22, 23, ... 30)
        energy: energy of incident at which to compute vignetting
        phot_index: photon index if provided used the vignetting maps are weighted with power law model
        useshadowmask: if True subroutine ignores pixels covered by collimator
        ignoreedgestrips: if True ignores 4 edges strips, which were shown to produce larger noise
        emin, emax: edges of the energy band within which to weight  effective area with phot_index power law

    return: scipy.interpolate.RegularGridInterpolator containing scalled effective area depending on offset in mm from the center of detector

    """

    if not energy is None:
        return make_vignetting_from_inverse_psf_ayut_cspec(urdn, np.array([energy-1e-5, energy+1e-5]), [1,])

    if not cspec is None:
        print(grid, cspec)
        return make_vignetting_from_inverse_psf_ayut_cspec(urdn, grid, cspec, app)

    if not phot_index is None:
        return make_vignetting_from_inverse_psf_ayut(urdn, emin, emax, phot_index, app)

    return make_vignetting_from_inverse_psf(urdn)


def make_overall_vignetting(subgrid=10, urdweights={}, **kwargs):
    """
    produces combined effective area of seven detector as projected on sky

    --------------
    Parameters:
        same as for make_vignetting_for_urdn, with additional arguments of urdweights - which weight vignetting map of each urd

    returns:
        scipy.interpolate.RegularGridInterpolator provideds projection of the effective area of seven detectors on sky, depending on offset from the "telescope axis" defined
        by mean quaternion stored in CALDB, offsets can be coverted to vectors with arttools._det_spatial.offset_to_vec which assumes focal length arttools._det_spatial.F
    """
    if subgrid < 1:
        print("ahtung! subgrid defines splines of the translation of multiple vigneting file into one map")
        print("set subgrid to 2")
        subgrid = 2
    #x, y = np.meshgrid(np.linspace(-24., 24., 48*subgrid), np.np.linspace(-24., 24., 48*subgrid))
    xmin, xmax = -40.*DL, 40.*DL
    ymin, ymax = -40.*DL, 40.*DL

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
    shape = x.shape
    newvmap = np.zeros(shape, np.double)
    vecs = offset_to_vec(np.ravel(x), np.ravel(y))

    for urdn in URDNS:
        vmap = make_vignetting_for_urdn(urdn, **kwargs)
        quat = get_boresight_by_device(urdn)
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs, inverse=True))).reshape(shape)*urdweights.get(urdn, 1.)

    mask = newvmap > 0.
    mx = mask.any(axis=1)
    my = mask.any(axis=0)
    mask = mx[:, np.newaxis] & my[np.newaxis, :]
    print(x.shape, mask.shape, newvmap.shape)
    vmapnew = np.copy(newvmap[mx, :][:, my])
    vmapnew[np.isnan(vmapnew)] = 0.0

    vmap = RegularGridInterpolator((x[:, 0][mx], y[0][my]), vmapnew, bounds_error=False, fill_value=0)
    return vmap
