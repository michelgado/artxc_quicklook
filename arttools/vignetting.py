from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAXOFFSET
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumtrapz
from ._det_spatial import offset_to_raw_xy, DL, F, \
    offset_to_vec, vec_to_offset_pairs, vec_to_offset
from .telescope import URDNS
from .caldb import get_boresight_by_device, get_inverse_psf, get_optical_axis_offset_by_device
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
    ipsf = get_inverse_psf()
    nax1 = ipsf[1].header["NAXIS1"]
    nax2 = ipsf[1].header["NAXIS2"]
    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]

    img = np.zeros((46*9 + 121, 46*9 + 121), np.double)
    for xl, yl in zip(x1, y1):
        if (xl - x0 + 26) < 0 or (xl - x0 + 26) > 52 or (yl - y0 + 26) < 0 or (yl - y0 + 26) > 52:
            shmask[xl, yl] = False
            continue
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        """
        sl = img[int((xl - x0 + 23.)*9) + 60 - 60: int((xl - x0 + 23.)*9) + 60 + 61, int((yl - y0 + 23.)*9) + 60 - 60: int((yl - y0 + 23.)*9) + 60 + 61]
        """
        sl += ipsf[1].data[int(np.round(xl + 0.5 - x0)) + 26, int(np.round(yl + 0.5 - y0)) + 26, : sl.shape[0], :sl.shape[1]]

    dx = (np.arange(img.shape[0]) - img.shape[0]//2)/9.*DL
    return RegularGridInterpolator((dx, dx), img/img.max(), bounds_error=False, fill_value=0.)

@lru_cache(maxsize=7)
def make_vignetting_for_urdn(urdn, energy=7.2, flat=False, phot_index=None,
                             useshadowmask=True, ignoreedgestrips=True,
                             emin=0, emax=np.inf):
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

    return make_vignetting_from_inverse_psf(urdn)

    shmask = get_shadowmask_by_urd(urdn).astype(np.uint8) if useshadowmask else np.ones((48, 48), np.uint8)
    shmask[[0, -1], :] = 0
    shmask[:, [0, -1]] = 0
    """
    return RegularGridInterpolator((np.arange(-23.5, 23.6, 1.)*DL, np.arange(-23.5, 23.6, 1.)*DL),
                                   shmask, bounds_error=False, fill_value=0.)
    """

    vignfile = get_vigneting_by_urd(urdn)
    #TO DO: put max eff area in CALDB
    norm = 65.71259133631082 # = np.max(vignfile["5 arcmin PSF"].data["EFFAREA"])
    if useshadowmask:
        shmask = get_shadowmask_by_urd(urdn).astype(np.uint8)
    else:
        x, y = np.mgrid[-23.5:23.6:1, -23.5:23.6:1]
        shmask = x**2 + y**2 < 25.**2.

    if ignoreedgestrips:
        shmask[[0, -1], :] = False
        shmask[:, [0, -1]] = False

    pixmask = PixInterpolator(shmask)

    if flat:
        return pixmask

    efint = interp1d(vignfile["5 arcmin PSF"].data["E"],
                     vignfile["5 arcmin PSF"].data["EFFAREA"],
                     axis=0)

    if not phot_index is None:
        s, e = np.searchsorted(vignfile["5 arcmin PSF"].data["E"], [emin, emax])
        es = np.copy(vignfile["5 arcmin PSF"].data["E"][max(s - 1, 1): e+1])
        e0 = np.copy(es)
        es[0] = max(es[0], emin)
        es[-1] = min(es[-1], emax)
        vmap = np.copy(vignfile["5 arcmin PSF"].data["EFFAREA"][max(s - 1, 1): e+1])
        de = es[1:] - es[:-1]
        if phot_index == 1:
            s1 = np.log(es[1:]/es[:-1])
        else:
            s1 = (es[1:]**(1. - phot_index) - es[:-1]**(1. - phot_index))/(1. - phot_index)
        if phot_index == 2:
            s2 = np.log(es[1:]/es[:-1])
        else:
            s2 = (es[1:]**(2. - phot_index) - es[:-1]**(2. - phot_index))/(2. - phot_index)
        a = vmap[:-1]
        b = (vmap[1:] - vmap[:-1])/(e0[1:, np.newaxis, np.newaxis] - e0[:-1, np.newaxis, np.newaxis])
        vignmap = np.sum((a - b*e0[:-1, np.newaxis, np.newaxis])*s1[:, np.newaxis, np.newaxis] + \
                b*s2[:, np.newaxis, np.newaxis], axis=0)/np.sum(s1)
    else:
        print("compute for energy", energy)
        vignmap = efint(energy)

    vignmap = vignmap/norm
    print("check vignetting map:", vignmap.max())

    x = np.tan(vignfile["Offset angles"].data["X"]*pi/180/60.)*F - (24. - OPAXOFFSET[urdn][0])*DL
    y = np.tan(vignfile["Offset angles"].data["Y"]*pi/180/60.)*F - (24. - OPAXOFFSET[urdn][1])*DL

    X, Y = np.meshgrid(x, y)
    mask = pixmask((X.ravel(), Y.ravel())).reshape(X.shape)

    #temporary solution, we need to account for psf for shadow mask
    cutzero = cutmask(mask)
    x = x[mask.any(axis=1)]
    y = y[mask.any(axis=0)]
    vignmap = cutzero(vignmap)
    mask = cutzero(mask)
    vignmap[np.logical_not(mask)] = 0.
    vmap = RegularGridInterpolator((x, y), vignmap[:, :], bounds_error=False, fill_value=0.)
    return vmap


def make_overall_vignetting(energy=7.2, *args,
                            subgrid=10, urdweights={},
                            **kwargs):
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
    xmin, xmax = -28.*DL, 28.*DL
    ymin, ymax = -28.*DL, 28.*DL

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
        vmap = make_vignetting_for_urdn(urdn, energy, *args, **kwargs)
        quat = get_boresight_by_device(urdn)
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs, inverse=True))).reshape(shape)*urdweights.get(urdn, 1.)

    vmap = RegularGridInterpolator((x[:, 0], y[0]), newvmap, bounds_error=False, fill_value=0)
    return vmap
