from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAXOFFSET
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from scipy.integrate import cumtrapz
from .energy  import get_arf_energy_function
from ._det_spatial import offset_to_raw_xy, DL, F, raw_xy_to_offset, \
    offset_to_vec, vec_to_offset_pairs, vec_to_offset, rawxy_to_qcorr
from .telescope import URDNS
from .filters import Intervals
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device
from .psf import xy_to_opaxoffset, unpack_inverse_psf_ayut, unpack_inverse_psf_specweighted_ayut
from .spectr import get_specweights
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

def get_energycorr_for_offset(urdn, xo, yo):
    shmask = get_shadowmask_by_urd(urdn)
    x0, y0 = get_optical_axis_offset_by_device(urdn)

def basevignetting_function(ipsf, scale, brate):
    return ipsf*scale

def sensitivity_second_order(ipsf, scale, brate):
    return (scale*ipsf)**2./brate

class DetectorVignetting(object):
    def __init__(self, iifun, app=None):
        self._set_app_shmask(app)
        self._set_ipsf_functions(iifun)
        self._img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
        self.dpix = np.zeros((48, 48), bool)
        self.bmap = np.zeros((48, 48), bool)
        self.vignfun = basevignetting_function
        self.vignscale = 1.

    def _set_ipsf_functions(self, iifun):
        self.iifun = iifun
        self.norm = np.sum([self.iifun(i, j)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])

    def set_vignetting_functions(self, vfun):
        self.vignfun = vfun

    def set_vignscale(self, scale):
        self.vignscale = scale

    def _clean_img(self):
        self._img[:, :] = 0.
        self.dpix[:, :] = False

    def set_bkgratemap(self, bmap):
        self.bmap = bmap

    def _set_app_shmask(self, app):
        if app is None:
            self.psfmask = None
        else:
            x, y = np.mgrid[-60:61:1, -60:61:1]
            self.psfmask = x**2. + y**2. > app**2./25.
            self.app = app

    def add_pix(self, x, y, i, j):
        if ~self.dpix[x, y]:
            self._img[(x - 1)*9: (x - 1)*9 + 121, (y - 1)*9: (y - 1)*9 + 121] += self.vignfun(self.iifun(i, j), self.vignscale, self.bmap[x, y])
            self.dpix[x, y] = True

    @property
    def img(self):
        return self._img/self.norm

    def produce_vignentting(self, x, y, i, j):
        for xp, yp, il, jl in zip(x, y, i, j):
            self.add_pix(xp, yp, il, jl)
        return self.img

    def get_corners():
        x = np.arange(48)[np.any(self.dpix, axis=1)]
        xmin, xmax = x.min(), x.max()
        y = np.arange(48)[np.any(self.dpix, axis=0)]
        ymin, ymax = y.min(), y.max()
        return raw_xy_to_offset(np.array([xmin, xmax, xmax, xmin]), np.array([ymin, ymin, ymax, ymax]))



DEFAULVIGNIFUN = RegularGridInterpolator((np.arange(-262.5, 263, 1)/9.*DL, np.arange(-262.5, 263, 1)/9.*DL), np.zeros((526, 526)), bounds_error=False, fill_value=0.)


def make_vignetting_for_urdn(urdn, imgfilter, cspec=None, app=None, brate=None, vfun=None, scale=None):
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

    x, y = np.mgrid[0:48:1, 0:48:1]
    #shmask = imgfilter.apply(np.column_stack([y.ravel(), x.ravel()]).ravel().view([("RAW_X", np.int), ("RAW_Y", np.int)])).reshape(x.shape)
    shmask = imgfilter.filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    x0, y0 = get_optical_axis_offset_by_device(urdn)

    x, y = np.mgrid[0:48:1, 0:48:1]
    x1, y1 = x[shmask], y[shmask]
    x2, y2 = xy_to_opaxoffset(x1, y1, urdn)


    #ee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])
    #w = get_specweights(imgfilter.filters, ee, cspec)
    iifun = unpack_inverse_psf_specweighted_ayut(imgfilter.filters, cspec=cspec)
    vmap = DetectorVignetting(iifun, app)
    if not brate is None:
        vmap.set_bkgratemap(get_background_surface_brigtnress(urdn, imgfilter, normalize=True)*brate)
    if not scale is None:
        vmap.set_vignscale(scale)
    if not vfun is None:
        vmap.set_vignetting_functions(vfun)

    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        vmap.add_pix(xl, yl, xo, yo)

    """
    if app is None:
        psfmask = None
    else:
        x, y = np.mgrid[-60:61:1, -60:61:1]
        psfmask = x**2. + y**2. > app**2./25.

    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    for xl, yl, xo, yo in zip(x1, y1, x2, y2):
        dx, dy = xl - x0, yl - y0
        sl = img[(xl - 1)*9: (xl - 1)*9 + 121, (yl - 1)*9: (yl - 1)*9 + 121]
        lipsf = np.sum(unpack_inverse_psf_ayut(xo, yo)*w[:, np.newaxis, np.newaxis], axis=0)
        #print(lipsf.shape, x0, y0, xl, yl, sl.shape)
        if not app is None:
            lipsf[psfmask] = 0.
        sl += lipsf
    """

    dx = (np.arange(vmap.img.shape[0]) - (vmap.img.shape[0] - 1.)/2.)/9.*DL
    #imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    return RegularGridInterpolator((dx, dx), vmap.img, bounds_error=False, fill_value=0.) #TODO for spefici masks, pixel at optical axis position can be switched off, broking normalization


def get_blank_vignetting_interpolation_func():
    img = np.zeros((46*9 + 121 - 9, 46*9 + 121 - 9), np.double)
    dx = (np.arange(img.shape[0]) - (img.shape[0] - 1.)/2.)/9.*DL
    return RegularGridInterpolator((dx, dx), img, bounds_error=False, fill_value=0.)


def make_overall_vignetting(imgfilters, subgrid=10, urdweights={}, **kwargs):
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
        vmap = make_vignetting_for_urdn(urdn, imgfilters[urdn].filters, **kwargs)
        quat = get_boresight_by_device(urdn)
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs, inverse=True))).reshape(shape)*urdweights.get(urdn, 1.)

    mask = newvmap > 0.
    mx = mask.any(axis=1)
    my = mask.any(axis=0)
    mask = mx[:, np.newaxis] & my[np.newaxis, :]
    vmapnew = np.copy(newvmap[mx, :][:, my])
    vmapnew[np.isnan(vmapnew)] = 0.0

    vmap = RegularGridInterpolator((x[:, 0][mx], y[0][my]), vmapnew, bounds_error=False, fill_value=0)
    return vmap

