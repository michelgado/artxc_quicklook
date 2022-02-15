from .caldb import get_optical_axis_offset_by_device, get_inverse_psf_data, get_arf, \
        get_inversed_psf_data_packed, get_inverse_psf_datacube_packed, get_ayut_inverse_psf_datacube_packed
from .energy  import get_arf_energy_function
from .spectr import get_filtered_crab_spectrum, Spec
from ._det_spatial import offset_to_vec
from .sphere import get_vec_triangle_area
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad

def xy_to_opaxoffset(x, y, urdn):
    x0, y0 = (23, 23) if urdn is None else get_optical_axis_offset_by_device(urdn)
    return np.round(x + 0.5 - x0).astype(np.int), np.round(y + 0.5 - y0).astype(np.int)

def rawxy_to_opaxoffset(rawx, rawy, urdn):
    x, y = get_optical_axis_offset_by_device(urdn)
    return np.round(rawx + 0.5 - x).astype(np.int), np.round(rawy + 0.5 - y).astype(np.int)

def urddata_to_opaxoffset(urddata, urdn):
    x, y = get_optical_axis_offset_by_device(urdn)
    return np.round(urddata["RAW_X"] + 0.5 - x).astype(np.int), np.round(urddata["RAW_Y"] + 0.5 - y).astype(np.int)

def opaxoffset_to_pix(x, y, urdn=None):
    x0, y0 = (23, 23) if urdn is None else get_optical_axis_offset_by_device(urdn)
    return np.round(x0 - 0.5 + x).astype(np.int), np.round(y - 0.5 + y0).astype(np.int)


def get_inversed_psf_profiles(xshift, yshift):
    ipsf = get_inversed_psf_profiles()
    x0, y0 = rawxy_to_opaxoffset()


def unpack_pix_index(i, j):
    """
    inverse psf is an integral of the product of psf and vignetting over ART-XC detectors pixels
    since this characteristics is a result of integral we cannot use differential approximation to extract
    PSF - (for psf we can use only one parameter - offset from optical axis)
    But! we can account for pixel symmetries and store only 1/8 of the data since rest can be restored with
    the help of square pixel symmetries - transposition and two mirror mappings

    k = i*(i - 1)/2 + j


    i = int(sqrt(k + 1/4) + 1/2)
    j = k - i*(i-1)

    """
    if type(i) in (int, np.int64, np.int32, np.int16):
        ia = abs(i)
        ja = abs(j)
        if ja > ia:
            ja, ia = ia, ja
    else:
        ia = np.abs(i)
        ja = np.abs(j)
        m = ja > ia
        ja[m], ia[m] = ia[m], ja[m]
    k = (ia + 1)*ia//2 + ja
    return k


def unpack_inverse_psf(i, j):
    """
    symmetries
    i < 0 : inverse y
    j < 0 : inverse x
    i < j : transpose
    """
    k = unpack_pix_index(i, j)
    data = get_inverse_psf_datacube_packed()[k]
    if abs(j) > abs(i):
        data = np.transpose(data)
    if i < 0:
        data = np.flip(data, axis=0)
    if j < 0:
        data = np.flip(data, axis=1)
    return data

ayutee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])


def get_ayut_specweights(imgfilter, cspec=None):
    w = np.zeros(ayutee.size - 1, np.double)
    if not cspec is None:
        egloc, egaps = imgfilter["ENERGY"].make_tedges(ayutee)
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        arf = get_arf_energy_function(get_arf())
        cspec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(egloc[:-1][egaps], egloc[1:][egaps])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        cspec = cspec/cspec.sum()
        np.add.at(w, np.searchsorted(ayutee, ec) - 1, cspec)
    else:
        rgrid, cspec = get_filtered_crab_spectrum(imgfilter, collapsegrades=True)
        crabspec = Spec(rgrid["ENERGY"][:-1], rgrid["ENERGY"][1:], cspec)
        egloc, egaps = imgfilter["ENERGY"].make_tedges(np.unique(np.concatenate([ayutee, rgrid["ENERGY"]])))
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        cspec = crabspec.integrate_in_bins(np.array([egloc[:-1], egloc[1:]]).T[egaps])
        cspec = cspec.sum()
        np.add.at(w, np.searchsorted(ayutee, ec) - 1, cspec)
    return w


def unpack_inverse_psf_ayut(i, j, e=None):
    """
    inverse psf is an integral of the product of psf and vignetting over ART-XC detectors pixels
    since this characteristics is a result of integral we cannot use differential approximation to extract
    PSF - (for psf we can use only one parameter - offset from optical axis)
    But! we can account for pixel symmetries and store only 1/8 of the data since rest can be restored with
    the help of square pixel symmetries - transposition and two mirror mappings

    k = i*(i - 1)/2 + j


    i = int(sqrt(k + 1/4) + 1/2)
    j = k - i*(i-1)

    symmetries
    i < 0 : inverse y
    j < 0 : inverse x
    i < j : transpose
    """
    k = unpack_pix_index(i, j)
    data = get_ayut_inverse_psf_datacube_packed()
    data = data[k]
    if abs(j) > abs(i):
        data = np.transpose(data, axes=(0, 2, 1))
    if i < 0:
        data = np.flip(data, axis=1)
    if j < 0:
        data = np.flip(data, axis=2)
    if not e is None:
        return data[np.searchsorted(ayutee, e) - 1]
    else:
        return data

def unpack_inverse_psf_with_weights(weightfunc):
    def newfunc(i, j):
        data = unpack_inverse_psf_ayut(i, j)
        return weightfunc(data)
    return newfunc

def unpack_inverse_psf_ayut(i, j, e=None):
    """
    inverse psf is an integral of the product of psf and vignetting over ART-XC detectors pixels
    since this characteristics is a result of integral we cannot use differential approximation to extract
    PSF - (for psf we can use only one parameter - offset from optical axis)
    But! we can account for pixel symmetries and store only 1/8 of the data since rest can be restored with
    the help of square pixel symmetries - transposition and two mirror mappings

    k = i*(i - 1)/2 + j


    i = int(sqrt(k + 1/4) + 1/2)
    j = k - i*(i-1)

    symmetries
    i < 0 : inverse y
    j < 0 : inverse x
    i < j : transpose
    """
    k = unpack_pix_index(i, j)
    data = get_ayut_inverse_psf_datacube_packed()
    data = data[k]
    if abs(j) > abs(i):
        data = np.transpose(data, axes=(0, 2, 1))
    if i < 0:
        data = np.flip(data, axis=1)
    if j < 0:
        data = np.flip(data, axis=2)
    if not e is None:
        return data[np.searchsorted(ayutee, e) - 1]
    else:
        return data


def unpack_inverse_psf_specweighted_ayut(imgfilter, cspec=None, app=None):
    """
    produces spectrum weights for ipsf channels
    """

    w = get_ayut_specweights(imgfilter, cspec)

    data = np.sum(get_ayut_inverse_psf_datacube_packed()*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    data = data/imgmax #(data[0, 60, 60] + data[1, 60, 51]*4 + data[2, 51, 51]*4)
    x, y = np.mgrid[-60:61:1, -60:61:1]
    if not app is None:
        psfmask = x**2. + y**2. <= app**2./25.
        data = data*psfmask[np.newaxis, :, :]
    def newfunc(i, j):
        k = unpack_pix_index(i, j)
        d = data[k]
        if abs(j) > abs(i):
            d = d.T #np.transpose(data, axes=(0, 2, 1))
        if i < 0:
            d = np.flip(d, axis=0)
        if j < 0:
            d = np.flip(d, axis=1)
        return d
    return newfunc

def get_pix_overall_countrate_constbkg_ayut(imgfilter, cspec=None, app=None):
    iifun = get_ipsf_interpolation_func()
    sarea = get_vec_triangle_area(offset_to_vec(iifun.grid[0][:-1], iifun.grid[0][:-1]),
                                  offset_to_vec(iifun.grid[0][1: ], iifun.grid[0][:-1]),
                                  offset_to_vec(iifun.grid[0][1: ], iifun.grid[0][1: ]))
    sarea = np.mean(sarea)*2.
    print("psf fun pix area", sarea)
    w = get_ayut_specweights(imgfilter, cspec)
    data = np.sum(get_ayut_inverse_psf_datacube_packed()*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    data = data/imgmax
    data = data.sum(axis=(1, 2))*sarea
    def newfunc(i, j):
        return data[unpack_pix_index(i, j)]
    return newfunc

def get_ipsf_interpolation_func(app=6.*60):
    ipsf = get_inversed_psf_data_packed()
    xo = ipsf["offset"].data["x_offset"] #*0.9874317205607761 #*1.0127289656 #*1.0211676541662125
    yo = ipsf["offset"].data["y_offset"] #*0.9874317205607761 #*1.0127289656 #1.0211676541662125
    #xo = xo[(xo > - app) & (xo < app)]
    return RegularGridInterpolator((xo, yo), np.empty((xo.size, yo.size), np.double), bounds_error=False, fill_value=0.)


def select_psf_grups(i, j, energy=None):
    if energy is None:
        ijpairs, iidx, counts = np.unique(np.array([i, j]), axis=1, return_counts=True, return_inverse=True)
    else:
        eidx = np.searchsorted(ayutee, energy) - 1
        ijpairs, iidx, counts = np.unique(np.array([i, j, eidx]), axis=1, return_counts=True, return_inverse=True)
    isidx = np.argsort(iidx)
    ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
    return ijpairs, isidx, ii, counts
