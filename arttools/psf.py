from .caldb import get_optical_axis_offset_by_device, get_ayut_inverse_psf_datacube_packed, get_ayut_inversed_psf_data_packed
from .spectr import get_filtered_crab_spectrum, Spec, get_specweights
from ._det_spatial import offset_to_vec, DL, raw_xy_to_offset, vec_to_offset
from .sphere import get_vec_triangle_area
import numpy as np
from math import sin, cos, pi, sqrt
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import quad
from functools import lru_cache

def xy_to_opaxoffset(x, y, urdn):
    x0, y0 = (24, 24) if urdn is None else get_optical_axis_offset_by_device(urdn)
    return np.round(x + 0.5 - x0).astype(np.int), np.round(y + 0.5 - y0).astype(np.int)

def rawxy_to_opaxoffset(rawx, rawy, urdn=None):
    """
    returns i and j coordinate of the pixel relative to the pixel 0, 0 which we assume that contain optical axis in its center
    """
    x0, y0 = (0.6, 0.6) if urdn is None else get_optical_axis_offset_by_device(urdn)
    return np.round(rawx + 0.5 - x0).astype(np.int), np.round(rawy + 0.5 - y0).astype(np.int)

def urddata_to_opaxoffset(urddata, urdn):
    return rawxy_to_opaxoffset(urddata["RAW_X"], urddata["RAW_Y"], urdn)

@lru_cache(maxsize=7)
def get_urddata_opaxofset_map(urdn):
    x, y = get_optical_axis_offset_by_device(urdn)
    X, Y = np.arange(48), np.arange(48)
    return np.round(X + 0.5 - x).astype(np.int), np.round(Y + 0.5 - y).astype(np.int)

def opaxoffset_to_pix(x, y, urdn=None):
    x0, y0 = (0, 0) if urdn is None else get_optical_axis_offset_by_device(urdn)
    return np.round(x0 + 0.5 + x).astype(np.int), np.round(y + 0.5 + y0).astype(np.int)

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



def vec_to_ipsfpix(rawx, rawy, vec, urdn=None):
    i, j = rawxy_to_opaxoffset(rawx, rawy, urdn)
    xof, yof = (0., 0) if urdn is None else raw_xy_to_offset(rawx, rawy)
    xo, yo = vec_to_offset(vec)
    xl, yl = xo - xof, yo - yof   #considering, that 2nd order components are still small on the offsets < 0.5arcdeg scale, we have in detectors plane
    m = np.abs(j) > np.abs(i)
    xl[m], yl[m] = yl[m], xl[m]
    m = j < 0
    xl[m] = -xl[m]
    m = i < 0
    yl[m] = -yl[m]
    return unpack_pix_index(i, j), xl, yl

def naive_bispline_interpolation(rawx, rawy, vec, energy=None, urdn=None, data=None): #imgfilter=None, cspec=None): #, data=None):
    """
    for specified event provides bilinearly interpolated ipsf core values towards defined direction
    """
    iifun = get_ipsf_interpolation_func()

    imgmax = np.sum([unpack_inverse_psf_ayut(i, j)[:, 60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)], axis=0)
    """
    if imgfilter is None:
        imgmax = np.sum([unpack_inverse_psf_ayut(i, j)[:, 60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)], axis=0)
    else:
        w = get_specweights(imgfilter, ayutee, None)
        data = np.sum(get_ayut_inverse_psf_datacube_packed()*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
        imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])

    if data is None:
        data = get_ayut_inverse_psf_datacube_packed()
        imgmax = np.sum([unpack_inverse_psf_ayut(i, j)[:, 60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)], axis=0)
        data = data/imgmax
    """
    data = get_ayut_inverse_psf_datacube_packed()

    k, xl, yl = vec_to_ipsfpix(rawx, rawy, vec, urdn)
    mask = np.all([xl > iifun.grid[0][0], xl < iifun.grid[0][-1], yl > iifun.grid[1][0], yl < iifun.grid[1][-1]], axis=0)
    #print("mask sum and size", mask.size, mask.sum())
    k, xl, yl = k[mask], xl[mask], yl[mask]
    ip = np.searchsorted(iifun.grid[0], xl) - 1
    jp = np.searchsorted(iifun.grid[1], yl) - 1
    eidx = np.searchsorted(ayutee, energy[mask]) - 1 if type(energy) is np.ndarray else np.searchsorted(ayutee, energy)
    ishift = 1 - 2*(xl < iifun.grid[0][ip])
    jshift = 1 - 2*(yl < iifun.grid[1][jp])
    #print(ip.size, eidx.size)
    xg, yg = iifun.grid
    s = 1./((xg[ip + ishift] - xg[ip])*(yg[jp + jshift] - yg[jp]))*( \
        data[k, eidx, ip, jp]*(xg[ip+ishift] - xl)*(yg[jp + jshift] - yl) + \
        data[k, eidx, ip+ishift, jp]*(xl - xg[ip])*(yg[jp + jshift] - yl) + \
        data[k, eidx, ip, jp+jshift]*(xg[ip+ishift] - xl)*(yl - yg[jp]) + \
        data[k, eidx, ip+ishift, jp+jshift]*(xl - xg[ip])*(yl - yg[jp]))
    mask[mask] = s > 0.
    return mask, (s/imgmax[eidx])[s > 0.]

def naive_bispline_interpolation_specweight(rawx, rawy, vec, data, urdn=None, cspec=None):
    """
    for specified event provides bilinearly interpolated ipsf core values towards defined direction
    """
    iifun = get_ipsf_interpolation_func()
    if data is None:
        w = get_specweights(imgfilter, ayutee, cspec)
        data = np.sum(get_ayut_inverse_psf_datacube_packed()*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
        imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
        data = data/imgmax #(data[0, 60, 60] + data[1, 60, 51]*4 + data[2, 51, 51]*4)

    #data = get_ayut_inverse_psf_datacube_packed()
    k, xl, yl = vec_to_ipsfpix(rawx, rawy, vec, urdn)
    mask = np.all([xl > iifun.grid[0][0], xl < iifun.grid[0][-1], yl > iifun.grid[1][0], yl < iifun.grid[1][-1]], axis=0)
    k, xl, yl = k[mask], xl[mask], yl[mask]
    ip = np.searchsorted(iifun.grid[0], xl) - 1
    jp = np.searchsorted(iifun.grid[1], yl) - 1
    ishift = 1 - 2*(xl < iifun.grid[0][ip])
    jshift = 1 - 2*(yl < iifun.grid[1][jp])
    xg, yg = iifun.grid
    s = 1./((xg[ip + ishift] - xg[ip])*(yg[jp + jshift] - yg[jp]))*( \
        data[k, ip, jp]*(xg[ip+ishift] - xl)*(yg[jp + jshift] - yl) + \
        data[k, ip+ishift, jp]*(xl - xg[ip])*(yg[jp + jshift] - yl) + \
        data[k, ip, jp+jshift]*(xg[ip+ishift] - xl)*(yl - yg[jp]) + \
        data[k, ip+ishift, jp+jshift]*(xl - xg[ip])*(yl - yg[jp]))
    return mask, s




def offset_to_psfcoord(i, j, xo, yo, energy=None):
    iifun = get_ipsf_interpolation_func()
    xi, yi = np.searchsorted(iifun.grid[0], xo), np.searchsorted(iifun.grid[1], yo)
    mask = np.all([xi > 0, xi < iifun.grid[0].size,  yi > 0, yi < iifun.grid[1].size], axis=0)
    xi, yi = xi[mask], yi[mask]
    k = unpack_pix_index(i[mask], j[mask])
    m = np.abs(j) < np.abs(i)
    xi[m], yi[m] = yi[m], xi[m]
    m = i < 0
    xi[m] = iifun.grid[0].size - 1 - xi[m]
    m = j < 0
    yi[m] = iifun.grid[1].size - 1 - yi[m]
    if energy is None:
        return mask, k, xi, yi
    else:
        return mask, k, xi, yi, np.searchsorted(ayutee, energy[mask]) - 1

def opaxoffset_to_ipsf_data_coords(xo, yo, scalarfield=None):
    iifun = get_ipsf_interpolation_func()
    i, j = opaxoffset_to_pix(xo/DL, yo/DL)
    il, jl = opaxoffset_to_pix((xo + iifun.grid[0][0])/DL, (yo - iifun.grid[1][0])/DL)
    ir, jr = opaxoffset_to_pix((xo + iifun.grid[0][-1])/DL, (yo + iifun.grid[1][-1])/DL)
    imax, jmax = np.max(ir - il), np.max(jr - jl)
    imax = imax + (imax%2 - 1)
    jmax = jmax + (jmax%2 - 1)
    ig, jg = np.mgrid[-imax//2:imax//2+1:1, -jmax//2:jmax//2+1:1]
    k, xi, yi, svec = [], [], [], []
    for ishift, jshift in zip(ig.ravel(), jg.ravel()):
        ml, kl, xil, yil = offset_to_psfcoord(i + ishift, j + jshift, xo - ishift*DL, yo - jshift*DL)
        if not scalarfield is None:
            svec.append(scalarfield(xo[ml] - ishift*DL, yo[ml] - jshift*DL))
        k.append(kl)
        xi.append(xil)
        yi.append(yil)
    if scalarfield is None:
        return np.concatenate(k), np.concatenate(xi), np.concatenate(yi)
    else:
        return np.concatenate(k), np.concatenate(xi), np.concatenate(yi), np.concatenate(svec)

ayutee = np.array([4., 6., 8., 10., 12., 16., 20., 24., 30.])

def get_ipsf_energy_index(urddata):
    return np.searchsorted(ayutee, urddata['ENERGY']) - 1

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


def get_iicore_normalization(filters, cspec=None):
    w = get_specweights(filters, ayutee, cspec)
    imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    return imgmax


def unpack_inverse_psf_datacube_specweight_ayut(imgfilter, cspec, app=None):
    w = get_specweights(imgfilter, ayutee, cspec)

    x, y = np.mgrid[-60:61:1, -60:61:1]
    data = get_ayut_inverse_psf_datacube_packed()
    if not app is None:
        psfmask = x**2. + y**2. <= app**2./25.
        data = data*psfmask[np.newaxis, np.newaxis, :, :]

    data = np.sum(data*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    data = data/imgmax #(d
    return data



def unpack_inverse_psf_specweighted_ayut(imgfilter, cspec=None, app=None):
    """
    produces spectrum weights for ipsf channels
    """
    data = unpack_inverse_psf_datacube_specweight_ayut(imgfilter, cspec, app)
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
    """
    return integral over effectiveness on sky area (integral in radians)
    """
    iifun = get_ipsf_interpolation_func()
    xo, yo = np.meshgrid(iifun.grid[0], iifun.grid[1])
    vecs = offset_to_vec(xo.ravel(), yo.ravel())
    v0 = offset_to_vec(0, 0)
    if app is None:
        appmask = np.ones(xo.shape, bool)
    else:
        appmask = np.sum(vecs*v0) > cos(app*pi/180./3600.)
    sarea = get_vec_triangle_area(offset_to_vec(iifun.grid[0][:-1], iifun.grid[0][:-1]),
                                  offset_to_vec(iifun.grid[0][1: ], iifun.grid[0][:-1]),
                                  offset_to_vec(iifun.grid[0][1: ], iifun.grid[0][1: ]))
    sarea = np.mean(sarea)*2.
    #print("psf fun pix area", sarea)
    w = get_specweights(imgfilter, ayutee, cspec)
    data = np.sum(get_ayut_inverse_psf_datacube_packed()*w[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
    imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
    data = data/imgmax
    #print(sarea.shape, data.shape, appmask.shape)
    data = (data*appmask[np.newaxis, :, :]).sum(axis=(1, 2))*sarea
    def newfunc(i, j):
        return data[unpack_pix_index(i, j)]
    return newfunc

def photbkg_pix_coeff(urdn, imgfilter, cspec=None):
    x, y = np.mgrid[0:48:1, 0:48:1]
    shmask = imgfilter.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
    xp, yp = x[shmask], y[shmask]
    bkgprofile = np.zeros((48, 48), float)
    pixi = get_pix_overall_countrate_constbkg_ayut(imgfilter, cspec)
    bkgprofile[xp, yp] = pixi(*rawxy_to_opaxoffset(xp, yp, urdn))
    return bkgprofile


def get_ipsf_interpolation_func(app=6.*60):
    """
    provides with new instance of ReugularInterpolatorGrid with grid (x, y coordinates) set to the resolution of IPSF crrently stores in caldb
    """
    ipsf = get_ayut_inversed_psf_data_packed()
    xo = ipsf["offset"].data["x_offset"] #*0.9874317205607761 #*1.0127289656 #*1.0211676541662125
    yo = ipsf["offset"].data["y_offset"] #*0.9874317205607761 #*1.0127289656 #1.0211676541662125
    #xo = xo[(xo > - app) & (xo < app)]
    return RegularGridInterpolator((xo, yo), np.empty((xo.size, yo.size), np.double), bounds_error=False, fill_value=0.)


def select_psf_groups(i, j, energy=None):
    if energy is None:
        ijpairs, iidx, counts = np.unique(np.array([i, j]), axis=1, return_counts=True, return_inverse=True)
    else:
        eidx = np.searchsorted(ayutee, energy) - 1
        ijpairs, iidx, counts = np.unique(np.array([i, j, eidx]), axis=1, return_counts=True, return_inverse=True)
    isidx = np.argsort(iidx)
    ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
    return ijpairs, isidx, ii, counts

def get_urdevents_ipsf_weights(vecs, urddata, **kwargs):
    iifun = get_ipsf_interpolation_func()
    xy, iidx = np.unique(urddata[["RAW_X", "RAW_Y"]], return_index=True)
    iicore = unpack_inverse_psf_specweighted_ayut(urddata.filters, **kwargs)
    w = np.zeros(urddata.data.size, float)
    for npix, (x, y) in enumerate(xy):
        iifun.values = iicore(*rawxy_to_opaxoffset(x, y))
        mask = iidx == npix
        w[mask] = iifun(vec_to_offset_pairs(vecs[mask]))
    return w
