from .caldb import get_optical_axis_offset_by_device, get_inverse_psf_data, \
        get_inversed_psf_data_packed, get_inverse_psf_datacube_packed
import numpy as np
from scipy.interpolate import RegularGridInterpolator

def xy_to_opaxoffset(x, y, urdn):
    x0, y0 = get_optical_axis_offset_by_device(urdn)
    return np.round(x + 0.5 - x0).astype(np.int), np.round(y + 0.5 - y0).astype(np.int)

def rawxy_to_opaxoffset(urddata, urdn):
    x, y = get_optical_axis_offset_by_device(urdn)
    return np.round(urddata["RAW_X"] + 0.5 - x).astype(np.int), np.round(urddata["RAW_Y"] + 0.5 - y).astype(np.int)

def get_inversed_psf_profiles(xshift, yshift):
    ipsf = get_inversed_psf_profiles()
    x0, y0 = rawxy_to_opaxoffset()

def unpack_inverse_psf(i, j):
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
    ia = abs(i)
    ja = abs(j)
    if ja > ia:
        ja, ia = ia, ja
    k = (ia + 1)*ia//2 + ja
    data = get_inverse_psf_datacube_packed()[k]
    if i < 0:
        data = np.flip(data, axis=0)
    if j < 0:
        data = np.flip(data, axis=1)
    if abs(j) > abs(i):
        data = np.transpose(data)
    return data

def get_ipsf_interpolation_func():
    ipsf = get_inversed_psf_data_packed()
    xo = ipsf["offset"].data["x_offset"]
    yo = ipsf["offset"].data["y_offset"]
    return RegularGridInterpolator((xo, yo), np.empty((xo.size, yo.size), np.double))
