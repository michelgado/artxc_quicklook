"""
_det spatial module contains all simple function which treate spatial entities in the detector coordinate system - Y, Z along rawx rawy axis and X normal to the detectors plane
whith that module you can convery spatial distances  in the detector plane from pixels to mm, convert offsets to vectors, corresponding to photons trajectoris in the detector plane
there are also functions which can be used to create masks on the urd data to exclude pixels, located in the colimator and protection shawdow (since only part of the detector is open)

"""


import numpy as np
from math import pi
from .aux import cache_single_result_np
from scipy.spatial.transform import Rotation

DL = 0.595 # distance between strips in mm
F = 2693. # focal length in mm
dxya = np.arctan(DL/F)*180./pi #0.595 - distance between strips, 2693 - ART-XC focal length


#@cache_single_result_np
def raw_xy_to_offset(rawx, rawy):
    """
    this function converts event parameters (rawx, rawy) to the spatial offset from the detector center in the detector plane
    the cooredinates of the offset are assumed to be aligned with y and z detecto axis (which are rolled 15degree clockwise around X axis, relative to the spacecraft coordinates system)
    the conversition is done thgrough the constant spatial size of an ART-XC detector pixel size, which assumed to be constant for all detectors.


    :param 'rawx': any number or array of numbers, expected to be the content
    :param 'rawx': any number or array of numbers, expected to be

    :return: depending on input retur two numbers or two array of numbers which are spatial offset from the detector center in mm in detector plane
    """
    return (rawx - 23.5)*DL, (rawy - 23.5)*DL

#@cache_single_result_np
def offset_to_raw_xy(x, y):
    """
    converts the spatial offset in the detector plane to the pixel integer coordinates (read rawx, rawy)
    note! no verifications that the coorinates are within a detector area is done j


    :param 'x' - spatial offset along rawx coordinate axis (assumed to be a number or array of numbers, which are offsets from the detector center in mm)
    :param 'y' - same as x byt in rawy axis direction

    returns: rawx, rawy int  coordinates, which corresponds to the pixel which covers provided spatial offset
    """
    x = x/DL + 24
    y = y/DL + 24
    return x.astype(np.int) - (x < 0), y.astype(np.int) - (y < 0)

#@cache_single_result_np
def raw_xy_to_vec(rawx, rawy):
    """
    assuming that the detector is located in the YZ vizier plane and X is normal to it
    we produce vectors, correspongin to the direction at which each particular pixel observe sky in the
    vizier coordinate system


    params: 'rawx' - spatial offset in detector  pixels along rawx axis
    params: 'rawy' - spatial offset in detector  pixels along rawy axis

    returns: vector defining direction on which pixel have to be projected
    """
    return offset_to_vec(*raw_xy_to_offset(rawx, rawy))

#@cache_single_result_np
def offset_to_vec(x, y):
    """
    converts spatial offset in the detector plane to the vector, defining offset from the optical axis, corresponding to the spatial offset


    params: 'x' numeric or array; spatial offset(s) in mm from the detector center along rawx axis
    params: 'y' numeric or array; spatial offset(s) in mm from the detector center along rawy axis

    return: unit vector corresponding to the offset from optical axis, which put the focused light in the pixel
    """
    outvec = np.empty(x.shape + (3,), np.double)
    outvec[..., 0] = F
    outvec[..., 1] = x
    outvec[..., 2] = -y
    return outvec

#@cache_single_result_np
def vec_to_offset(vec):
    """
    converts vector defined in the detector coordinate system to the spatial offset in mm


    params: 'vec' - array of shape (..., 3) last dimmention cooresponds to the 3d spatial compoents

    returns: pair of numbers or arrays which are offsets in mm along rawx and rawy axis, corresponding to vector offset from optical axis
    """
    return vec[...,1]*F/vec[...,0], -vec[...,2]*F/vec[...,0]

#@cache_single_result_np
def vec_to_offset_pairs(vec):
    """
    converts vector defined in the detector coordinate system to the spatial offset in mm


    params: 'vec' - array of shape (..., 3) last dimmention cooresponds to the 3d spatial compoents

    returns: arrays of shape (..., 2) which contains offsets in mm along rawx and rawy axis, corresponding to vector offset from optical axis
    """
    return (vec[...,[1,2]]/vec[...,0][..., np.newaxis])*[F, -F]


#@cache_single_result_np
def multiply_coord(x, y, subscale=1):
    """
    assuming that x and y are ceil indexes
    """
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    X = np.repeat(x, subscale*subscale) + \
            np.tile(np.tile(sscale, subscale), x.size)
    Y = np.repeat(y, subscale*subscale) + \
            np.tile(np.repeat(sscale, subscale), x.size)
    return X, Y


def multiply_photons(urddata, subscale=1):
    """
    for the provided urddata splits each pixel on subscale x subscale subpixes and return their detector coordinates
    """
    return multiply_coord(urddata["RAW_X"], urddata["RAW_Y"], subscale)


def urd_to_vec(urddata, subscale=1):
    """
    shorcut for offset_to_vec(urd_to_offset())
    produces unit vectors corresponding to offset from optical axis which are required for photon to be scatterd in the pixel with rawx rawy coordinates
    """
    if subscale == 1:
        return raw_xy_to_vec(urddata["RAW_X"], urddata["RAW_Y"])
    else:
        return raw_xy_to_vec(*urddata["RAW_X"], urddata["RAW_Y"], subscale)

def weight_coordinate(PI, rawcoord, mask):
    """
    future subresolution: YES WE CAN
    """
    return np.sum(PI*mask*rawcoord, axis=0)/np.sum(PI*mask, axis=0)

def weight_2D_coordinate(PIb, PIt, rawx, rawy, maskb, maskt):
    """
    spatial subpixel resolution with help of amplitudes distribution in the detector
    """
    return weight_coordinate(PIb, rawx, maskb), weight_coordinate(PIt, rawy, maskt)

def get_shadowed_pix_mask(rawx, rawy, det_spat_mask):
    """
    provides mask for eventlist, which excludes events, ocured in the part of the detector covered by the
    colimator

    params: rawx, rawy - array of pixel coordinates
    params: det_spat_mask - 2d array which defines which pixels are closed to the X-ray illumination

    return: boolean mask wich allow to  remove events, which could not be associated with incident X-ray photon registation
    """
    return det_spat_mask[rawx, rawy] #equivalent to [det_spat_mask[i, j] for i, j in zip(rawx, rawy)]

def get_shadowed_pix_mask_for_urddata(urddata, det_spat_mask):
    """
    provides mask for eventlist, which excludes events, ocured in the part of the detector covered by the
    colimator

    params: urddata  - fits record of the urd eventlis
    params: det_spat_mask - 2d array which defines which pixels are closed to the X-ray illumination

    return: boolean mask wich allow to  remove events, which could not be associated with incident X-ray photon registation
    """
    return get_shadowed_pix_mask(urddata["RAW_X"], urddata["RAW_Y"], det_spat_mask) #equivalent to [det_spat_mask[i, j] for i, j in zip(rawx, rawy)]


def offset_to_qcorr(x, y):
    """
    for x, y spatical cartesian coordinates produces a quaternion, which moves sphere to put center of coorddinates [1, 0, 0] in the x, y coordinate

    it should be noterd, that additional quaternion, which changes  rotates image on pi is applied first
    """
    v1 = np.array([0., 1., 0.])
    if type(x) is np.ndarray:
        v1 = np.tile(v1, x.shape + (1,))
    else:
        x, y = np.array([x,]), np.array([y,])
    q1 = Rotation.from_rotvec(v1*np.arctan2(-(23.5 - y)*DL, F)[:, np.newaxis])
    v1 = q1.apply(np.roll(v1, 1, axis=-1))
    q2 = Rotation.from_rotvec(v1*np.arctan2(-(23.5 - x)*DL, F)[:, np.newaxis])
    return q2*q1
