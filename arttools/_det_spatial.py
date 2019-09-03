import numpy as np


dxya = np.arctan(0.595/2693.) #0.595 - distance between strips, 2693 - ART-XC focal length
F = 2693.

def raw_xy_to_vec(x, y):
    """
    assuming that the detector is located in the  YZ vizier plane and X is normal to it
    we produce vectors, correspongin to the direction at which each particular pixel observe sky in the
    vizier coordinate system

    center of the detector is located at coordinate 23.5, 23.5 (detector strips have indexes from 0 to 47)
    """
    outvec = np.empty((x.size, 3), np.double)
    outvec[:, 0] = 1.
    outvec[:, 1] = np.tan((x - 23.5)*dxya)
    outvec[:, 2] = np.tan((23.5 - y)*dxya)
    return outvec

def offset_to_vec(x, y):
    outvec = np.empty((x.size, 3), np.double)
    outvec[:, 0] = F
    outvec[:, 1] = x
    outvec[:, 2] = -y
    return outvec

def urd_to_vec(urddata, subscale=1):
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    x = np.repeat(urddata["RAW_X"], subscale*subscale) + \
            np.tile(np.tile(sscale, subscale), urddata.size)
    y = np.repeat(urddata["RAW_Y"], subscale*subscale) + \
            np.tile(np.repeat(sscale, subscale), urddata.size)
    return raw_xy_to_vec(x, y)

def weight_coordinate(PI, rawcoord, mask):
    return np.sum(PI*mask*rawcoord, axis=0)/np.sum(PI*mask, axis=0)

def weight_2D_coordinate(PIb, PIt, rawx, rawy, maskb, maskt):
    return weight_coordinate(PIb, rawx, maskb), weight_coordinate(PIt, rawy, maskt)

def get_shadowed_pix_mask(rawx, rawy, det_spat_mask):
    """
    provide mask for eventlist, which excludes events, ocured in the part of the detector covered by the 
    colimator
    """
    return det_spat_mask[rawx, rawy] #equivalent to [det_spat_mask[i, j] for i, j in zip(rawx, rawy)]


def get_shadowed_pix_mask_for_urddata(urddata, det_spat_mask):
    """
    provide mask for eventlist, which excludes events, ocured in the part of the detector covered by the 
    colimator
    """
    return get_shadowed_pix_mask(urddata["RAW_X"], urddata["RAW_Y"], det_spat_mask) #equivalent to [det_spat_mask[i, j] for i, j in zip(rawx, rawy)]


