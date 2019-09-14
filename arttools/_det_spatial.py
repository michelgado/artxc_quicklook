import numpy as np

DL = 0.595 # distance between strips in mm
F = 2693. # focal length in mm
dxya = np.arctan(DL/F) #0.595 - distance between strips, 2693 - ART-XC focal length


def raw_xy_to_offset(rawx, rawy):
    return (rawx - 23.5)*DL, (rawy - 23.5)*DL

def raw_xy_to_vec(rawx, rawy):
    """
    assuming that the detector is located in the  YZ vizier plane and X is normal to it
    we produce vectors, correspongin to the direction at which each particular pixel observe sky in the
    vizier coordinate system

    center of the detector is located at coordinate 23.5, 23.5 (detector strips have indexes from 0 to 47)
    """
    return offset_to_vec(*raw_xy_to_offset(rawx, rawy))

def offset_to_vec(x, y):
    outvec = np.empty(x.shape + (3,), np.double)
    outvec[..., 0] = F
    outvec[..., 1] = x
    outvec[..., 2] = -y
    return outvec

def vec_to_offset(vec):
    return vec[...,1]*F/vec[...,0], -vec[...,2]*F/vec[...,0]

def vec_to_offset_pairs(vec):
    return (vec[...,[1,2]]/vec[...,0][..., np.newaxis])*[F, -F]

def urd_to_vec(urddata, subscale=1):
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    rawx = np.repeat(urddata["RAW_X"], subscale*subscale) + \
            np.tile(np.tile(sscale, subscale), urddata.size)
    rawy = np.repeat(urddata["RAW_Y"], subscale*subscale) + \
            np.tile(np.repeat(sscale, subscale), urddata.size)
    return raw_xy_to_vec(rawx, rawy)

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
