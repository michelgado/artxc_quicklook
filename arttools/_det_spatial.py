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
