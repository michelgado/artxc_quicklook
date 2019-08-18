import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, get_gyro_quat
from math import pi, cos, sin
import copy

dxya = np.arctan(0.595/2693.) #45./3600./180.*pi

def get_rawxy_hist(data, phabotmin = 50, phabotmax = 300, mask = None):
    mask = (data["PHA_BOT"] > phabotmin) & (data["PHA_BOT"] < phabotmax)
    masked_data = data[mask]
    img = np.histogram2d(masked_data['RAW_X'], masked_data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def xy_to_vec(x, y):
    """
    assuming that the detector is located in the  YZ vizier plane and X is normal to it
    we produce vectors, correspongin to the direction at which each particular pixel observe sky in the
    vizier coordinate system
    """
    outvec = np.empty((x.size, 3), np.double)
    outvec[:, 0] = 1.
    outvec[:, 1] = np.tan((x - 23.5)*dxya)
    outvec[:, 2] = np.tan((23.5 - y)*dxya)
    return outvec 

def urd_to_vec(urddata, subscale=1):
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    x = np.repeat(urddata["RAW_X"], subscale*subscale) + \
            np.tile(np.tile(sscale, subscale), urddata.size)
    y = np.repeat(urddata["RAW_Y"], subscale*subscale) + \
            np.tile(np.repeat(sscale, subscale), urddata.size)
    return xy_to_vec(x, y)

def get_photons_vectors(urddata, URDN, attdata):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(urddata["TIME"])
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata)
    phvec = qall.apply(photonvecs)
    return phvec

def get_photons_sky_coord(urddata, URDN, attdata):
    phvec = get_photons_vectors(urddata, URDN, attdata)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return ra, dec

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], 25))  #subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, 5) #subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    
    return np.histogram2d(dec, ra, [ye, xe])[0]


if __name__ == "__main__":
    pass
