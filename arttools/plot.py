import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, get_gyro_quat
from ._det_spatial import raw_xy_to_vec, offset_to_vec
from math import pi, cos, sin
import copy


def get_rawxy_hist(data):
    img = np.histogram2d(data['RAW_X'], data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def urd_to_vec(urddata, subscale=1):
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    x = np.repeat(urddata["RAW_X"], subscale*subscale) + \
            np.tile(np.tile(sscale, subscale), urddata.size)
    y = np.repeat(urddata["RAW_Y"], subscale*subscale) + \
            np.tile(np.repeat(sscale, subscale), urddata.size)
    return raw_xy_to_vec(x, y)

def get_photons_vectors(urddata, URDN, attdata, subscale=1):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    return phvec

def get_photons_sky_coord(urddata, URDN, attdata, subscale=1):
    phvec = get_photons_vectors(urddata, URDN, attdata, subscale)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return ra, dec

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    
    return np.histogram2d(dec, ra, [ye, xe])[0]

"""
def mk_expomap(vignmap, attfile, gti, energy = 8.5):
    for i in range(gti):

    

    eidx = np.searchsorted(vignmap["Vign_EA"].data["E"], energy)
    xoffset, yoffset = vignmap["Coord"]["X"], vignmax["Coord"]["Y"]
    vecs = offset_to_vec(xoffset, yoffset)
    quat = get_gyro_quat(attdata)
    arrq = quat.as_quat()
    atthist = 
"""





if __name__ == "__main__":
    pass
