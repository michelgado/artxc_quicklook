import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, get_gyro_quat
from math import pi, cos, sin
import copy

dxya = 45./3600./180.*pi

detwcs = WCS(naxis=2)
detwcs.wcs.cdelt = np.array([45., 45.])/3600.
detwcs.wcs.crpix = [24.5, 24.5]

def get_rawxy_hist(data, phabotmin = 150, phabotmax = 300, mask = None):
    mask = (data["PHA_BOT"] > phabotmin) & (data["PHA_BOT"] < phabotmax)
    masked_data = data[mask]
    img = np.histogram2d(masked_data['RAW_X'], masked_data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def logstretch(img):
    norm = ImageNormalize(img, stretch=LogStretch())
    plt.imshow(img, interpolation="nearest", origin="lower", norm=norm)

def urd_to_vec(urddata, subscale=1):
    sscale = (np.arange(subscale) - (subscale - 1)/2.)/subscale
    x = np.tile(urddata["RAW_X"], subscale*subscale) + \
            np.repeat(np.tile(sscale, urddata.size), subscale)
    y = np.tile(urddata["RAW_Y"], subscale*subscale) + \
            np.tile(np.tile(sscale, urddata.size), subscale)
    return xy_to_vec(x, y)

def xy_to_vec(x, y):
    """
    normshift 
    """
    return np.array([np.ones(x.size), np.tan((x - 24.)*dxya), np.tan((y-24.)*dxya)]).T


def get_sky(urddata, URDN, attdata, xe, ye):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], 25))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]
    photonvecs = urd_to_vec(urddata, 5)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    
    """
    ra, dec, roll = extract_raw_gyro(attdata, qinit)
    ra, dec, roll = ra[0], dec[0], roll[0]
    locwcs = copy.copy(detwcs)
    locwcs.wcs.crval = np.array([ra, dec])*180./pi
    #locwcs.wcs.pc = np.array([[cos(roll), -sin(roll)], [sin(roll), cos(roll)]])
    locwcs.wcs.pc = np.array([[cos(roll), sin(roll)], [-sin(roll), cos(roll)]])

    x = np.tile(urddata["RAW_X"], 25) + np.repeat(np.tile(np.array([-0.4, -0.2, 0., 0.2, 0.4]), urddata.size), 5)
    y = np.tile(urddata["RAW_Y"], 25) + np.tile(np.tile(np.array([-0.4, -0.2, 0., 0.2, 0.4]), urddata.size), 5)
    ra, dec = locwcs.all_pix2world(np.array([x, y]).T, 0).T
    """
    return np.histogram2d(ra, dec, [xe, ye])[0]


if __name__ == "__main__":
    pass
