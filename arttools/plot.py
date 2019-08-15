import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qort0, ART_det_QUAT
from math import pi, cos, sin
import copy

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


def get_sky(urddata, URDN, attdata, xe, ye):
    qinit = qrot0*ART_det_QUAT[URDN]
    ra, dec, roll = extract_raw_gyro(attdata, qinit)
    ra, dec, roll = ra[0], dec[0], roll[0]
    locwcs = copy.copy(detwcs)
    locwcs.wcs.crval = np.array([ra, dec])*180./pi
    locwcs.wcs.pc = np.array([[cos(roll), -sin(roll)], [sin(roll), cos(roll)]])

    x = np.tile(urddata["RAW_X"], 25) + np.repeat(np.tile(np.array([-0.4, -0.2, 0., 0.2, 0.4]), urddata.size), 5)
    y = np.tile(urddata["RAW_Y"], 25) + np.tile(np.tile(np.array([-0.4, -0.2, 0., 0.2, 0.4]), urddata.size), 5)
    ra, dec = locwcs.all_pix2world(np.array([x, y]).T, 0).T
    return np.histogram2d(ra, dec, [xe, ye])[0]


if __name__ == "__main__":
    pass
