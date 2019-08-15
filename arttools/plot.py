import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro
from math import pi, cos, sin
import copy

qrot0 = Rotation([sin(-15.*pi/360.), 0., 0., cos(-15.*pi/360.)])

ART_det_QUAT = {
        28 : Rotation([-0.0110150413845477,     -0.0013329854433192,     -0.0010840551373762,      0.9999378564878738]),
        22 : Rotation([-0.0097210046441978,     -0.0012050978602830,     -0.0010652583365244,      0.9999514563380221]),
        23 : Rotation([-0.0109821945673236,     -0.0013205569544235,     -0.0010833616131844,      0.9999382350222591]), 
        24 : Rotation([-0.0108749144342713,     -0.0012081051721620,     -0.0009784162714686,      0.9999396578891848]), 
        25 : Rotation([-0.0083598094506972,     -0.0012398390463856,     -0.0011014394837848,      0.9999636809485386]), 
        26 : Rotation([-0.0100908546351636,     -0.0012650094280487,     -0.0011698124374266,      0.9999476015985739]),
        30 : Rotation([-0.0108764670901360,     -0.0012574641047721,     -0.0010592143554341,      0.9999394978260493])
                }


detwcs = WCS(naxis=2)
detwcs.wcs.cdelt = np.array([45., 45.])/3600.
detwcs.wcs.crval = [24.5, 24.5]

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
    print(ra, dec, roll)
    ra, dec, roll = ra[0], dec[0], roll[0]
    print(ra, dec, roll)
    locwcs = copy.copy(detwcs)
    locwcs.wcs.crpix = np.array([ra, dec])*180./pi
    print(locwcs.wcs)
    locwcs.wcs.pc = np.array([[cos(roll), -sin(roll)], [sin(roll), cos(roll)]])

    ra, dec = locwcs.all_pix2world(np.array([urddata["RAW_X"], urddata["RAW_Y"]]).T, 0).T
    plt.scatter(ra, dec)
    plt.show()
    return np.histogram2d(ra, dec, [xe, ye])[0]


if __name__ == "__main__":
    pass
