import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, \
        get_gyro_quat, filter_gyrodata, vec_to_pol
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec
from math import pi, cos, sin
import copy


def get_rawxy_hist(data):
    img = np.histogram2d(data['RAW_X'], data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    attdata = filter_gyrodata(attdata)
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    
    return np.histogram2d(dec, ra, [ye, xe])[0]

def mktimegrid(gti):
    ts = np.arange(gti[0], gti[1] + 0.9, 1.)
    ts[-1] = gti[1]
    tc = (ts[1:] + ts[:-1])/2.
    dt = ts[1:] - ts[:-1]
    return tc, dt

def make_expomap(vignmap, attdata, gti, urdn, wcs, energy = 8.5):
    """
    to do:
    limits - in lon direction not more then 2pi/3 range
    """
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))

    eidx = np.searchsorted(vignmap["Vign_EA"].data["E"], energy)
    xoffset, yoffset = np.meshgrid(vignmap["Coord"].data["X"], vignmap["Coord"].data["Y"])
    vignvec = offset_to_vec(np.ravel(xoffset), np.ravel(yoffset))
    effarea = np.ravel(vignmap["Vign_EA"].data["EFFAREA"][eidx])

    tc, dt = zip(*[mktimegrid(gt) for gt in gti])
    tc = np.concatenate(tc)
    dt = np.concatenate(dt)

    qlist = qj2000(tc)

    def make_wcs_image(quat, i, size):
        print("%d out of %d" % (i, size))
        qall = quat*qrot0*ART_det_QUAT[urdn]
        r, d = vec_to_pol(qall.apply(vignvec))
        x, y = wcs.all_world2pix(np.array([r, d]).T, 1).T
        return np.histogram2d(x, y, 
            [np.arange(int(wcs.wcs.crpix[0]*2) + 1), np.arange(int(wcs.wcs.crpix[1]*2) + 1)], 
            weights=effarea)[0].T

    img = sum(make_wcs_image(qlist[i], i, tc.size)*dt[i] for i in range(tc.size))
    return img

if __name__ == "__main__":
    pass
