from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin
from ._det_spatial import urd_to_vec

qrot0 = Rotation([sin(15*pi/360.), 0., 0., cos(15*pi/360.)])

ART_det_QUAT = {
     28 : Rotation([-0.0116889160688843,      -0.0013389302010230,      -0.0010349697278926,       0.9999302502398423]),
     22 : Rotation([-0.0167377628868913,      -0.0012291384508458,      -0.0009991424307663,       0.9998586591246854]),
     23 : Rotation([-0.0099830838530707,      -0.0013258248395097,      -0.0010365397578859,       0.9999487515921043]),
     24 : Rotation([-0.0160846803318252,      -0.0012248074882882,      -0.0009421246479771,       0.9998694391301234]),
     25 : Rotation([-0.0241800760598627,      -0.0012497930988159,      -0.0010644654363138,       0.9997062712878639]),
     26 : Rotation([-0.0215736973366166,      -0.0012753621199235,      -0.0011089729127380,       0.9997658321896032]),
     30 : Rotation([-0.0191570016725280,      -0.0012708747390465,      -0.0010039631945285,       0.9998151760311606]),
            }

def to_2pi_range(val): return val%(2.*pi)


def get_photons_vectors(urddata, URDN, attdata, subscale=1):
    #attdata = filter_gyrodata(attdata)
    print("any unsorted times", np.any(attdata["TIME"][1:] <= attdata["TIME"][:-1]))
    attdata = attdata[np.argsort(attdata["TIME"])]
    print("after sorting", np.any(attdata["TIME"][1:] <= attdata["TIME"][:-1]))
    if np.any(attdata["TIME"][1:] <= attdata["TIME"][:-1]):
        idx = np.where(attdata["TIME"][1:] <= attdata["TIME"][:-1])[0]
        print(attdata["TIME"][idx], attdata["TIME"][idx + 1])
        utime, idx = np.unique(attdata["TIME"], return_index=True)
        attdata = attdata[idx]
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    return phvec

def vec_to_pol(phvec):
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))
    return ra, dec

def pol_to_vec(phi, theta):
    vec = np.empty((theta.size, 3), np.double)
    vec[:, 0] = np.cos(theta)*np.cos(phi)
    vec[:, 1] = np.cos(theta)*np.sin(phi)
    vec[:, 2] = np.sin(theta)
    return vec

def get_photons_sky_coord(urddata, URDN, attdata, subscale=1):
    phvec = get_photons_vectors(urddata, URDN, attdata, subscale)
    return vec_to_pol(phvec)

def get_gyro_quat(gyrodata):
    quat = Rotation(np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T)
    q0 = Rotation([0, 0, 0, 1]) #gyro axis initial rotattion in J2000 system
    qfin = q0*quat
    return qfin

def filter_gyrodata(gyrodata):
    return gyrodata[nonzero_quaternions(np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T)]

def nonzero_quaternions(quat):
    mask = np.sum(quat**2, axis=1) > 0
    return mask

def get_gyro_quat_as_arr(gyrodata):
    return np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T

def quat_to_pol_and_roll(qfin, opaxis=[1, 0, 0], north=[0, 0, 1]):
    """
    it is assumed that quaternion is acting on the sattelite coordinate system
    in order to orient in in icrs coordinates
    opaxis - define dirrection of the main axis (x axis [1, 0, 0] coinside with optical axis)
    we assume that north is oriented along z coordinate, we also name this coordinate
    north for detectors
    """
    opticaxis = qfin.apply(opaxis)
    dec = np.arctan(opticaxis[:,2]/np.sqrt(opticaxis[:,1]**2 + opticaxis[:,0]**2))
    ra = np.arctan2(opticaxis[:,1], opticaxis[:,0])%(2.*pi)

    yzprojection = np.cross(opticaxis, north)
    vort = np.cross(north, opaxis)

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply(vort), axis=1),
                           np.sum(yzprojection*qfin.apply(north), axis=1))
    return ra, dec, rollangle

def extract_raw_gyro(gyrodata, qadd=qrot0):
    """
    unpacks row gyro fits file in to RA, DEC and roll angle (of the telescope coordinate system)
    in J2000 coordinates.
    """
    qfin = get_gyro_quat(gyrodata)*qadd
    return quat_to_pol_and_roll(qfin)
