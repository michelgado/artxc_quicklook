from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin
from ._det_spatial import urd_to_vec

qrot0 = Rotation([sin(15*pi/360.), 0., 0., cos(15*pi/360.)])

ART_det_QUAT = {
            28 : Rotation([-0.0253822607926940,      -0.0013460478969772,     -0.0010308083508865,      0.9996763808484496]), 
            22 : Rotation([-0.0278942419232757,      -0.0012308887991602,     -0.0009912718735396,      0.9996096305860415]),
            23 : Rotation([-0.0218437922616629,      -0.0013321831701231,     -0.0010327175254869,      0.9997599749550605]),
            24 : Rotation([-0.0236571226930715,      -0.0012197597513018,     -0.0009268017282141,      0.9997189573928217]),
            25 : Rotation([-0.0319250004074392,      -0.0012520331046345,     -0.0010569697192827,      0.9994889241893090]),
            26 : Rotation([-0.0282551980978759,      -0.0012799715693514,     -0.0011048317403854,      0.9995993121246417]),
            30 : Rotation([-0.0295448995670286,      -0.0012731151181261,     -0.0009942313703070,      0.9995621489389505]),
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
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    return ra, dec

def pol_to_vec(theta, phi):
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


