from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin
from ._det_spatial import urd_to_vec

qrot0 = Rotation([sin(15*pi/360.), 0., 0., cos(15*pi/360.)]) #ART detectors cs to the spasecraft cs
OPAX = np.array([1, 0, 0])

ART_det_QUAT = {
     28 : Rotation([-0.0137255059682812,     -0.0013620640076146,     -0.0010460333208161,      0.9999043259641622]),
     22 : Rotation([-0.0140917096993521,     -0.0012663112050864,     -0.0010144580968526,      0.9998993904630860]),
     23 : Rotation([-0.0114706849757146,     -0.0013527546861559,     -0.0010473507511275,      0.9999327459871246]),
     24 : Rotation([-0.0182380330465064,     -0.0012561795468084,     -0.0009592189897345,      0.9998324239903759]),
     25 : Rotation([-0.0300494492471172,     -0.0012795577372984,     -0.0010788490728475,      0.9995470121092953]),
     26 : Rotation([-0.0285317706400531,     -0.0012989035844850,     -0.0011192377364951,      0.9995914156396656]),
     30 : Rotation([-0.0205228922068051,     -0.0012988613403022,     -0.0010164725268631,      0.9997880228519886]),
            }

ART_det_mean_QUAT = Rotation([-0.0194994955435183, -0.0014672512426498, -0.0011597505547702, 0.9998081175035487])

def to_2pi_range(val): return val%(2.*pi)

def make_orientation_gti(attdata, rac, decc, deltara, deltadec):
    qval = get_gyro_quat(attdata)*ART_det_mean_QUAT
    r, d = vec_to_pol(qval.apply(OPAX))
    r, d = r*180./pi, d*180./pi
    masktor = np.empty(r.size + 2, np.bool)
    masktor[1:-1] = np.all([r > rac - deltara, r < rac + deltara,
                            d > decc - deltadec, d < decc + deltadec], axis=0)
    masktor[[0, -1]] = False
    start = np.where(np.logical_and(mastor[:-1], np.logical_not(masktor[1:])))[0]
    end = np.where(np.logical_and(np.logical_not(mastor[:-1]), masktor[1:]))[0] - 1
    mask = end - start > 0
    gti = attdata["TIME"][np.array([stars, end]).T[mask]] + [-1e-6, +1e-6]
    return gti

def clear_att(attdata):
    attdata = filter_gyrodata(attdata)
    attdata = attdata[np.argsort(attdata["TIME"])]
    if np.any(attdata["TIME"][1:] <= attdata["TIME"][:-1]):
        idx = np.where(attdata["TIME"][1:] <= attdata["TIME"][:-1])[0]
        utime, idx = np.unique(attdata["TIME"], return_index=True)
        attdata = attdata[idx]
    return attdata

def get_photons_vectors(urddata, URDN, attdata, subscale=1):
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata)*ART_det_QUAT[URDN])
    qall = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
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
    qfin = q0*quat*qrot0
    return qfin

def get_gyro_quat_for_urdn(urdn, gyrodata):
    return get_gyro_quat(gyrodata)*ART_det_QUAT[urdn]

def filter_gyrodata(gyrodata):
    return gyrodata[nonzero_quaternions(np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T)]

def nonzero_quaternions(quat):
    mask = np.sum(quat**2, axis=1) > 0
    return mask

def get_gyro_quat_as_arr(gyrodata):
    return np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T

def get_bokz_quat(quatdata):
    pass

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

def extract_raw_gyro(gyrodata, qadd=Rotation([0, 0, 0, 1])):
    """
    unpacks row gyro fits file in to RA, DEC and roll angle (of the telescope coordinate system)
    in J2000 coordinates.
    """
    qfin = get_gyro_quat(gyrodata)*qadd
    return quat_to_pol_and_roll(qfin)


def get_axis_movement_speed(attdata):
    """
    for provided gyrodata computes angular speed
    returns:
        ts - centers of the time bins
        dt - withd of the time bins
        dlaphadt - angular speed in time bin
    """
    quats = get_gyro_quat(attdata)
    vecs = quats.apply(OPAX)
    dt = (attdata["TIME"][1:] - attdata["TIME"][:-1])
    dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))/dt*180./pi*3600.
    return (attdata["TIME"][1:] + attdata["TIME"][:-1])/2., dt, dalphadt


