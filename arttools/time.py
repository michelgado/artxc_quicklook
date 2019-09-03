import numpy as np
from .orientation import get_gyro_quat_as_arr, vec_to_pol, quat_to_pol_and_roll, extract_raw_gyro
from math import pi
from scipy.spatial.transforms import Rotation, Slerp

DELTASKY = 5./3600./180.*pi
"""
optica axis is shifted 11' away of the sattelite x axis, therefore we need some more fine resolution
5'' binning at the edge of the detector, is rotation take place around its center is 2*pi/9/24 
(hint: pix size 45'', 5''=45''/9)
"""
DELTAROLL = 2.*pi/24./18. 


def merge_consecutive_kvea_gtis(urdfile):
    pass

def fill_att_gaps(attdata):
    pass 

def angular_speed(attdata):
    ra, dec, 
    quat = get_gyro_quat_as_arr(attdata)
    dqrota = np.sum(quat[:, 1:]*quat[:, :-1], axis=1)

def mkgtimask(time, gti):
    mask = np.zeros(time.size, np.bool)
    idx = np.searchsorted(time, gti)

def hist_quat(quat):
    ra, dec, roll = quat_to_pol_and_roll(quat)

    orhist = np.empty((ra.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, np.int)
    orhist[:, 1] = np.asarray(cos(dec)*ra/DELTASKY, np.int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, np.int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def hist_orientation(attdata, gti, v0=None):
    quatint = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qval = quatint((attdata["TIME"][1:] + attdata["TIME"][:-1])/2.)

    ra, dec, roll = extract_raw_gyro(attdata)
    vec = vec_to_pol(dec, ra)
    dt = attdata["TIME"][1:] - attdata["TIME"][:-1]
    dalpha = np.arccos(vec[1:, :]*vec[:-1, :])
    droll = (roll[1:] - roll[:-1])

    maskstable = (dalpha < DELTASKY) & (droll < DELTAROLL)
    qvalstable = qval[masktable]
    print("stable vs moving", np.sum(dt[maskstable]), np.sum(dt[np.logical_not(

    oruniq, uidx, count = hist_quat(qval)


    orhist = np.empty((attdata.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/deltasky, np.int)
    orhist[:, 1] = np.asarray(cos(dec)*ra/deltasky, np.int)
    orhist[:, 2] = np.asarray(roll/deltaroll, np.int)

    oruniq, idx, count = np.unique(orhist, return_index=True, return_counts=True, axis=0)
    return attdata[idx], count
