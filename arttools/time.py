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

    idx = np.searchsorted(attdata["TIME"], gti)
    tnew = np.empty(np.sum(idx[:,1] - idx[:,0]) + 2*idx.shape[0])
    cidx = np.empty(idx.shape[0] + 1, np.int)
    cidx[1:] = np.cumsum(idx[:,1] - idx[:,0] + 2)
    for i in range(idx.shape[0]):
        tnew[cidx[i]+1: cidx[i+1]-1] = attdata["TIME"][idx[i,0]:idx[i:1]]
        tnew[cidx[i]] = gti[i, 0]
        tnew[cidx[i + 1] - 1] = gti[i, 1]
    ts = [np.array([gti[i, 0] + list(attdata["TIME"][idx[i, 0]:idx[i:1]]) + gti[i, 1]]) for i in range(gti.shape[0])]
    te = np.concatenate(ts)
    dt = np.concatenate([t[1:] - t[:-1] for t in ts])
    ts = np.concatenate([(t[1:] + t[:-1])/2. for t in ts])
    qval = quatint(ts)

    ra, dec, roll = quat_to_pol_and_roll(quatint(te))
    ra, dec, roll = extract_raw_gyro(attdata)
    vec = vec_to_pol(dec, ra)
    dalpha = np.arccos(vec[1:, :]*vec[:-1, :])
    droll = (roll[1:] - roll[:-1])

    maskmoving = (dalpha < DELTASKY) & (droll < DELTAROLL)
    qvalstable = qval[maskmoving]
    print("stable vs moving", np.sum(dt[maskmoving]), np.sum(dt[np.logical_not(masktable)]))

    oruniq, uidx, invidx = hist_quat(qval)
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt[maskmoving])

    maskstable = np.logical_not(maskmoving)

    ts = attdata["TIME"][:-1][maskstable]
    size = np.maximum(dalpha[maskstable]/DELTASKY, droll[maskstable]/DELTAROLL)
    dt = (attdata["TIME"][1:][maskstable] - ts)/size
    ar = np.arange(size.sum()) - np.repeat(np.cumsum([0,] + list(size[:-1])), size) + 0.5
    tnew = np.repeat(ts, size) + ar*dt
    # alternarive solution: tnew = np.concatenate([t0 + (np.arange(s) + 0.5)*dtloc for t0, s, dtloc in zip(ts, size, dt)])

    exptime = np.concatenate(exptime, dt)
    qval = np.concatenate(qvalstable, quatint(tnew))
    return exptime, qval

