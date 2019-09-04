import numpy as np
from .orientation import get_gyro_quat_as_arr, vec_to_pol, \
        quat_to_pol_and_roll, extract_raw_gyro, get_gyro_quat, pol_to_vec
from math import pi
from scipy.spatial.transform import Rotation, Slerp

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
    orhist[:, 1] = np.asarray(np.cos(dec)*ra/DELTASKY, np.int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, np.int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def make_ingti_times(time, gti):
    idx = np.searchsorted(time, gti)
    tnew = np.empty(np.sum(idx[:,1] - idx[:,0]) + 2*idx.shape[0], np.double)
    cidx = np.empty(idx.shape[0] + 1, np.int)
    cidx[1:] = np.cumsum(idx[:,1] - idx[:,0] + 2)
    cidx[0] = 0
    for i in range(idx.shape[0]):
        tnew[cidx[i]+1: cidx[i+1]-1] = time[idx[i,0]:idx[i,1]]
        tnew[cidx[i]] = gti[i, 0]
        tnew[cidx[i + 1] - 1] = gti[i, 1]
    maskgaps = np.ones(tnew.size - 1, np.bool)
    maskgaps[cidx[1:-1] - 1] = False
    return tnew, maskgaps

def hist_orientation(attdata, gti, v0=None):
    quatint = Slerp(attdata["TIME"], get_gyro_quat(attdata))

    tnew, maskgaps = make_ingti_times(attdata["TIME"], gti)
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    qval = quatint(ts)

    ra, dec, roll = quat_to_pol_and_roll(quatint(tnew))

    """
    import matplotlib.pyplot as plt
    plt.scatter(ra, dec)
    plt.show()
    """
    """
    to do:
    formally, this subroutine should not know that optic axis is [1, 0, 0],
    need to fix this
    vec = qval.apply([1., 0, 0])
    """
    vec = pol_to_vec(ra, dec)
    vecprod = np.sum(vec[1:, :]*vec[:-1, :], axis=1)
    """
    this ugly thing appears due to the numerical precision
    """
    vecprod[vecprod > 1.] = 1.
    dalpha = np.arccos(vecprod)[maskgaps]
    cs = np.cos(roll)
    ss = np.sin(roll)
    vecprod = np.minimum(ss[1:]*ss[:-1] + cs[1:]*cs[:-1], 1.)
    droll = np.arccos(vecprod)[maskgaps]

    maskmoving = (dalpha < DELTASKY) & (droll < DELTAROLL)
    qvalstable = qval[maskmoving]
    print("stable vs moving", np.sum(dt[maskmoving]), np.sum(dt[np.logical_not(maskmoving)]))

    oruniq, uidx, invidx = hist_quat(qval[maskmoving])
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt[maskmoving])

    maskstable = np.logical_not(maskmoving)

    tsn = (ts - dt/2.)[maskstable]
    size = np.maximum(dalpha[maskstable]/DELTASKY, droll[maskstable]/DELTAROLL).astype(np.int)
    print("check most", droll[maskstable][np.argmax(size)]/DELTAROLL, dalpha[maskstable][np.argmax(size)], dt[maskstable][np.argmax(size)], size[np.argmax(size)])
    print(size.sum())
    print(size.max())
    print(dt.max())
    print(droll.max())
    print(dalpha.max())
    print(ts.size)
    print(size)
    dt = np.repeat(dt[maskstable]/size, size)
    ar = np.arange(size.sum()) - np.repeat(np.cumsum([0,] + list(size[:-1])), size) + 0.5
    tnew = np.repeat(tsn, size) + ar*dt
    # alternarive solution: tnew = np.concatenate([t0 + (np.arange(s) + 0.5)*dtloc for t0, s, dtloc in zip(ts, size, dt)])

    exptime = np.concatenate([exptime, dt])
    qval = quatint(np.concatenate([ts, tnew]))
    return exptime, qval

