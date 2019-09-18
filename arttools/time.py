import numpy as np
from .orientation import get_gyro_quat_as_arr, vec_to_pol, \
        quat_to_pol_and_roll, extract_raw_gyro, get_gyro_quat, pol_to_vec
from math import pi
from scipy.spatial.transform import Rotation, Slerp

DELTASKY = 15./3600./180.*pi #previously I set it to be 5''
"""
optica axis is shifted 11' away of the sattelite x axis, therefore we need some more fine resolution
5'' binning at the edge of the detector, is rotation take place around its center is 2*pi/9/24
(hint: pix size 45'', 5''=45''/9)
"""
DELTAROLL = 2.*pi/24./18.

def get_gti(ffile):
    return np.array([ffile["GTI"].data["START"], ffile["GTI"].data["STOP"]]).T

def check_gti_shape(gti):
    if gti.ndim != 2 or gti.shape[1] != 2:
        raise ValueError("gti is expected to be a numpy array of shape (n, 2), good luck next time")

def intervals_in_gti(gti, tlist):
    return np.searchsorted(gti[:, 0], tlist) - 1 == np.searchsorted(gti[:, 1], tlist)

def filter_nonitersect(gti, gtifilt):
    if gtifilt.size == 0 or gti.size == 0:
        return np.empty((0, 2), np.double)
    gti = gti[(gti[:, 0] < gtifilt[-1, 1]) & (gti[:, 1] > gtifilt[0, 0])]
    gtis = np.searchsorted(gtifilt[:,1], gti[:,0])
    mask = gti[:,1] > gtifilt[gtis, 0]
    return gti[mask]

def gti_union(gti):
    """
    we want to erase all casess when time intervals in gti are intersected

    to do that we use a simple algorithm:

    we want to find all start and end times for which all
    start and end times of other gtis are greater or lesser....
    such start and end times contain intersected gti intervals and will be start and end times of new gti

    1. sort gtis by the start time in ascending order
    1.1 find sorted index of raveled  sorted gtis where start times ar evend and end times are odd
    1.2 check start times which have sorted index equal to their positional indexes
    1.3 extract these times - they are start times for new gtis

    2. repeat for end times

    3. compbine obtained start and end times in new gti
    """
    gtiss = np.argsort(gti[:,0])
    gtise = np.argsort(gti[:,1])

    gtirs = np.ravel(gti[gtiss, :])
    gtire = np.ravel(gti[gtise, :])
    gtiidxs = np.argsort(gtirs)
    gtiidxe = np.argsort(gtire)

    newgti = np.array([
        gti[gtiss[np.arange(0, gtiidxs.size, 2) == gtiidxs[0::2]], 0],
        gti[gtise[np.arange(1, gtiidxe.size, 2) == gtiidxe[1::2]], 1]
                ]).T
    return newgti

def gti_intersection(gti1, gti2):
    check_gti_shape(gti1)
    check_gti_shape(gti2)

    gti1 = gti_union(gti1)
    gti2 = gti_union(gti2)

    gti1 = filter_nonitersect(gti1, gti2)
    gti2 = filter_nonitersect(gti2, gti1)

    tend = np.concatenate([gti1[intervals_in_gti(gti2, gti1[:,1]), 1], gti2[intervals_in_gti(gti1, gti2[:,1]), 1]])
    tend = np.unique(tend)
    ts = np.sort(np.concatenate([gti1[:,0], gti2[:,0]]))
    gtinew = np.array([ts[np.searchsorted(ts, tend) - 1], tend]).T
    return gtinew

def merge_consecutive_kvea_gtis(urdfile):
    pass

def fill_att_gaps(attdata):
    pass

def angular_speed(attdata):
    quat = get_gyro_quat_as_arr(attdata)
    dqrota = np.sum(quat[:, 1:]*quat[:, :-1], axis=1)
    return dqrota

def mkgtimask(time, gti):
    mask = np.zeros(time.size, np.bool)
    idx = np.searchsorted(time, gti)

def hist_quat(quat):
    ra, dec, roll = quat_to_pol_and_roll(quat)

    orhist = np.empty((ra.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, np.int)
    orhist[:, 1] = np.asarray(np.cos(dec - dec%(pi/180.*15./3600))*ra/DELTASKY, np.int)
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

def make_small_steps_quats(times, quats, gti):
    quatint = Slerp(times, quats)
    tnew, maskgaps = make_ingti_times(times, gti)
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    qval = quatint(ts)

    ra, dec, roll = quat_to_pol_and_roll(quatint(tnew))

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
    maskstable = np.logical_not(maskmoving)
    if np.any(maskstable):
        tsm = (ts - dt/2.)[maskstable]
        size = np.maximum(dalpha[maskstable]/DELTASKY, droll[maskstable]/DELTAROLL).astype(np.int)
        dtm = np.repeat(dt[maskstable]/size, size)
        ar = np.arange(size.sum()) - np.repeat(np.cumsum([0,] + list(size[:-1])), size) + 0.5
        tnew = np.repeat(tsm, size) + ar*dtm
        dtn = np.concatenate([dt[maskmoving], dtm])
        qval = quatint(np.concatenate([ts[maskmoving], tnew]))
    else:
        dtn = dt
    return qval, dtn

def hist_orientation(qval, dt):
    oruniq, uidx, invidx = hist_quat(qval)
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt)
    return exptime, qval[uidx]

def hist_orientation_for_attfile(attdata, gti, v0=None):
    quats = get_gyro_quat(attdata)
    qval, dtn = make_small_steps_quats(attdata["TIME"], quats, gti)
    return hist_orientation(qval, dtn)
