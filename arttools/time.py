import numpy as np
from .orientation import get_gyro_quat_as_arr, vec_to_pol, \
        quat_to_pol_and_roll, extract_raw_gyro, get_gyro_quat, pol_to_vec, \
        clear_att, hist_orientation
from .telescope import OPAX
from .mask import edges as maskedges
from math import pi
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d

DELTASKY = 15./3600./180.*pi #previously I set it to be 5''
"""
optica axis is shifted 11' away of the sattelite x axis, therefore we need some more fine resolution
5'' binning at the edge of the detector, is rotation take place around its center is 2*pi/9/24
(hint: pix size 45'', 5''=45''/9)
"""
DELTAROLL = 1./24./3.
ARTDEADTIME = 770e-6 #seconds - ART-XC detector deadtime

class ART_TIME_ERROR(ValueError):
    pass

def get_gti(ffile):
    gti = np.array([ffile["GTI"].data["START"], ffile["GTI"].data["STOP"]]).T
    return gti_union(gti + [-0.5, +0.5]) + [+0.5, -0.5]

def get_filtered_elist(urddata, gti):
    return np.concatenate([urddata[s:e] for s, e in np.searchsorted(urddata["TIME"], gti)])

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
    produces union of the input gti interval
    !!!AHTUNG!!!
    the gti intervals with TSTART > TSTOP will be eliminated

    algorithm:
    we want to erase all casess when time intervals in gti are intersected

    to do that we use a simple algorithm:

    1. sort gtis by the start time in ascending order
    2. produce time searies which on even place hold this sorted start times, and on odd - corresponding end times
    3. from 1d time array, obtained in previous step produce sorted array
    since the start times were sorted, they can change their placing only one or several end times are greater.
    Therefore just check which start times did not change their positions due to sorting - those are start times of union gti
    The end times for the gti intervals located one index before start times in sorted array.
    the last  end time is also last time of sorted array.
    Taking in mind, that first start time is a first  time in sorted array produce bit mask for the start time and roll it 1step back to get end times mask
    done...
    """
    gti = gti[gti[:, 1] > gti[:, 0]]
    gti = gti[np.argsort(gti[:, 0])]
    idx = np.argsort(np.ravel(gti))
    gtis = np.ravel(gti)[idx]
    mask = np.zeros(gtis.size, np.bool)
    mask[::2] = idx[::2] == np.arange(0, idx.size, 2)
    mask[1::2] = np.roll(mask[::2], -1)
    return gtis[mask].reshape((-1, 2))

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
    for s, e in idx:
        mask[s:e] = True
    return mask

def hist_quat(quat):
    ra, dec, roll = quat_to_pol_and_roll(quat)

    orhist = np.empty((ra.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, np.int)
    orhist[:, 1] = np.asarray(np.cos(dec - dec%(pi/180.*15./3600))*ra/DELTASKY, np.int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, np.int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def make_ingti_times(time, ggti):
    gti = gti_intersection(np.array([time[[0, -1]],]), ggti)
    idx = np.searchsorted(time, gti)
    tnew = np.empty(np.sum(idx[:,1] - idx[:,0]) + 2*idx.shape[0], np.double)
    cidx = np.empty(idx.shape[0] + 1, np.int)
    cidx[1:] = np.cumsum(idx[:,1] - idx[:,0] + 2)
    cidx[0] = 0
    for i in range(idx.shape[0]):
        tnew[cidx[i]+1: cidx[i+1]-1] = time[idx[i,0]:idx[i,1]]
        tnew[cidx[i]] = gti[i, 0]
        tnew[cidx[i + 1] - 1] = gti[i, 1]
    maskgaps = np.ones(max(tnew.size - 1, 0), np.bool)
    maskgaps[cidx[1:-1] - 1] = False
    return tnew, maskgaps

def deadtime_correction(time, urdhk):
    if time[0] > urdhk["TIME"][-1] or time[-1] < urdhk["TIME"][0]:
        raise ART_TIME_ERROR("provided urd houskeeping data does not include provided time interval")

    tcrate = (urdhk["EVENTS"][1:] - urdhk["EVENTS"][:-1])/(urdhk["TIME"][1:] - urdhk["TIME"][:-1])
    icrate = interp1d((urdhk["TIME"][1:] + urdhk["TIME"][:-1])/2.,
                       crate, bounds_error=False, fill_value=np.median(crate))
    return (1. - 770e-6*icrate(ts))

def make_small_steps_quats(times, quats, gti, urdhk=None):
    quatint = Slerp(times, quats)
    tnew, maskgaps = make_ingti_times(times, gti)
    if tnew.size == 0:
        return Rotation(np.empty((0, 4), np.double)), np.array([])
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]

    if not urdhk is None:
        dt = dt*deadtime_correction(ts, urdhk)

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


def estimate_epxtime(skyvec, ts, quats, gti):
    pass



def make_sky_vec_exptime(skyvec, ts, quats, gti):
    quatint = Slerp(times, quats)
    tnew, maskgaps = make_ingti_times(times, gti)
    if tnew.size == 0:
        return Rotation(np.empty((0, 4), np.double)), np.array([])
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    if not urdhk is None:
        crate = (urdhk["EVENTS"][1:] - urdhk["EVENTS"][:-1])/\
            (urdhk["TIME"][1:] - urdhk["TIME"][:-1])
        icrate = interp1d((urdhk["TIME"][1:] + urdhk["TIME"][:-1])/2.,
                          crate, bounds_error=False, fill_value=np.median(crate))
        dt = dt*(1. - 770e-6*icrate(ts))

    qval = quatint(ts)


def hist_orientation_for_attfile(attdata, gti, v0=None):
    quats = get_gyro_quat(attdata)
    qval, dtn = make_small_steps_quats(attdata["TIME"], quats, gti)
    return hist_orientation(qval, dtn)

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

def make_hv_gti(hkdata):
    '''
    input: HK hdu extention of an L0 events fits file
    '''
    return hkdata["TIME"][maskedges(hkdata["HV"] < -95.) + [0, -1]]
