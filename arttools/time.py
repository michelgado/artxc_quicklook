import numpy as np
from .mask import edges as maskedges
from math import pi
from scipy.interpolate import interp1d

ARTDEADTIME = 770e-6 #seconds - ART-XC detector deadtime

class ART_TIME_ERROR(ValueError):
    pass


def get_gti(ffile):
    gti = np.array([ffile["GTI"].data["START"], ffile["GTI"].data["STOP"]]).T
    gti = gti_union(gti + [-0.5, +0.5]) + [+0.5, -0.5]
    return gti_intersection(gti, make_hv_gti(ffile["HK"].data))

def get_filtered_table(tabledata, gti):
    """
    tabledata - any numpy record like array, containing unique TIME value in each row
    """
    return np.concatenate([tabledata[s:e] for s, e in np.searchsorted(tabledata["TIME"], gti)])

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
    gti = np.copy(gti[gti[:, 1] > gti[:, 0]])
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

def gti_difference(gti1, gti2):
    """
    prdouce difference of the gti2 relative to gti1
    AHTUNG the order of the argunets is important
    result is
    gti in gti2 and not in gti1
    """
    check_gti_shape(gti1)
    check_gti_shape(gti2)

    if gti1.size == 0 or gti2.size == 0:
        return gti2

    gti1 = gti_union(gti1)
    gti2 = gti_union(gti2)

    gti3 = np.empty((gti1.shape[0] + 1, 2), np.double)
    gti3[:-1,1] = gti1[:,0]
    gti3[1:, 0] = gti1[:,1]
    gti3[[0, -1], [0, 1]] = gti2[[0, -1], [0, 1]]
    return gti_intersection(gti2, gti3)

class GTI(np.ndarray):

    def __new__(cls, arr):
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("gti array should be of shape Nx2")
        arr = arr[np.argsort(arr[:,0])]
        arr = arr[arr[:,0] < arr[:,1]]
        idx = np.argsort(np.ravel(arr))
        gtis = np.ravel(arr)[idx]
        mask = np.zeros(gtis.size, np.bool)
        mask[::2] = idx[::2] == np.arange(0, idx.size, 2)
        mask[1::2] = np.roll(mask[::2], -1)
        return np.asarray(gtis[mask].reshape((-1, 2))).view(cls)


    def __init__(self, arr):
        """
        read Nx2 array, which is assumed to be a set of 1D intervals
        make a regularization of these intervals - they should have ascending ordered
        and not intersect each other

        examples of usage:
            cretion of GTI::
                gti = GTI(np.array(Nx2)) produce
                gti = GTI.from_hdu(fits["GTI"].data)

            intersections of two gtis:
                gti3 = gti1 & gti2

            invert gti:
                gti3 = -gti1

            difference of gti1 relative to gti2
                gti3 = gti1 & -gti2

            sometime you want shift gtis, i.e. add mjdref etc
            this can be doe with usuall and and sub operations
            gti3 = gti1 +[-] floatval

            the product of these operations is also GTI
            you also can add different values to start and stop times
            this feature (not a bug) can be used for two userfull tricks:
                1) merge close intervals
                gtinew = gti + [-dt/2, +dt/2] + [dt/2, - dt/2]
                result of first sum is a new GTI instance with gti intervals close then dt is merged,
                the bound of the new GTI unfortunately is shifted, therefore with second addition we return then back
                2) iliminate small gti intervals
                gtinew = gti + [dt/2, - dt/2] + [-dt/2., dt/2]
                first action will iliminate small intervals, second return bounds of left intervals to initial state
        """


    @classmethod
    def from_hdu(cls, gtihdu):
        arr = np.array([gtihdu.data["START"], gtihdu["STOP"]]).T
        return clf(arr)

    def __and__(self, other):
        idx = np.searchsorted(self[:,1], other[:,1])
        idx[-1] = min(self.shape[0] - 1, idx[-1])
        arr1 = np.array([np.maximum(self[idx,0], other[:,0]), other[:,1]]).T
        arr1[-1, 1] = min(self[-1, 1], other[-1, 1])
        idx = np.searchsorted(other[:,1], self[:,1])
        idx[-1] = min(other.shape[0] - 1, idx[-1])
        arr2 = np.array([np.maximum(other[idx,0], self[:,0]), self[:,1]]).T
        arr2[-1, 1] = min(self[-1, 1], other[-1, 1])
        return GTI(np.concatenate([arr1, arr2]))

    def __neg__(self):
        arr = np.empty((self.shape[0] + 1, 2), np.double)
        arr[1:,0] = self[:, 1]
        arr[:-1, 1] = self[:, 0]
        arr[[0, -1], [0, 1]] = [-np.inf, np.inf]
        return GTI(arr)

    def __or__(self, other):
        return GTI(np.concatenate([self, other]))

    def __add__(self, val):
        return GTI(super().__add__(val))

    def __sub__(self, val):
        return GTI(super().__sub__(val))

    def __iadd__(self, val):
        pass

    def __mul__(self, val):
        return GTI(super().__mul__(val))

    def __div__(self, val):
        return GTI(super().__div__(val))


def merge_consecutive_kvea_gtis(urdfile):
    pass

def fill_att_gaps(attdata):
    pass

def mkgtimask(time, gti):
    mask = np.zeros(time.size, np.bool)
    idx = np.searchsorted(time, gti)
    for s, e in idx:
        mask[s:e] = True
    return mask

def tarange(dt, gti, conservecrit=0.1):
    dtl = dt - (gti[1] - gti[0])%dt
    dtl = -dtl/2. if dtl < dt*conservecrit else dtl/2.
    return np.arange(gti[0] - dtl, gti[1] + dtl + 0.1*abs(dtl), dt)

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

def deadtime_correction(urdhk):
    ts = urdhk["TIME"]
    tcrate = (urdhk["EVENTS"][1:] - urdhk["EVENTS"][:-1])/(ts[1:] - ts[:-1])
    dtcorr = interp1d((ts[1:] + ts[:-1])/2., (1. - ARTDEADTIME*tcrate),
                      bounds_error=False, fill_value=(1. - ARTDEADTIME*np.median(tcrate)))
    return dtcorr



def make_hv_gti(hkdata):
    '''
    input: HK hdu extention of an L0 events fits file
    '''
    return hkdata["TIME"][maskedges(hkdata["HV"] < -95.) + [0, -1]]
