import numpy as np
from .mask import edges as maskedges
from math import pi
from scipy.interpolate import interp1d
import astropy.time as atime

ARTDEADTIME = 770e-6 #seconds - ART-XC detector deadtime

class ART_TIME_ERROR(ValueError):
    pass

class GTI(object):
    """
    this class provides a number of userfull function to work with consequitive ordered unintersected 1d intervals
    in this code this class is used to store Good Time Intervals
    most userfull functions of the class are
    *intersection GTI1 & GTI2
    *unification GTI1 | GTI2
    and negation -GTI1

    the class also has method to produce masks of exclude from intervals times
    mask_outofgti_times
    """

    def _regularize(self, arr=None):
        if arr is None:
            arr = self.arr
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("gti array should be of shape Nx2")
        arr = arr[arr[:,0] < arr[:,1]]
        us, iidx = np.unique(arr[:, 0], return_inverse=True)
        arrn = np.empty((us.size, 2), np.double)
        arrn[:, 0] = us
        arrn[:, 1] = -np.inf
        np.maximum.at(arrn[:, 1], iidx, arr[:, 1])
        arr = arrn
        idx = np.argsort(np.ravel(arr))
        gtis = np.ravel(arr)[idx]
        mask = np.zeros(gtis.size, np.bool)
        mask[::2] = idx[::2] == np.arange(0, idx.size, 2)
        mask[1::2] = np.roll(mask[::2], -1)
        arr = np.asarray(gtis[mask].reshape((-1, 2)))
        return arr


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
        if issubclass(self.__class__, type(arr)):
            self.arr = np.copy(arr.arr)
        else:
            self.arr = self._regularize(np.asarray(arr).reshape((-1, 2)))

    @property
    def shape(self):
        return self.arr.shape

    @property
    def size(self):
        return self.arr.size

    def __repr__(self):
        return self.arr.__repr__()

    @classmethod
    def from_hdu(cls, gtihdu):
        arr = np.array([gtihdu.data["START"], gtihdu["STOP"]]).T
        return cls(arr)

    @classmethod
    def from_tedges(cls, ts):
        arr = np.array([ts[:-1], ts[1:]]).T
        return cls(arr)

    def make_tedges(self, ts):
        """
        assuming that ts is a series of intervals edges,
        produce new series of intervals edges, lying in the GTIs
        """
        gtloc = self & self.__class__(ts[[0, -1]])
        ts = ts[gtloc.mask_outofgti_times(ts)]
        newts = np.unique(np.concatenate([ts, gtloc.arr.ravel()]))
        idxgaps = newts.searchsorted((gtloc.arr[:-1, 1] + gtloc.arr[1:, 0])/2.)
        maskgaps = np.ones(newts.size - 1 if newts.size else 0, np.bool)
        maskgaps[idxgaps - 1] = False

        return newts, maskgaps

    def mask_outofgti_times(self, ts):
        """
        creates bitwise mask for time series, mask is True for times located in any of good time intervals
        """
        return self.arr.ravel().searchsorted(ts)%2 == 1

    def __and__(self, other):
        if self.size == 0 or other.size == 0:
            return emptyGTI
        tt = np.concatenate([self.arr.ravel(), other.arr.ravel()])
        ms = np.ones(tt.size, np.int8)
        ms[1::2] = -1
        idx = np.argsort(tt)
        tt = tt[idx]
        gti = np.lib.stride_tricks.as_strided(tt, (tt.size - 1, 2), tt.strides*2)
        #make empty GTI instance, all condintion on GTI already fullfield, no regularization required
        gres = self.__class__.__new__(self.__class__)
        gres.arr = np.copy(gti[np.cumsum(ms[idx][:-1]) == 2])
        return gres

    def merge_joint(self):
        mask = np.ones(self.arr.shape[0] + 1, np.bool)
        mask[1:-1] = self.arr[1:, 0] != self.arr[:-1, 1]
        mask = np.lib.stride_tricks.as_strided(mask, (mask.size - 1, 2), mask.strides*2)
        self.arr = self.arr[mask].reshape((-1, 2))

    @property
    def exposure(self):
        return np.sum(self.arr[:,1] - self.arr[:,0])

    def __neg__(self):
        arr = np.empty((self.shape[0] + 1, 2), np.double)
        arr[1:,0] = self[:, 1]
        arr[:-1, 1] = self[:, 0]
        arr[[0, -1], [0, 1]] = [-np.inf, np.inf]
        return GTI(arr)

    def __getitem__(self, *args):
        return self.arr.__getitem__(*args)

    def __or__(self, other):
        return GTI(np.concatenate([self.arr, other.arr]))

    def __add__(self, val):
        return GTI(self.arr + val)

    def __sub__(self, val):
        return GTI(self.arr - val)

    def __iadd__(self, val):
        self.arr = self._regularize(self.arr + val)
        return self

    def __isub__(self, val):
        self.arr = self._regularize(self.arr - val)
        return self

    def __idiv__(self, val):
        self.arr = self._regularize(self.arr/val)

    def __imul__(self, val):
        self.arr = self._regularize(self.arr*val)

    def __mul__(self, val):
        return GTI(self.arr*val)

    def __div__(self, val):
        return GTI(super().__div__(val))

    def merge_close_intervals(self, dt):
        """
        merge GTI intervals which separated by less then dt
        """
        self.arr = self._regularize(self.arr + [-dt/2., dt/2.]) - [-dt/2., dt/2.]

    def remove_short_intervals(self, dt):
        """
        remove intervals shorter then dt
        """
        self.arr = self._regularize(self.arr + [dt/2., -dt/2.]) - [dt/2., + dt/2.]

    def searchtimes(self, tseries):
        return np.searchsorted(tseries, self.arr)

    def local_arange(self, dt, epoch=None):
        te = np.concatenate([np.minimum(e, np.arange(int(np.ceil((e - s)/dt)) + 1)*dt + s) for s, e in self.arr])
        return self.make_tedges(te)

    def arange(self, dt, epoch=None, joinsize=0.2):
        tsize = np.ceil((self.arr[:,1] - self.arr[:, 0])/dt).astype(np.int) + 1
        ctot = np.empty(tsize.size + 1, np.int)
        ctot[1:] = np.cumsum(tsize)
        ctot[0] = 0
        arange = np.arange(ctot[-1]) - np.repeat(ctot[:-1], tsize)
        if epoch is None:
            t0 = self.arr[:, 0] - dt*(tsize%1)/2.
        else:
            t0 = self.arr[:, 0] - (self.arr[:, 0] - epoch)%dt
        te = arange*dt + np.repeat(t0, tsize)
        return self.make_tedges(te)



tGTI = GTI([-np.inf, np.inf])
emptyGTI = GTI([])


def get_gti(ffile, gtiextname="GTI"):
    gti = GTI(np.array([ffile[gtiextname].data["START"], ffile[gtiextname].data["STOP"]]).T)
    gti.merge_close_intervals(0.5)
    return gti & make_hv_gti(ffile["HK"].data)

def get_filtered_table(tabledata, gti):
    """
    tabledata - any numpy record like array, containing unique TIME value in each row
    """
    return tabledata[gti.mask_outofgti_times(tabledata["TIME"])]

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
    Ahtung #2!!! this function was used before GTI class was introduced

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
    """
    produces effectivenesess of the events registration depending on overall countrate

    ART-XC detectors have deadtime. Therefore photons which reach detectors instantly after previous are not registered.
    if the overall countrate in the decector is k events per second, we want to know real expected countrate n
    if real count rate is n then expectration time for the next photon is distributed
    as n exp(-tn), probability to lost l events during the deadtime P(l) = (n\tau)^l/l! exp(-n\tau)
    mean number of lost events per event: n\tau
    therefore if the are k eventin during T then real expected countrate n = k*(1 + ntau)/T
    k/T = c - observed countrate
    n = c(1 + n \tau)
    n(1 - c\tau) = c
    n = c/(1 - c\tau)
    """
    ts = urdhk["TIME"]
    dt = (ts[1:] - ts[:-1])
    mask = (dt > 1.) & (urdhk["EVENTS"][1:] > urdhk["EVENTS"][:-1])
    tcrate = (urdhk["EVENTS"][1:] - urdhk["EVENTS"][:-1])/dt
    print("received tcrate", tcrate)
    dtcorr = interp1d((ts[1:] + ts[:-1])[mask]/2., (1. - ARTDEADTIME*tcrate[mask]),
                      bounds_error=False, fill_value=(1. - ARTDEADTIME*np.median(tcrate)))
    return dtcorr

def get_hdu_times(hdu):
    return atime.Time(hdu.header["MJDREF"], format="mjd") + \
            atime.TimeDelta(hdu.data["TIME"], format="sec")

def make_hv_gti(hkdata):
    '''
    input: HK hdu extention of an L0 events fits file
    '''
    return GTI(hkdata["TIME"][maskedges(hkdata["HV"] < -95.) + [0, -1]])
