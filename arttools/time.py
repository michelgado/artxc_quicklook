import numpy as np
from .mask import edges as maskedges
from scipy.signal import medfilt
from .filters import Intervals
from math import pi
from .aux import interp1d
from .caldb import get_deadtime_for_dev
from functools import reduce
import astropy.time as atime
from astropy.table import Table

ARTDEADTIME = 770e-6 #seconds - ART-XC detector deadtime

class ART_TIME_ERROR(ValueError):
    pass

class GTI(Intervals):
    """
    this class provides a number of userfull function to work with consequitive ordered unintersected 1d intervals
    in this code this class is used to store Good Time Intervals
    most userfull functions of the class are
    *intersection GTI1 & GTI2
    *unification GTI1 | GTI2
    and negation -GTI1

    the class also has method to produce masks of exclude from intervals times
    mask_external
    """

    @classmethod
    def from_hdu(cls, gtihdu):
        arr = np.array([gtihdu.data["START"], gtihdu.data["STOP"]]).T
        return cls(arr)

    @classmethod
    def from_tedges(cls, ts):
        arr = np.array([ts[:-1], ts[1:]]).T
        return cls(arr)


    def filter_data(self, data):
        try:
            return data[self.mask_external(data["TIME"])]
        except IndexError:
            #TODO in the log here should be a worning, that user is responsible himself for input data to be a time
            return data[self.mask_external(data)]

    @property
    def exposure(self):
        return np.sum(self.arr[:,1] - self.arr[:,0])

    def searchtimes(self, tseries):
        return np.searchsorted(tseries, self.arr)

    def local_arange(self, dt, epoch=None):
        """
        for each interval in the set produces a series of evenly spaced points,
        return this points with mask, showing position of the gaps between intervals
        """
        te = np.concatenate(
                [np.minimum(e, np.arange(int(np.ceil((e - s)/dt)) + 1)*dt + s) \
                for s, e in self.arr])
        return self.make_tedges(te)

    def arange(self, dt, joinsize=0.2):
        t0 = np.median(self.arr[:, 0]%dt) + (self.arr[0, 0]//dt - 1)*dt
        te = np.unique(np.concatenate([np.arange((s - t0)//dt + 1, (e - t0)//dt + 1)*dt + t0 for s, e in self.arr]))
        eidx = np.searchsorted(te, self.arr)
        mempty = eidx[:, 0] != eidx[:, 1]
        sidx = np.searchsorted(te, self.arr[mempty, 0])
        m1 = np.ones(te.size, bool)
        m1[sidx] = te[sidx] - self.arr[mempty, 0] > dt*joinsize
        #print(np.array([self.arr[mempty, 0], te[sidx], self.arr[mempty, 1], te[sidx] - self.arr[mempty, 0], np.array(m1, dtype=np.double)]).T)
        te = te[m1]

        eidx = np.searchsorted(te, self.arr)
        mempty = eidx[:, 0] != eidx[:, 1]
        sidx = np.searchsorted(te, self.arr[mempty, 1]) - 1
        m1 = np.ones(te.size, bool)
        m1[sidx] = self.arr[mempty, 1] - te[sidx] > dt*joinsize
        te = te[m1]

        te = np.unique(np.concatenate([self.arr.ravel(), te]))
        mgaps = self.mask_external((te[1:] + te[:-1])/2.)

        return te, mgaps


tGTI = GTI([-np.inf, np.inf])
emptyGTI = GTI(np.empty((0, 2)))


def board_time_to_jyear(timeseries):
    return 2000. + (timeseries - 54005.152032)/31557600.0


def get_gti(ffile, gtiextname=None, excludebki=True, merge_interval_dt=None, usehkgti=True):
    if not gtiextname is None:
        try:
            gti = GTI(np.array([ffile[gtiextname].data["START"], ffile[gtiextname].data["STOP"]]).T)
        except Exception:
            gti = GTI(np.array([ffile[gtiextname].data["TSTART"], ffile[gtiextname].data["TSTOP"]]).T)
    else:
        gti = tGTI
        for hdu in ffile:
            if hdu.name in ["GTI", "STD_GTI", "KVEA_GTI"]:
                gti = gti & GTI.from_hdu(hdu)

    if not merge_interval_dt is None:
        gti.merge_close_intervals(0.5)
    else:
        gaps = gti.arr.ravel()[1:-1].reshape((-1, 2)) # get bounds of the gaps between gti
        #print("min diff", np.min(gti.arr[:, 1] - gti.arr[:, 0]))
        crate = -np.subtract.reduce(np.searchsorted(ffile["EVENTS"].data["TIME"], gti.arr), axis=1)/(gti.arr[:,1] - gti.arr[:,0])
        crate[crate == 0] = np.median(crate)
        garr = np.copy(gti.arr)
        garr[1:, 0] -= 7./crate[1:]
        garr[:-1, 1] += 7./crate[:-1]
        gti = GTI(garr)

    if usehkgti:
        gti = gti & make_hv_gti(ffile["HK"].data)
    if excludebki:
        gti = gti & ~make_bki_gti(ffile)
    return gti


def make_bki_gti(ffile):
    if 'BKI_STATE' in ffile["HK"].data.dtype.names:
        bkiedges = maskedges(ffile["HK"].data["BKI_STATE"] != 1) + [0, -1]
        bkigti = GTI(ffile["HK"].data["TIME"][bkiedges]) + [-10, 10]
    else:
        dt = np.diff(ffile["HK"].data["TIME"], 1)
        rate = np.diff(ffile["HK"].data["EVENTS"].astype(int), 1)/dt
        bkigti = GTI(ffile["HK"].data["TIME"][maskedges(rate > 200) + [1, 0]]) + [-10, 10]
    return bkigti

def get_filtered_table(tabledata, gti):
    """
    tabledata - any numpy record like array, containing unique TIME value in each row
    """
    return tabledata[gti.mask_external(tabledata["TIME"])]

def deadtime_correction(urdhk, urdn=None): #deadtime=ARTDEADTIME):
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
    if urdn is None:
        deadtime = ARTDEADTIME
    else:
        deadtime = get_deadtime_for_dev(urdn)

    ts = urdhk["TIME"]
    dt = (ts[1:] - ts[:-1])
    mask = (dt > 1.) & (urdhk["EVENTS"][1:] > urdhk["EVENTS"][:-1])
    tcrate = (urdhk["EVENTS"][1:] - urdhk["EVENTS"][:-1])/dt
    dtcorr = interp1d(ts[1:][mask], (1. - deadtime*tcrate[mask]), kind="next",
                      bounds_error=False, fill_value=(1. - deadtime*np.median(tcrate)))
    return dtcorr

def deadtime_correction_from_evt(urddata, cscale, dt=1, med_filter_dt=1000):
    te, gaps = urddaat.filters["TIME"].arange(dt)
    crate = np.diff(np.searchsorted(urddata["TIME"], te))/np.diff(te)
    mrate = medfilt(cs, 3)
    mrate[[0, -1]] = mrate[[1, -2]]
    nmed = int(dt/med_filter_dt)
    nmed = nmed + (nmed + 1)%2
    nrepeat = np.minimum(np.maximum((np.diff(te)/dt).astype(int), 1), nmed)
    nrepeat[[0, -1]] = nmed//2 + 1
    mgaps = np.repeat(gaps, nrepeat)
    mgaps[:nmed//2] = False
    mgaps[-(nmed//2):] = False
    mcrate = medfilt(np.repeat(crate, nrepeat), nmed)[mgaps]
    raise NotImpementedError("it seems that the median background count rate doesn't allow to trace deadtime variations with 1s resolution")
    return interp1d(te[1:],)


def combine_urddtc(urddtc, scales):
    return reduce(lambda a, b: a + b, [d._scale(s) for d, s in zip(urddtc.values(), scales)])


def get_hdu_times(hdu):
    return atime.Time(hdu.header["MJDREF"], format="mjd") + \
            atime.TimeDelta(hdu.data["TIME"], format="sec")

def make_hv_gti(hkdata):
    '''
    input: HK hdu extention of an L0 events fits file
    '''
    return GTI(hkdata["TIME"][maskedges(hkdata["HV"] < -95.) + [0, -1]])


#==================================================================================================
#below is old gti version, which spared for the compatibility reasons

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
    mask = np.zeros(gtis.size, bool)
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
    mask = np.zeros(time.size, bool)
    idx = np.searchsorted(time, gti)
    for s, e in idx:
        mask[s:e] = True
    return mask

def tarange(dt, gti, conservecrit=0.1):
    dtl = dt - (gti[1] - gti[0])%dt
    dtl = -dtl/2. if dtl < dt*conservecrit else dtl/2.
    return np.arange(gti[0] - dtl, gti[1] + dtl + 0.1*abs(dtl), dt)

def make_ingti_times(time, ggti, stick_frac=0.5):
    gti = gti_intersection(np.array([time[[0, -1]],]), ggti)
    idx = np.searchsorted(time, gti)
    tnew = np.empty(np.sum(idx[:,1] - idx[:,0]) + 2*idx.shape[0], np.double)
    cidx = np.empty(idx.shape[0] + 1, int)
    cidx[1:] = np.cumsum(idx[:,1] - idx[:,0] + 2)
    cidx[0] = 0
    for i in range(idx.shape[0]):
        tnew[cidx[i]+1: cidx[i+1]-1] = time[idx[i,0]:idx[i,1]]
        tnew[cidx[i]] = gti[i, 0]
        tnew[cidx[i + 1] - 1] = gti[i, 1]
    maskgaps = np.ones(max(tnew.size - 1, 0), bool)
    maskgaps[cidx[1:-1] - 1] = False
    return tnew, maskgaps

def get_global_time(obt, ctable):
    cidx = np.searchsorted(ctable["OBT"], obt) - 1
    deltat = 51544. + obt/86400. - ctable["T0"][cidx]  #Molkov's
    dt = ctable["ph0"][cidx] + (ctable["p0"][cidx] + ctable["pdot0"][cidx]*deltat)*deltat
    return dt

def add_utc_timecol(urddata, ctable):
    tnew = urddata["TIME"] + get_global_time(urddata["TIME"], ctable)
    told = np.copy(urddata["TIME"])
    urddata["TIME"] = tnew
    d = np.lib.recfunctions.append_fields(urddata.data, ["OBT"], [told,], usemask=False)
    return urddata.__class__(d, urddata.urdn, urddata.filters)   # see arttools/containers for Urddata initialization
