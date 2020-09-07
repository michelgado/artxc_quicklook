import numpy as np
from .mask import edges as maskedges
from .interval import Intervals
from math import pi
from scipy.interpolate import interp1d
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

    def make_tedges(self, ts, joinsize=0):
        """
        assuming that ts is a series of ascending evenly spaced points,
        produce new series of points, lying in the GTIs (with additional points at the edges of
        intervals if requeired), and mask, showing position of the gaps between points clusted from
        different intervals
        """
        gtloc = self & self.__class__(ts[[0, -1]])
        ts = ts[gtloc.mask_external(ts)]
        newts = np.unique(np.concatenate([ts, gtloc.arr.ravel()]))
        idxgaps = newts.searchsorted((gtloc.arr[:-1, 1] + gtloc.arr[1:, 0])/2.)
        maskgaps = np.ones(newts.size - 1 if newts.size else 0, np.bool)
        maskgaps[idxgaps - 1] = False
        #===============================================
        #join time intervals at the edges of the gti, if they are two short
        dt = np.diff(newts, 1)
        dtmed = np.median(dt)
        maskshort = np.ones(newts.size, np.bool)
        maskshort[idxgaps + 1] = dt[idxgaps] > dtmed*joinsize
        maskshort[idxgaps - 2] = dt[idxgaps - 2] > dtmed*joinsize
        if maskshort.size:
            maskshort[[0, -1]] = dt[[0, -1]] > dtmed*joinsize
        return newts[maskshort], maskgaps[maskshort[:-1]]

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

    def arange(self, dt, epoch=None, joinsize=0.2):
        tsize = np.ceil((self.arr[:,1] - self.arr[:, 0])/dt).astype(np.int) + 1
        ctot = np.empty(tsize.size + 1, np.int)
        ctot[1:] = np.cumsum(tsize)
        ctot[0] = 0
        arange = np.arange(ctot[-1]) - np.repeat(ctot[:-1], tsize)
        if epoch is None:
            t0 = np.median((self.arr.ravel() + dt/2.)%dt) - dt/2. + (self.arr[:, 0]//dt)*dt
            #t0 = self.arr[:, 0] - dt*(tsize%1)/2.
        else:
            t0 = self.arr[:, 0] - (self.arr[:, 0] - epoch)%dt
        te = arange*dt + np.repeat(t0, tsize)
        return self.make_tedges(te, joinsize)

tGTI = GTI([-np.inf, np.inf])
emptyGTI = GTI(np.empty((0, 2)))

def get_gti(ffile, gtiextname="GTI", excludebki=True):
    if not gtiextname is None:
        gti = GTI(np.array([ffile[gtiextname].data["START"], ffile[gtiextname].data["STOP"]]).T)
    else:
        gti = tGTI
        for hdu in ffile:
            if hdu.name in ["GTI", "STD_GTI", "KVEA_GTI"]:
                gti = gti & GTI.from_hdu(hdu)

    gti.merge_close_intervals(0.5)
    gti = gti & make_hv_gti(ffile["HK"].data)
    if excludebki:
        gti = gti & ~make_bki_gti(ffile)
    return gti


def make_bki_gti(ffile):
    if 'BKI_STATE' in ffile["HK"].data.dtype.names:
        bkiedges = maskedges(ffile["HK"].data["BKI_STATE"] != 1) + [0, -1]
        bkigti = GTI(ffile["HK"].data["TIME"][bkiedges]) + [-5, 5]
    else:
        dt = np.diff(ffile["HK"].data["TIME"], 1)
        rate = np.diff(ffile["HK"].data["EVENTS"].astype(np.int), 1)/dt
        bkigti = GTI(ffile["HK"].data["TIME"][maskedges(rate > 200) + [1, 0]]) + [-10, 10]
    return bkigti

def get_filtered_table(tabledata, gti):
    """
    tabledata - any numpy record like array, containing unique TIME value in each row
    """
    return tabledata[gti.mask_external(tabledata["TIME"])]

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

def make_ingti_times(time, ggti, stick_frac=0.5):
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
