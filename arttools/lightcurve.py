from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti, gti_intersection, GTI
from .caldb import get_energycal, urdbkgsc
from collections import defaultdict
from scipy.interpolate import interp1d
from scipy.integrate import quad
import numpy as np
from functools import reduce

#import matplotlib.pyplot as plt

def stepsize(arr):
    return arr[1:] - arr[:-1]

def weigt_time_intervals(gtis, scales={}, defaultscale=1):
    """
    for the provided dictionaries, containing keys and corresponding gtis and scales (also dictrs containing weights for keys in gtis)
    computes overall weights insided gtis which determined by which of keys were active in time intervals
    """
    gtitot = reduce(lambda a, b: a | b, gtis.values())
    edges = np.unique(np.concatenate([g.arr.ravel() for g in gtis.values()]))
    te, mgaps = gtitot.make_tedges(edges)
    tc = (te[1:] + te[:-1])/2.
    se = np.ones(mgaps.size + 2, np.double)*np.sum([scales.get(key, defaultscale) for key in gtis])
    #se[1:-1][np.logical_not(mgaps)] = 0
    for key, gti in gtis.items():
        mask = np.logical_not(gti.mask_external(tc))
        se[1:-1][mask] -= scales.get(key, defaultscale)
    se[[0, -1]] = 0.

    def scalefunc(times):
        return se[te.searchsorted(times)]
    dt = np.zeros(se.size, np.double)
    dt[1:-1] = (te[1:] - te[:-1])
    cse = np.cumsum(se*dt)

    cumscalefunc = interp1d(te, cse[:-1], kind="linear", bounds_error=False, fill_value=(cse[0], cse[-1]))

    return te, mgaps, se, scalefunc, cumscalefunc


def sum_lcs(tes, lcs, gaps=None, sigmas=None):
    if gaps is None:
        gaps = [np.ones(l.size, bool) for l in lcs]
    if sigmas is None:
        sigmas = [np.ones(l.size) for l in lcs]
    tetot = np.unique(np.concatenate(tes))
    tec = (tetot[1:] + tetot[:-1])/2.
    idxs = [np.searchsorted(te, tec) - 1 for te in tes]
    lest = np.zeros(tec.size, np.double)
    west = np.zeros(tec.size, np.double)
    for i, idx in enumerate(idxs):
        print(lcs[i].shape)
        print(idx)
        print(lcs[i][idx])
        lest = lest + lcs[i][idx]/sigmas[i][idx]**2.*gaps[i][idx]
        west = west + 1./sigmas[i][idx]**2.
    return tetot, lest/west

def join_lcs(urdbkg):
    tetot = np.unique(np.concatenate([d.te for d in urdbkg.values()]))
    crsum = np.zeros(tetot.size - 1, np.double)
    for d in urdbkg.values():
        cv = interp1d(d.te, np.concatenate([[0,], (d.crate*np.diff(d.te)).cumsum()]), bounds_error=False, fill_value=(0, np.sum(d.crate*np.diff(d.te))))
        crsum += np.diff(cv(tetot))/np.diff(tetot)
    return Bkgrate(tetot, crsum)


class Bkgrate(object):

    def __init__(self, te, crate, dtcorr=None):
        self.te = te
        self.crate = crate
        self.dtcorr = dtcorr

    def __call__(self, times):
        if self.dtcorr is None:
            return self.crate[np.minimum(np.searchsorted(self.te, times) - 1, self.crate.size - 1)]
        else:
            return self.crate[np.minimum(np.searchsorted(self.te, times) - 1, self.crate.size - 1)]*self.dtcorr(times)

    @classmethod
    def from_events(cls, times, gti, dt=None, scales=None):
        if dt is None:
            te = gti.arr.ravel()
            gaps = np.ones(te.size - 1, bool)
            gaps[1::2] = False
        else:
            te, gaps = gti.arange(dt)

        if type(scales) == interp1d:
            fill_value = i2.y[[0, -1]] if np.isnan(scales.fill_value) else fill_value
            tii, gloc = gti.make_tedges(np.unique(np.cocnatenate([te, scales.x])))
            tcc = (tii[1:] + tii[:-1])/2.
            dtt = (tii[1:] - tii[:-1])
            sloc = np.diff(interp(te, tii[1:], np.cumsum(scales(tcc)*gaps*dtt), left=fill_value[0], right=fill_value[1]))
        else:
            pass
        #rate = np.searchsorted

    def set_dtcorr(self, dtcorr):
        self.dtcorr = dtcorr

    def median(self, gti):
        tc = (self.te[:-1] +self.te[1:])/2.
        return np.median(self.crate[gti.mask_external(tc)])

    def _scale(self, val):
        return Bkgrate(self.te, self.crate*val)

    def __and__(self, other):
        times = np.concatenate(self.te, other.te)
        rates = np.concatenate(self.crate, other.crate)
        idx = np.argsort(times)
        times = times[idx]
        rates = rates[idx]

    def integrate_in_timebins(self, te, dtcorr=None):
        dtcorr = self.dtcorr
        if dtcorr is None:
            tloc = np.unique(np.concatenate([te,  self.te]))
            tc = (tloc[1:] + tloc[:-1])/2.
            lct = self(tc)*np.diff(tloc)
            idx = np.searchsorted(te, tc) - 1
            m = (idx >= 0) & (idx < te.size - 1)
            idx = idx[m]
            lc = np.zeros(te.size - 1)
            np.add.at(lc, idx, lct[m])
        else:
            if type(dtcorr) == interp1d:
                ttot = np.unique(np.concatenate([self.times, dtcorr.x]))
                d = np.cumsum(self(ttot)*dtcorr(ttot))
                di = interp1d(ttot, d, bounds_error=False, fill_value=(0., d[-1]))
                lc = np.diff(di(te))
            else:
                lc = np.array([quad(lambda t: self(t)*dtcorr(t), s, e)[0] for s, e in zip(te[:-1], te[1:])])
        return lc

def make_overall_lc(times, urdgtis, dt=100, scales=urdbkgsc, dtcorr={}):
    """
    for stored background events occurence times (times) produces overall for 7 detectors background lightcurve with time resolution dt
    """
    gtitot = reduce(lambda a, b: a | b, urdgtis.values())
    te, mgaps = gtitot.arange(dt, joinsize=0.8)
    if dtcorr != {}:
        dtt = np.zeros(te.size - 1, np.double)
        for urdn in urdgtis:
            tee, g = urdgtis[urdn].make_tedges(np.unique(np.concatenate([dtcorr[urdn].x, te])))
            tec = (tee[1:] + tee[:-1])[g]/2.
            dtcm = dtcorr[urdn](tec)*np.diff(tee)[g]*scales.get(urdn, 1.)
            idx = np.searchsorted(te, tec) - 1
            print(idx.size, dtcm.size, tee.size, g.sum(), scales.get(urdn, 1.))
            np.add.at(dtt, idx, dtcm)
        cs = np.diff(np.searchsorted(times, te))
        crate = cs/dtt
        crerr = np.sqrt(cs)/dtt
    else:
        print("urdweights case")
        teg, mgapsg, se, scalef, cscalef = weigt_time_intervals(urdgtis)
        cidx = times.searchsorted(te)
        csf = cscalef(te)
        print(np.diff(csf, 1)[mgaps].min())
        ccts = np.diff(cidx, 1)
        crate = ccts/np.diff(csf, 1)
        crerr = np.sqrt(ccts)/np.diff(csf, 1)

        crate[np.logical_not(mgaps)] = 0.

    return te, mgaps, crate, crerr, Bkgrate(te, crate)

def make_constantcounts_timeedges(times, gti, cts=1000):
    idx = times.searchsorted(gti.arr)
    csize = (idx[:, 1] - idx[:, 0])//cts + 1
    dt = (gti[:, 1] - gti[:, 0])/csize
    dtl = np.repeat(dt, csize + 1)
    cidx = np.cumsum(csize[:-1] + 1)
    dtl[cidx] = gti.arr[1:, 0] - gti.arr[:-1, 1]
    dtl[0] = 0.
    mgaps = np.zeros(dtl.size, bool)
    mgaps[cidx - 1] = False
    return gti.arr[0, 0] + dtl.cumsum(), mgaps

def make_lightcurve(times, gti):
    gti = get_gti(urdfile)
    ts = np.concatenate([tarange(dtbkg, g) for g in gti])
    tnew, maskgaps = make_ingti_times(ts, gti + [dtbkg*1e-6, -dtbkg*1e-6])
    lcs = np.searchsorted(tevt, tnew)
    lcs = lcs[1:] - lcs[:-1]
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    maskzero = dt > 0
    bkgrate = lcs[maskgaps][maskzero]/dt[maskzero]
    return ts[maskzero], bkgrate

def get_overall_countrate(urdfile, elow, ehigh, ingoreedgestrips=True):
    urddata = urdfile["EVENTS"].data
    print(urddata.size)
    energy, xc, yc, grade = get_events_energy(urddata, urdfile["HK"].data, get_energycal(urdfile))
    if ingoreedgestrips:
        mask = np.all([energy > elow, energy < ehigh, urddata["RAW_X"] > 0, urddata["RAW_Y"] > 0, urddata["RAW_X"] < 47, urddata["RAW_X"] < 47, grade > -1, grade < 10], axis=0)
    else:
        mask = np.all([energy > elow, energy < ehigh, grade > -1, grade < 10], axis=0)
    tevt = urddata["TIME"][mask]
    dtmed = np.median(tevt[1:] - tevt[:-1])
    dtbkg = 1000.*dtmed
    gti = get_gti(urdfile)

    ts = np.concatenate([tarange(dtbkg, g) for g in gti])
    tnew, maskgaps = make_ingti_times(ts, gti + [dtbkg*1e-6, -dtbkg*1e-6])
    lcs = np.searchsorted(tevt, tnew)
    lcs = lcs[1:] - lcs[:-1]
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    maskzero = dt > 0
    bkgrate = lcs[maskgaps][maskzero]/dt[maskzero]
    return ts[maskzero], bkgrate
