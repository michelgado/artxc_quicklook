from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti, gti_intersection, GTI
from .caldb import get_energycal, urdbkgsc
from scipy.interpolate import interp1d
import numpy as np
from functools import reduce

#import matplotlib.pyplot as plt

def stepsize(arr):
    return arr[1:] - arr[:-1]

def weigt_time_intervals(gtis, scales=urdbkgsc, defaultscale=1):
    """
    for the provided dictionaries, containing keys and corresponding gtis and scales (also dictrs containing weights for keys in gtis)
    computes overall weights insided gtis which determined by which of keys were active in time intervals
    """
    gtitot = reduce(lambda a, b: a | b, gtis.values())
    edges = np.unique(np.concatenate([g.arr.ravel() for g in gtis.values()]))
    te, mgaps = gtitot.make_tedges(edges)
    tc = (te[1:] + te[:-1])/2.
    se = np.ones(mgaps.size + 2, np.double)*np.sum([scales.get(key, defaultscale) for key in gtis])
    se[1:-1][np.logical_not(mgaps)] = 0
    for key, gti in gtis.items():
        mask = np.logical_not(gti.mask_outofgti_times(tc))
        se[1:-1][mask] -= scales.get(key, defaultscale)
    se[[0, -1]] = 0.

    def scalefunc(times):
        return se[te.searchsorted(times)]
    dt = np.zeros(se.size, np.double)
    dt[1:-1] = (te[1:] - te[:-1])
    cse = np.cumsum(se*dt)

    cumscalefunc = interp1d(te, cse[:-1], kind="linear", bounds_error=False, fill_value=(cse[0], cse[-1]))

    return te, mgaps, se, scalefunc, cumscalefunc

def make_overall_lc(times, urdgtis, dt=100, scales=urdbkgsc):
    """
    for stored background events occurence times (times) produces overall for 7 detectors background lightcurve with time resolution dt
    """
    gtitot = reduce(lambda a, b: a | b, urdgtis.values())
    te, mgaps = gtitot.arange(dt)
    teg, mgapsg, se, scalef, cscalef = weigt_time_intervals(urdgtis)
    cidx = times.searchsorted(te)
    csf = cscalef(te)
    ccts = stepsize(cidx)
    crate = ccts/stepsize(csf)
    crerr = np.sqrt(ccts)/stepsize(csf)

    crate[np.logical_not(mgaps)] = 0.

    def bkgrate(times, urdn=None, scales=scales, fill_value=0.):
        res = np.empty(times.size, np.double)
        idx = te.searchsorted(times)
        mask = (idx > 0) & (idx < crate.size)
        res[np.logical_not(mask)] = fill_value
        res[mask] = crate[idx[mask] - 1]
        return res*urdbkgsc.get(urdn, 1.)
    return te, mgaps, crate, crerr, bkgrate

def make_constantcounts_timeedges(times, gti, cts=1000):
    idx = times.searchsorted(gti.arr)
    csize = (idx[:, 1] - idx[:, 0])//cts + 1
    dt = (gti[:, 1] - gti[:, 0])/csize
    dtl = np.repeat(dt, csize + 1)
    cidx = np.cumsum(csize[:-1] + 1)
    dtl[cidx] = gti.arr[1:, 0] - gti.arr[:-1, 1]
    dtl[0] = 0.
    mgaps = np.zeros(dtl.size, np.bool)
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
