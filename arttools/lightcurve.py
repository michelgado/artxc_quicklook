from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti, gti_intersection
from .caldb import get_energycal
from scipy.interpolate import interp1d
import numpy as np
from functools import reduce

import matplotlib.pyplot as plt

urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.029129860773177,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

def get_time_intervals_weigts(gtis, scales):
    """
    for the provided dictionaries, containing keys and corresponding gtis and scales
    computes overall weights insided gtis which determined by which of keys were active in time intervals
    """
    gtitot = reduce(lambda a, b: a | b, gtis.values())
    edges = np.unique(np.concatenate([g.arr.ravel() for g in gtis.values()]))
    te, mgaps = gtitot.make_tedges(edges)
    tc = (te[1:] + te[:-1])[mgaps]/2.
    se = np.ones(mgaps.size, np.double)*np.sum(scales.values())
    se[mgaps] = 0
    for key, gti in gtis.items():
        mask = np.logical_not(gti.mask_outofgti_times(tc))
        se[mask] -= scale[key]
    return te, se, mgaps

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
