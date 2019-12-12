from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti, gti_intersection
from .caldb import get_energycal
from scipy.interpolate import interp1d
import numpy as np

import matplotlib.pyplot as plt

def groupevents(times, gti, cts=1000):
    tnew = times[gti.mask_outofgti_times(times)]
    tse = np.empty(tnew.size//cts + (2 if tnew.size%cts else 1))
    tse[:-1] = tnew[::cts]
    tse[[0, -1]] = gti.arr[[0, -1], [0, 1]] + [-1, 1]
    tse, mgaps = gti.make_tedges(tse)
    cs = tnew.searchsorted(tse)
    css = cs[1:] - cs[:-1]
    mg = css[np.logical_not(mgaps)]
    plt.hist(mg, np.arange(cts*2) - 0.5, histtype="step", log=True)
    print(css[-1])
    plt.hist(css[mgaps], np.arange(cts*2) - 0.5, histtype="step", log=True, lw=2)
    mss = np.logical_and(css < cts//2, mgaps)
    c1 = css[:-1][np.logical_not(mgaps[1:])]
    c2 = css[1:][np.logical_not(mgaps[:-1])]
    plt.hist(css[:-1][np.logical_not(mgaps[1:])], np.arange(cts*2) - 0.5, histtype="step", log=True)
    plt.hist(css[1:][np.logical_not(mgaps[:-1])], np.arange(cts*2) - 0.5, histtype="step", log=True)
    plt.hist(np.concatenate([c1, c2]), np.arange(cts*2) -0.5, histtype="step", log=True)
    mask = np.ones(tse.size, np.bool)
    print("check events", np.all(css[np.logical_not(mgaps)] == 0))
    plt.show()
    print(mg.max(), np.argmax(mg))
    mask[:-2][np.logical_and(mss[:-1], np.logical_not(mgaps[1:]))] = False
    mask[2:][np.logical_and(mss[1:], np.logical_not(mgaps[:-1]))] = False
    mask[-2] = css[-1] < cts//2
    tse = tse[mask]
    mgaps = mgaps[mask[:-1]]
    cs = tnew.searchsorted(tse)
    css = (cs[1:] - cs[:-1])[mgaps]
    print(css.min())
    css = css/(tse[1:] - tse[:-1])[mgaps]
    amin = np.argmin(css)
    t1 = tse.searchsorted(tse[1:][mgaps][amin])
    print(tse[t1], t1, tse.size, mgaps.sum())
    t2 = np.searchsorted(tse[1:][np.logical_not(mgaps)], tse[t1])
    print(tse[1:][np.logical_not(mgaps)][t2] - tse[t1])
    print(tse[1:][np.logical_not(mgaps)][t2 - 1] - tse[t1])
    print(tse[1:][np.logical_not(mgaps)][t2 + 1] - tse[t1])
    plt.show()

    """
    idx = tnew.searchsorted(gti.arr)
    csize = (idx[:, 1] - idx[:, 0])//cts + 1
    idx = np.arange(size.sum()) - np.repeat(np.cumsum([0,] + list(csize[:-1])), csize)
    t1 = tnew[np.repeat(idx[:, 0], csize) + idx]
    t2 = np.sort(np.concatenate([t1, gti.arr.ravel()]))
    ci = np.cumsum(csize)
    cs = (idx[:, 1] - idx[:, 0])/csize
    cadd = np.zeros(csize.sum(), np.int)
    cadd[ci] = 1
    tse = tnew[np.cumsum(np.repeat(cs, csize)).astype(np.int) + np.cumsum(cadd) - 1]
    tse[ci - 1] = gti.arr[:,1]
    tss = np.empty(tse.size, np.double)
    tss[1:] = tse[:-1]
    tss[ci[:-1]] = gti.arr[1:,0]
    tss[0] = gti.arr[0, 0]
    dt = tse - tss
    """
    return tse[1:][mgaps], css


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
