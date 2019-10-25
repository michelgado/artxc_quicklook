from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti, gti_intersection
from .caldb import get_energycal
from scipy.interpolate import interp1d
import numpy as np



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
