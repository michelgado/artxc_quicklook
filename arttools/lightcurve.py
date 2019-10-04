from .energy import get_events_energy
from .time import deadtime_correction, make_ingti_times, tarange, get_gti
from .caldb import get_energycal
from scipy.interpolate import interp1d
import numpy as np


def get_overall_countrate(urdfile, elow, ehigh):
    energy, xc, yc, grade = get_events_energy(urdfile["EVENTS"].data, urdfile["HK"].data, get_energycal(urdfile))
    tevt = urdfile["EVENTS"].data["TIME"][(energy > elow) & (energy < ehigh)]
    print(tevt.size)
    dtmed = np.median(tevt[1:] - tevt[:-1])
    print("median arrival time", dtmed)
    dtbkg = 1000.*dtmed
    print(dtbkg)
    gti = get_gti(urdfile)
    print("gti", gti, gti[:,1] - gti[:,0])
    print(tarange(dtbkg, gti[0]))

    print(gti)
    ts = np.concatenate([tarange(dtbkg, g) for g in gti])
    tnew, maskgaps = make_ingti_times(ts, gti)
    print("tnew", tnew)
    lcs = np.searchsorted(tevt, tnew)
    lcs = lcs[1:] - lcs[:-1]
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]
    bkgrate = lcs[maskgaps]/dt
    return ts, bkgrate
