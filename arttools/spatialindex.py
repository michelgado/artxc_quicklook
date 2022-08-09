from .mosaic2 import  HealpixSky
from .orientation import FullSphereChullGTI, ChullGTI, get_attdata, get_slews_gti
from .telescope import ANYTHINGTOURD
from .time import GTI, emptyGTI, get_gti
from math import sin, cos, pi, sqrt, log
from astropy.io import fits
import numpy as np


def make_attdata_spatial_index(flist, chull=None):
    if chull is None:
        chull = FullSphereChullGTI()
    else:
        chull = ChullGTI(chull.vertices)
    chull.split_on_equal_segments(5.)
    for fname in flist:
        print('starting:',  fname)
        attdata = get_attdata(fname)
        slews = get_slews_gti(attdata)
        print("chulls")
        for i, arr in enumerate((~slews).arr):
            attloc = attdata.apply_gti(GTI(arr))
            print("attloc done")
            for ch, g in attloc.get_covering_chulls():
                print("chull", ch.area)
                for sch in chull.get_all_child_intersect(ch.expand(pi/180.*26.5/60.)):
                    sch.update_gti_for_attdata(attloc)
    return chull 

def make_attdata_gti_index(flist):
    gtot = emptyGTI
    names = []
    times = []
    for fname in flist:
        attdata = get_attdata(fname)
        slews = get_slews_gti(attdata)
        mygti = attdata.gti & ~slews & ~gtot
        gtot = gtot | mygti
        idx = np.searchsorted(times, mygti.arr[:, 0])
        print(idx)
        for i in idx[::-1]:
            names.insert(i, fname)
        times = np.sort(np.concatenate([times, mygti.arr[:, 0]]))
    return times, names

def make_urd_gti_index(flist):
    gtot = emptyGTI
    names = []
    times = []
    for fname in flist:
        mygti = get_gti(fits.open(fname)) & ~gtot
        gtot = gtot | mygti
        idx = np.searchsorted(times, mygti.arr[:, 0])
        for i in idx[::-1]:
            names.insert(i, fname)
        times = np.sort(np.concatenate([times, mygti.arr[:, 0]]))
    return times, names

def get_flist_from_index(index, gti):
    indextimes = index[0]
    indexnames = index[1]
    te, gaps = gti.make_tedges(indextimes)
    tc = (te[1:] + te[:-1])[gaps]/2.
    idx = np.searchsorted(indextimes, tc) - 1
    return [indexnames[i] for i in np.unique(idx)]

