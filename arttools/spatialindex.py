from .mosaic2 import  HealpixSky
from .orientation import FullSphereChullGTI, ChullGTI, get_attdata, get_slews_gti, AttDATA, pack_attdata
from .telescope import ANYTHINGTOURD
from multiprocessing.pool import ThreadPool
from .time import GTI, emptyGTI, get_gti
from math import sin, cos, pi, sqrt, log
from astropy.io import fits
from threading import Thread
from multiprocessing import Process, Queue
import numpy as np


def read_attdata(qin, qout):
    while True:
        fname = qin.get()
        attn = get_attdata(fname)
        qout.put(pack_attdata(attn, pi/180.*5./3600.))

class DataReader(object):
    def __init__(self, flist):
        self.flist = flist
        self.fit = iter(self.flist)
        self.qin = Queue(2)
        self.qout = Queue(2)
        self.executor = Process(target=read_attdata, args=(self.qin, self.qout))
        self.executor.start()


    def __iter__(self):
        self.current = 0
        self.qin.put(self.flist[0])
        return self


    def __next__(self):
        att = self.qout.get()
        self.current += 1
        if self.current == len(self.flist):
            raise StopItteration
        self.qin.put(self.flist[self.current])
        return att


def make_attdata_spatial_index(flist, chull=None, callback=None, mpnum=2):
    if chull is None:
        chull = FullSphereChullGTI()
        chull.split_on_equal_segments(4.)
    else:
        if not type(chull) in [ChullGTI, FullSphereChullGTI]:
            chull = ChullGTI(chull.vertices)

    datalist = DataReader(flist)

    for fname, attdata in zip(flist, datalist): #i in range(1, len(flist)):
        print("starting", fname)
        slews = get_slews_gti(attdata)
        for i, arr in enumerate((~slews).arr):
            attloc = attdata.apply_gti(GTI(arr), check_interpolation=False)
            if attloc.gti.exposure == 0:
                continue
            print("attloc done")
            for ch, g in attloc.get_covering_chulls():
                for sch in chull.get_all_child_intersect(ch.expand(pi/180.*26.6/60.)):  #26.5 arcmin includes vignetting as a hole
                    sch.update_gti_for_attdata(attloc)

        if not callback is None:
            callback(fname, chull)
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

