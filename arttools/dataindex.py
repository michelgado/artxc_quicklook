from .mosaic2 import  HealpixSky
from .orientation import FullSphereChullGTI, ChullGTI, get_attdata, get_slews_gti, AttDATA, pack_attdata
from .telescope import ANYTHINGTOURD
from multiprocessing.pool import ThreadPool
from .time import GTI, emptyGTI, get_gti
from .filters import Intervals
from math import sin, cos, pi, sqrt, log
from astropy.io import fits
from threading import Thread
from multiprocessing import Process, Queue
import numpy as np

class FileIndex(object):
    """
    this class stores the unordered index of files
    the index is based on the comparison between the intervals on hypoterical natural numbers field, each intervals defined the
    subset of natural field (usually time) for each specific file is valid

    index consists of ts -- starts of the intervals since which file by i-th position in names is valid
    the index is a corresponding to the ts set of these indexes
    finally intervals -- is overally covered subset of the natural field.
    """

    def __init__(self, ts=None, intervals=None, index=None, names=None):
        self.ts = [] if ts is None else ts
        self.intervals = Intervals([]) if intervals is None else intervals
        self.index = np.empty(0, int) if index is None else index
        self.names = [] if names is None else names

    def add_file(self, name, intervals):
        if not name in self.names:
            snew = intervals & ~self.intervals
            if not snew.length == 0.:
                self.names.append(name)
                self.ts = np.concatenate([self.ts, snew.arr[:, 0]])
                idxs = np.argsort(self.ts)
                self.index = np.concatenate([self.index, np.full(snew.arr.shape[0], len(self.names) - 1, int)])
                self.index = self.index[idxs]
                self.ts = self.ts[idxs]
                self.intervals = self.intervals | intervals
                """
                sleft = self.intervals & ~snew
                idxold = list(np.array(self.index)[np.searchsorted(self.ts, sleft.arr[:, 0]) - 1])
                self.ts = self.ts + list(snew.arr[:, 0]) + list(sleft.arr[:, 0])
                self.index = self.index + [len(self.names),]*snew.arr.shape[0] + idxold
                isort = np.argsort(self.ts)
                self.ts = list(np.array(self.ts)[isort])
                self.index = list(np.array(self.index)[isort])
                self.intervals = self.intervals | intervals
                """

    def get_indexes(self, intervals):
        covered = intervals & self.intervals
        if covered.arr.size == 0:
            return np.empty(0, int)
        te, gaps = covered.make_tedges(self.ts)
        tc = (te[1:] + te[:-1])[gaps]/2.
        return self.index[np.unique(np.searchsorted(self.ts, tc)) - 1]

    def get_files(self, intervals):
        return [self.names[i] for i in self.get_indexes(intervals)]


def get_optimal_read_order(index, set_of_intervals):
    #index_set = {i: index.get_indexes(intervals) for i, intervals in enumerate(set_of_intervals)}
    index_set = [index.get_indexes(intervals) for intervals in set_of_intervals]
    tset = np.unique(np.concatenate(index_set))
    mask = np.ones(tset.size, bool)

    leftset = range(len(index_set))
    order = []
    for i in range(len(index_set)):
        idx = np.argmin([np.sum(mask[np.searchsorted(tset, index_set[k])]) for k in leftset])
        order.append(leftset[idx])
        mask[np.searchsorted(tset, index_set[leftset[idx]])] = False
        leftset.pop(idx)
    return order, index_set

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
                for sch in chull.get_all_child_intersect(ch.expand(pi/180.*28/60.)):  #26.5 arcmin includes vignetting as a hole
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

