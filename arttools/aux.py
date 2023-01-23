import numpy as np
from tqdm import tqdm
from scipy.interpolate import interp1d as i1d
from multiprocessing import Pool, Barrier, current_process
from threading import Thread
from itertools import repeat
import time

def numpy_array_naive_hash(arr):
    return arr.__array_interface__['data'] + arr.strides + (arr.size,) + (arr.dtype,)

def cache_single_result_np(function):
    lastcallsignature = None
    lastcallresult = None
    def newfunc(*args, **kwargs):
        nonlocal lastcallsignature
        nonlocal lastcallresult
        storeresult = kwargs.pop("storeresult", False)
        key = (numpy_array_naive_hash(val) if issubclass(val, np.ndarray) else val for val in args)
        if kwargs:
            for item in kwargs.items():
                key += item
        callsignature = hash(key)
        if lastcallresult == callsignature:
            result = lastcallresult
        else:
            result = function(*args, **kwargs)
            if storeresult:
                lastcallresult = result
                lastcallsignature = callsignature
        return result
    return newfunc


class interp1d(i1d):
    def integrate_in_intervals(self, xs, xe, mask=None):
        if mask is None:
            mask = np.ones(xs.size, bool)
        if self._kind in ["nearest", "nearest-up", "previous", "next"]:
            bl = self.x[0] > xs[0]
            br = self.x[-1] < xs[-1]
            size = self.x.size + br + bl + (self._kind in ["nearest", "nearest-up"])
            xx = np.empty(size, float)
            if bl:
                xx[0] = xs[0]
                if self._kind in ["nearest", "nearest-up"]:
                    xx[1] = self.x[0]
            else:
                xx[0] = self.x[0]

            if br:
                xx[-1] = xe[-1]
                if self._kind in ["nearest", "nearest-up"]:
                    xx[-2] = self.x[-1]
            else:
                xx[-1] = self.x[-1]

            if self._kind in ["nearest", "nearest-up"]:
                xx[int(bl) + 1:xx.size-int(br)-1] = self.x[:-1] + np.diff(self.x)/2.
            else:
                xx[int(bl):xx.size-int(br)] = self.x[:]


            sint = np.empty(xx.size, float)
            sint[0] = 0
            sint[1:] = np.cumsum(self((xx[1:] + xx[:-1])/2.)*np.diff(xx))
            return np.interp(xe[mask], xx, sint) - np.interp(xs[mask], xx, sint)
        else:
            return np.array([self._spline.integrate(s, e)[0] for s, e, m in zip(xs, xe, mask) if m])  # scipy.integrolate uses De Boor algorithm for spline, this aproach also have explicit integrateion (not vectorized unfortunately)


class DistributedObj(object):
    """
    this method is intended to be a superclass of any cluss, which has to repeat of specific method on the local copy of
    the same object

    to use the class one have to use it as a parent class, and setup mpnum and barrier arguments in the __init__ method
    also, in __init__ one have to add following intialization inside init:
        kwargs = local()
        kwargs.pop("self")
        kwargs.pop("__class__")
        super().__init__(**kwargs)
    """
    @classmethod
    def initizlie_local_obj(cls, kwargs):
        global localcopy
        localcopy = cls(**kwargs)


    def __init__(self, mpnum=0, barrier=None, **kwargs):
        if barrier is None and mpnum > 0:
            self.barrier = Barrier(mpnum)
            kwargs["barrier"] = self.barrier
            kwargs["mpnum"] = 0
            self._pool = Pool(mpnum, initializer=self.initizlie_local_obj, initargs=(kwargs,))
        else:
            self.barrier = barrier
            self._pool = None

    @staticmethod
    def perform_for_each_argument(vals):
        args, method, kwargs = vals
        global localcopy
        return localcopy.__getattribute__(method)(*args, **kwargs)

    @staticmethod
    def perform_staticmethod_for_each_argument(vals):
        args, method, kwargs = vals
        global localcopy
        return method(localcopy, *args, **kwargs) #localcopy.__getattribute__(method)(*args, **kwargs)

    @staticmethod
    def perform_for_each_proccess(vals):
        args, method, kwargs = vals
        localcopy.barrier.wait()
        return localcopy.__getattribute__(method)(*args, **kwargs)

    @staticmethod
    def perform_staticmethod_for_each_proccess(vals):
        args, method, kwargs = vals
        localcopy.barrier.wait()
        return method(localcopy, *args, **kwargs)

    @staticmethod
    def for_each_argument(method):
        def mpmethod(self, args, *largs, runmain=False, ordered=False, **kwargs):
            if self._pool is None or runmain:
                return method(self, args, *largs, **kwargs)
            else:
                if ordered:
                    return self._pool.imap(self.perform_for_each_argument, zip(args, repeat(method.__name__), repeat(kwargs)))
                else:
                    return self._pool.imap_unordered(self.perform_for_each_argument, zip(args, repeat(method.__name__), repeat(kwargs)))
        return mpmethod

    @staticmethod
    def for_each_process(method):
        def mpmethod(self, *args, apply_to_main=False, join=True, **kwargs):
            if self._pool is None:
                return method(self, *args, **kwargs)
            else:
                if apply_to_main:
                    method(self, *args, **kwargs)
                imap = self._pool.imap_unordered(self.perform_for_each_proccess, repeat((args, method.__name__, kwargs), self._pool._processes))
                return imap if not join else list(imap)
        return mpmethod

    def run_static_method(self, method, args, *largs, sync=True, join=True, **kwargs):
        if self._pool is None:
            return method(self, args, *largs, **kwargs)
        else:
            if sync:
                imap = self._pool.imap_unordered(self.perform_staticmethod_for_each_proccess, repeat((args, method, kwargs), self._pool._processes))
            else:
                imap = tqdm(self._pool.imap_unordered(self.perform_staticmethod_for_each_argument, zip(args, repeat(method), repeat(kwargs))))
            return imap if not join else list(imap)


def init_local_object(cls, barrier, args, kwargs):
    global localcopy
    localcopy = cls(*args, **kwargs)
    localcopy._barrier = barrier
    localcopy._pool = None

def perform_over_localcopy(args):
    args, method, kwargs = args
    global localcopy
    return localcopy.__getattribute__(method)(*args, **kwargs)

def perform_over_localcopy_barrier(args):
    args, method, kwargs = args
    global localcopy
    localcopy._barrier.wait()
    res = localcopy.__getattribute__(method)(*args, **kwargs)
    return res

def for_each_process(method):
    def mpmethod(self, *args, runmain=False, **kwargs):
        if runmain:
            method(self, *args, **kwargs)
        if self._barrier:
            self._barrier.reset()
            return self._pool.imap_unordered(perform_over_localcopy_barrier, repeat((args, method.__name__, kwargs), self._pool._processes))
        else:
            return method(self, *args, **kwargs)
    return mpmethod

def for_each_argument(method):
    def mpmethod(self, itargs, *args, **kwargs):
        if self._barrier:
            return self._pool.imap_unordered(perform_over_localcopy, zip(itargs, repeat(method.__name__), repeat(kwargs)))
        else:
            return method(self, itargs, *args, **kwargs)
    return mpmethod

class MPdistributed(type):
    def __new__(cls, clsname, superclasses, attributeddict):
        localcls = type.__new__(type, "localclass", superclasses, attributeddict) # superclasses, attributeddict)
        clsinit = attributeddict.get("__init__") #, lambda *args, **kwargs: pass)
        def newinit(*args, **kwargs):
            mpnum = kwargs.pop("mpnum")
            th = Thread(target = clsinit, args=args, kwargs=kwargs)
            th.start()
            if mpnum > 0:
                args[0]._barrier = Barrier(mpnum)
                args[0]._pool = Pool(mpnum, initializer=init_local_object, initargs=(localcls, args[0]._barrier, args[1:], kwargs))
            else:
                args[0]._barrier = False
            th.join()
        attributeddict["__init__"] = newinit
        for method in attributeddict["apply_for_each_process"]:
            attributeddict[method] = for_each_process(attributeddict[method])
        for method in attributeddict["apply_for_each_argument"]:
            attributeddict[method] = for_each_argument(attributeddict[method])
        return type.__new__(cls, clsname, superclasses, attributeddict)
