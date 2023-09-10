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
    IORDER = {"nearest": 0, "nearest-up": 0, "previous": 0, "next":0, "slinear": 0.9, "linear": 1, "quadratic": 2, "cubic": 3}
    def integrate_in_intervals(self, ii):
        xs, xe = ii.T
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
            return np.interp(xe, xx, sint) - np.interp(xs, xx, sint)

        if self._kind=="linear":
            k0 = np.empty(self.y.size + 1, float)
            k0[1:-1] = self.y[:-1]
            k0[[0, -1]] = tuple(self.fill_value)[0], tuple(self.fill_value)[-1]
            k1 = np.empty(self.y.size + 1, float)
            k1[1:-1] = np.diff(self.y)/np.diff(self.x)/2.
            k1[[0, -1]] = 0.
            sint = np.empty(self.y.size + 1, float)
            sint[[0, -1]] = 0
            sint[1:-1] = (self.y[1:] + self.y[:-1])*np.diff(self.x)/2.
            sint = np.cumsum(sint)
            ss = np.searchsorted(self.x, xs)
            se = np.searchsorted(self.x, xe)
            fadd = np.zeros(ss.size, float)
            m = ss == se
            fadd[m] = (self(xs[m]) + k1[ss[m]]*(xe - xs)[m])*(xe - xs)[m]
            m = ss != se
            fadd[m] = (self(xs[m]) + k1[ss[m]]*(self.x[ss[m]] - xs[m]))*(self.x[ss[m]] - xs[m]) + (k0[se[m] - 1] + k1[se[m]]*(xe[m] - self.x[se[m] - 1]))*(xe[m] - self.x[se[m] - 1]) + sint[se[m] - 1] - sint[ss[m]]
            return fadd
        else:
            if self._spline.k == 1:
                return self.__class__(self.x, self.y, bounds_error=self.bounds_error, fill_value=self.fill_value, kind="linear").integrate_in_intervals(ii)
            else:
                return np.array([self._spline.integrate(s, e) for s, e in ii])  # scipy.integrolate uses De Boor algorithm for spline, this aproach also have explicit integrateion (not vectorized unfortunately)

    @property
    def kind(self):
        if self._kind in ["linear", "nearest", "nearest-up", "previous", "next"]:
            return self._kind
        return [None, "slinear", "quadratic", "cubic"][self._spline.k]

    def __add__(self, other):
        if self.kind != other.kind:
            raise ValueError("can add only interpolations of the same kind")
        xx = np.unique(np.concatenate([self.x, other.x]))
        yy = self(xx) + other(xx)

        bounds_error=self.bounds_error
        if not self.fill_value is None and not other.fill_value is None:
            fill_value = tuple(np.asarray(self.fill_value)+np.asarray(other.fill_value))
        else:
            fill_value = np.nan
        return self.__class__(xx, yy, kind=self.kind, bounds_error=bounds_error, fill_value=tuple(fill_value))

    def __mul__(self, other):
        kind = self._kind if self.IORDER[self._kind] > self.IORDER[other._kind] else other._kind
        xx = np.unique(np.concatenate([self.x, other.x]))
        yy = self(xx)*other(xx)
        print(yy.min(), yy.max(), yy.size, self(xx).max(), self(xx).min(), other(xx).max(), other(xx).min())
        bounds_error=self.bounds_error
        if not self.fill_value is None and not other.fill_value is None:
            fill_value = tuple(np.asarray(self.fill_value)*np.asarray(other.fill_value))
        else:
            fill_value = np.nan
        return self.__class__(xx, yy, kind=kind, bounds_error=bounds_error, fill_value=tuple(fill_value))

    def __and__(self, other):
        if self.kind != other.kind:
            raise ValueError("can join only interpolations of the same kind")
        if (other.x[0] > self.x[0] and other.x[0] < self.x[1]) or (other.x[1] > self.x[0] and other.x[0] < self.x[1]) or \
                (self.x[0] > other.x[0] and self.x[0] < other.x[1]) or (self.x[1] > other.x[0] and self.x[0] < other.x[1]):
            raise ValueError("joined interpolation should not intersect")

        left, right = (self, other) if self.x[-1] < other.x[0] else (other, self)
        fill_value = (left.fill_value[0], right.fill_value[-1])
        return interp1d(np.concatenate([left.x, right.x]), np.concatenate([left.y, right.y]), kind=self.kind, bounds_error=self.bounds_error & other.bounds_error, fill_value=fill_value)


    def _scale(self, scale):
        return self.__class__(self.x, self.y*scale, kind=self.kind, bounds_error=self.bounds_error, fill_value=tuple(np.asarray(self.fill_value)*scale))


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
    def initizlie_local_obj(cls, barrier, kwargs):
        global localcopy
        localcopy = cls.__new__(cls)
        localcopy.mpnum = 0
        localcopy.barrier = barrier
        localcopy._pool = None
        print(kwargs)
        try:
            localcopy.__init__(**kwargs)
        except:
            pass


    def __init__(self, mpnum=0, barrier=None, **kwargs):
        if barrier is None and mpnum > 0:
            self.barrier = Barrier(mpnum)
            self._pool = Pool(mpnum, initializer=self.initizlie_local_obj, initargs=(self.barrier, kwargs,))
        else:
            self.barrier = barrier
            self._pool = None #ThreadPool(threads_per_lcopy)
        #super().__init__(**kwargs)

    @staticmethod
    def perform_for_each_argument(vals):
        args, method, kwargs = vals
        global localcopy
        #print(args, method, kwargs)
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

    def __del__(self,):
        if not self._pool is None:
            self._pool.close()


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
