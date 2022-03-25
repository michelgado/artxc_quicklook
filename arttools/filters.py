import numpy as np
from collections import OrderedDict
from astropy.io import fits
from astropy.table import Table
from .caldb import get_shadowmask_by_urd
from .telescope import URDNS
from .interval import Intervals


"""
filters is a set of classes which can be used for data filtering

as I see it there are three main types of filters:
    Intervals : checks weather the provided digits are inside 1d intervals
    Set & InversedSet: checks weather the provided numbers are in set or not in InversedSet
    and multidimensional indexed mask, which works the following way: each axis has function converting input column to mask index in corresponding axis

    each filter should have the same apply method, which consumes numpy array, and returns a boolean mask


    all of those can be accumulated to the IndependentFilters - a set of filters which can be applied one by one in arbitrary order
"""

class Intervals(object):
    """
    this class provides a number of userfull function to work with
    consequitive ordered unintersected 1d intervals
    most userfull functions of the class are
    *intersection interval1 & interval2
    *unification inverval1 | interval2
    and invertion ~interval
    """

    def _regularize(self, arr=None):
        """
        this function modifies input array in order to fulfill class conditions for the intervals set
        """
        if arr is None:
            arr = self.arr
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("an intervals array should be of shape Nx2")
        arr = arr[arr[:,0] < arr[:,1]]
        us, iidx = np.unique(arr[:, 0], return_inverse=True)
        arrn = np.empty((us.size, 2), np.double)
        arrn[:, 0] = us
        arrn[:, 1] = -np.inf
        np.maximum.at(arrn[:, 1], iidx, arr[:, 1])
        arr = arrn
        idx = np.argsort(np.ravel(arr))
        arr = np.ravel(arr)[idx]
        mask = np.zeros(arr.size, np.bool)
        mask[::2] = idx[::2] == np.arange(0, idx.size, 2)
        mask[1::2] = np.roll(mask[::2], -1)
        arr = np.asarray(arr[mask].reshape((-1, 2)))
        if arr.size and (arr[0, 0] == arr[0, 1] == -np.inf or arr[-1, 0] == arr[-1, 1] == np.inf):
            raise ValueError("invervals of type [inf, inf] are prohibited due to undifined width")
        return arr

    @classmethod
    def get_blank(cls):
        return cls((-np.inf, np.inf))

    @classmethod
    def get_empty(cls):
        return cls([])

    @classmethod
    def read(cls, fname):
        return cls(np.loadtxt(fname))

    def writeto(self, fname):
        np.savetxt(fname, self.arr)

    def __init__(self, arr):
        """
        read Nx2 array, which is assumed to be a set of 1D intervals
        make a regularization of these intervals - intervals with start > stop will be removed, intersected intervals joined and all intervals will be sorted ascendingly
        thus after regularization, the set of intervals consists of nonintersecting ordered 1d intervals

        examples of usage:
            cretion of Intervals::
                i1 = Intervals(np.array(Nx2))

            intersections of two set of intervals:
                i3 = i1 & i2 ([[0, 2]] & [[1, 10]] == [[1, 2]])

            union of intervals
                i3 = i1 | i2 ([[0, 2]] | [[1, 10]] == [[0, 10]])

            invert gti:
                i3 = -i1 (-[[0; 1]] == [[-inf; 0], [1, inf]] e.t.c.)

            difference of two intervals can be obtained combining intersection and negation
                i3 = i1 & ~i2

            for the sake of usability some additional simple operations were added:
            all intervals can be shifted with + or -
            inew = i + floatval ([[0, 10]] + 5 == [[5, 15]] e.t.c.)

            same operations can be used to change width simultaneously for all intervals in the set:
            inew = i + [floatval, floatval (([0, 10]] + [0, -1] == [[0, 9]] e.t.c.)

        """
        if issubclass(self.__class__, type(arr)):
            self.arr = np.copy(arr.arr)
        else:
            self.arr = self._regularize(np.asarray(arr).reshape((-1, 2)))

    @property
    def shape(self):
        return self.arr.shape

    @property
    def size(self):
        return self.arr.size


    @property
    def length(self):
        return np.sum(self.arr[:, 1] - self.arr[:, 0])

    def __repr__(self):
        return self.arr.__repr__()

    def __and__(self, other):
        """
        this operation must left us with intersections of the two sets of 1d intervals
        """
        if self.size == 0 or other.size == 0:
            return self.__class__([[]])
        #both self.arr and other.arr already regularized
        tt = np.concatenate([self.arr.ravel(), other.arr.ravel()])
        #in this set of edges all even are starts and odd and ends, but not ordered
        ms = np.ones(tt.size, np.int8)
        #mark all starts with 1 and stops with -1
        ms[1::2] = -1
        #sort all edges but store their position
        idx = np.argsort(tt)
        tt = tt[idx]
        #rearrange arr in form of (N - 1 x 2), which look like [...,[x_i, x_i+1], [x_i+1, x_i+2],...]
        arr = np.lib.stride_tricks.as_strided(tt, (tt.size - 1, 2), tt.strides*2)
        #make empty intervals instance,
        #all condintion on intervals are already fullfield, no regularization required, so pass __init__
        #gres = self.__class__(arr[np.cumsum(ms[idx][:-1]) == 2])
        gres = self.__class__.__new__(self.__class__)
        # we would like to left intervals, bounded by edges, located inside the other intervals set
        gres.arr = np.copy(arr[np.cumsum(ms[idx][:-1]) == 2])
        gres.arr = gres.arr[gres.arr[:, 0] < gres.arr[:, 1]]
        return gres

    def merge_joint(self):
        """
        merge intervals which has stop_left == start_right
        such intervals are technicaly permitted,
        it should be notted however, that due to machine precision, such intervals can
        be joined in _regularize subroutine (during comparison
        two different (stored in different memory cells) but equal float values can randomly
        be greater or lesser to each other
        """
        mask = np.ones(self.arr.shape[0] + 1, np.bool)
        mask[1:-1] = self.arr[1:, 0] != self.arr[:-1, 1]
        mask = np.lib.stride_tricks.as_strided(mask, (mask.size - 1, 2), mask.strides*2)
        self.arr = self.arr[mask].reshape((-1, 2))

    def __invert__(self):
        if self.size == 0:
            return self.__class__([-np.inf, np.inf])
        arr = np.empty((self.shape[0] + 1, 2), np.double)
        arr[1:,0] = self[:, 1]
        arr[:-1, 1] = self[:, 0]
        arr[[0, -1], [0, 1]] = [-np.inf, np.inf]
        # we allow user to create intervals with 0 width (start == stop),
        # therefore regularization does not affect such intervals in set
        # but there is a special case where we can get left interval [-inf, -inf] or [inf, inf]
        # the width of such intervals is undifined, therefore we avoid them in any operations
        if self.arr[0, 0] == -np.inf:
            arr = arr[1:]
        if self.arr[-1, 1] == np.inf:
            arr = arr[:-1]
        return self.__class__(arr)

    def __getitem__(self, *args):
        return self.arr.__getitem__(*args)

    def __or__(self, other):
        #note, since during each init set of intervals are regularized,
        #no additional actions are required to get union
        return self.__class__(np.concatenate([self.arr, other.arr]))

    def __iadd__(self, val):
        if type(val) is float is np.inf:
            raise ValueError("adding inf to the set of intervals will lead to the incorrect behavior")
        self.arr = self._regularize(self.arr + val)
        return self

    def __add__(self, val):
        if type(val) is float is np.inf:
            raise ValueError("adding inf to the set of intervals will lead to the incorrect behavior")
        return self.__class__(self.arr + val)

    def __sub__(self, val):
        if type(val) is float is np.inf:
            raise ValueError("adding inf to the set of intervals will lead to the incorrect behavior")
        return self.__class__(self.arr - val)

    def __isub__(self, val):
        if type(val) is float and abs(val) is np.inf:
            raise ValueError("adding inf to the set of intervals will lead to the incorrect behavior")
        self.arr = self._regularize(self.arr - val)
        return self

    def __idiv__(self, val):
        if type(val) is float  is np.inf:
            raise ValueError("division on inf to the set of intervals will lead to the incorrect behavior")
        self.arr = self._regularize(self.arr/val)

    def __imul__(self, val):
        if type(val) is float and abs(val) is np.inf:
            raise ValueError("multiplication on inf to the set of intervals will lead to the incorrect behavior")
        self.arr = self._regularize(self.arr*val)

    def __mul__(self, val):
        if type(val) is float is np.inf:
            raise ValueError("multiplication on inf to the set of intervals will lead to the incorrect behavior")
        return self.__class__(self.arr*val)

    def __div__(self, val):
        if type(val) is float is np.inf:
            raise ValueError("division on inf to the set of intervals will lead to the incorrect behavior")
        return self.__class__(self.arr/val)

    def __eq__(self, other):
        return (self.length == other.length) & (self.length == (self & other).length)

    def merge_close_intervals(self, dt):
        """
        merge GTI intervals which separated by less then dt
        """
        self.arr = self._regularize(self.arr + [-dt/2., dt/2.]) - [-dt/2., dt/2.]

    def remove_short_intervals(self, dt):
        """
        remove intervals shorter then dt
        """
        self.arr = self._regularize(self.arr + [dt/2., -dt/2.]) - [dt/2., + dt/2.]

    def mask_external(self, ts, open_bounds=False):
        """
        creates bitwise mask, which marks points out of intervals, sorted in the set
        example intervals i1 = [[0, 2], [4, 6]]
        ts = [1, 3, 5, 7]
        i1.mask_external(ts) = [true, false, true, false]
        """
        idxs = np.argsort(ts)
        idx = self.arr.ravel().searchsorted(ts[idxs]) #%2 == 1
        mask = idx%2 == 1
        if open_bounds:
            idxb = np.where(mask[1:] & ~mask[:-1])[0]
            mask[idxb] = self.arr.ravel()[idx[idxb]] == ts[idxb]
        res = np.empty(mask.size, np.bool)
        res[idxs] = mask[:]
        return res

    def apply(self, ts):
        return self.mask_external(ts, True)

    def __str__(self):
        return " ".join(["[%.1f, %.1f]" % tuple(a) for a in self.arr])

    @property
    def length(self):
        return np.sum(self.arr[:,1] - self.arr[:,0])

    def make_tedges(self, ts, joinsize=0):
        """
        assuming that ts is a series of ascending evenly spaced points,
        produce new series of points, lying in the GTIs (with additional points at the edges of
        intervals if requeired), and mask, showing position of the gaps between points clusted from
        different intervals
        """
        dtmed = np.median(np.diff(ts, 1))
        gtloc = self & self.__class__(ts[[0, -1]])
        if joinsize > 0:
            gtloc.remove_short_intervals(joinsize*dtmed)
        if gtloc.length == 0:
            return np.array([]), np.array([])
        ts = ts[gtloc.mask_external(ts)]
        newts = np.unique(np.concatenate([ts, gtloc.arr.ravel()]))
        idxgaps = newts.searchsorted((gtloc.arr[:-1, 1] + gtloc.arr[1:, 0])/2.)
        maskgaps = np.ones(newts.size - 1 if newts.size else 0, np.bool)
        maskgaps[idxgaps - 1] = False
        #print(ts[:3] - ts[0], ts[-3:] - ts[0])
        #print(newts[:3] - ts[0], ts[-3:] - ts[0])
        #print(newts.size)
        #print(idxgaps[:3], idxgaps[-3:])
        #print(gtloc.arr - ts[0])
        #===============================================
        #join time intervals at the edges of the gti, if they are two short
        dt = np.diff(newts, 1)
        dtmed = np.median(dt)
        maskshort = np.ones(newts.size, np.bool)
        maskshort[idxgaps + 1] = dt[idxgaps] > dtmed*joinsize
        maskshort[idxgaps - 2] = dt[idxgaps - 2] > dtmed*joinsize
        return newts[maskshort], maskgaps[maskshort[:-1]]

    def arange(self, dt, joinsize=0.2, t0=None):
        """
        returns a set of digits, wichi are separated with dt within intervals and also including edges of the intervals
        signature:
            dt - width of the time step
            joinsize - minimum allowed size of step in fraction of dt, if step is shorter it is joined with the joint one
            t0 - starting epoch

        returns a set of steps within intervals and mask, which marks which intervals within digits are within the initial intervals
        """
        if t0 is None:
            t0 = np.median(self.arr[:, 0]%dt) + (self.arr[0, 0]//dt - 1)*dt
        te = np.unique(np.concatenate([np.arange((s - t0)//dt + 1, (e - t0)//dt + 1)*dt + t0 for s, e in self.arr]))
        eidx = np.searchsorted(te, self.arr)
        mempty = eidx[:, 0] != eidx[:, 1]
        sidx = np.searchsorted(te, self.arr[mempty, 0])
        m1 = np.ones(te.size, np.bool)
        m1[sidx] = te[sidx] - self.arr[mempty, 0] > dt*joinsize
        te = te[m1]

        eidx = np.searchsorted(te, self.arr)
        mempty = eidx[:, 0] != eidx[:, 1]
        sidx = np.searchsorted(te, self.arr[mempty, 1]) - 1
        m1 = np.ones(te.size, np.bool)
        m1[sidx] = self.arr[mempty, 1] - te[sidx] > dt*joinsize
        te = te[m1]

        te = np.unique(np.concatenate([self.arr.ravel(), te]))
        mgaps = self.mask_external((te[1:] + te[:-1])/2.)

        return te, mgaps


#======================================================================================================================


class RationalSet(set):
    def apply(self, vals):
        return np.isin(vals, list(self))

    def __and__(self, other):
        if type(other) == RationalSet:
            return RationalSet(self.intersection(other))
        if type(other) == InversedRationalSet:
            return RationalSet(self.difference(other))

    def __or__(self, other):
        if type(other) == RationalSet:
            return self.__class__(self.union(other))
        if type(other) == InversedRationalSet:
            return InversedRationalSet(other.difference(self))

    @property
    def size(self):
        return len(self)

    def __invert__(self):
        return InversedRationalSet(self)

    @classmethod
    def get_blank(cls):
        return ~cls([])

    @classmethod
    def get_empty(cls):
        return cls([])


class InversedRationalSet(set):
    def apply(self, vals):
        return np.isin(vals, list(self), invert=True)

    def __and__(self, other):
        if type(other) == InversedRationalSet:
            return InversedRationalSet(self.intersection(other))
        if type(other) == RationalSet:
            return other & self

    def __or__(self, other):
        if type(other) == InversedRationalSet:
            return InversedRationalSet(self & other)
        if type(other) == RationalSet:
            return other & self

    def __invert__(self):
        return RationalSet(self)

    def __init__(self, *args, vol=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vol = set(vol) if not vol is None else set(self) #define weather a inversed set defined in the finite sample

    @classmethod
    def get_blank(cls):
        return cls([])

    @classmethod
    def get_empty(cls):
        return ~cls([])

    @property
    def size(self):
        return len(self.vol) - len(self) #its hard to define size for inversed rational set, by default it can be considered inf


#======================================================================================================================


def strightindex(x):
    """
    straitindex function prodice int from the correspondding float, considering, that input [i, i + [0, 1]] value corresponds to i-th index
    """
    return x.astype(int)


class InversedIndexMask(OrderedDict):
    def __init__(self, indexfunctions, mask):
        """
        indexfunction a zip of ordered according to mask axis order set of indexing functions
        attention!!!
        provided mask should not be a subset of greater numpy array or array generated by
        numpy strides tricks (broadcasted array) because its implicitly assumed that strides correspond to the shape
        """
        super().__init__(indexfunctions)
        self.mask = mask
        if self.mask.ndim == 1:
            self.shifts = [1,]
        else:
            self.shifts = np.concatenate([np.cumprod(self.mask.shape[-2::-1])[::-1], [1,]])

    def apply(self, fields):
        if not all([k in fields.dtype.names for k in self]):
            raise ValueError("not all of the fields are present in query")
        return self.mask.ravel()[sum(s*self[name](fields[name]) for s, name in zip(self.shifts, self))]

    @property
    def size(self):
        return self.mask.sum()

    def __and__(self, other):
        if type(other) == type(self):
            if self.mask.shape != other.mask.shape:
                raise ValueError("incompatible masks")
            if not all([key in other for key in self]) or not all([key in self for key in other]):
                raise ValueError("imcompatible fields")
            swaporder = [list(other.keys()).index(k) for k in self]
            mc = np.copy(other.mask)
            for i in range(len(swaporder)):
                if swaporder[i] == i:
                    continue
                mc = np.copy(mc.swapaxes(i, swaporder[i]))
                swaporder[swaporder[i]] = swaporder[i]
            return InversedIndexMask(self.items(), np.logical_and(self.mask, mc))


    def __or__(self, other):
        print("ok")
        if type(other) == type(self):
            print("class ok")
            if self.mask.shape != other.mask.shape:
                raise ValueError("incompatible masks")
            if not all([key in other for key in self]) or not all([key in self for key in other]):
                raise ValueError("imcompatible fields")
            swaporder = [list(other.keys()).index(k) for k in self]
            mc = np.copy(other.mask)
            print(swaporder)
            for i in range(len(swaporder)):
                if swaporder[i] == i:
                    continue
                mc = np.copy(mc.swapaxes(i, swaporder[i]))
                swaporder[swaporder[i]] = swaporder[i]
            return InversedIndexMask(self.items(), np.logical_or(self.mask, mc))

    def __eq__(self, other):
        return np.all(self.mask == other.mask)

    def __invert__(self):
        return self.__class__(self, ~self.mask)

    def meshgrid(self, arrays):
        keys = list(self.keys())[::-1]
        ud = np.meshgrid(*[self[name](arrays[name]) for name in keys])
        shape = ud[0].shape
        data = np.column_stack([a.ravel() for a in ud[::-1]]).ravel().view(
                                    [(name, np.int) for name in keys[::-1]]).reshape(shape)

        return self.apply(data)

    def _to_hdu(self):
        h = {"FIELDS": " ".join(self)}
        fits.ImageHDU(data=mask.astype(np.unit8), header=h)

INDEXMAPS = {"INTMAP": strightindex}

#======================================================================================================================

FILTERSOBJS = {"RSET": RationalSet, "IRSET": InversedRationalSet, "INTERVAL": Intervals, "FMAP": InversedIndexMask}

class IndependentFilters(dict):
    """
    a set of independent fiilters which can be consequetly and in random order to the data with the same output result

    simple rule, which has to be applied to the filters, name of the filter field should correspond to the filtered columns,
    i.e. RationalSet filtering GRADE column should be provided as {"GRADE": correspondingRationalSet},

    key of the multidimensional filter (InversedIndexMask) should be a tuple, containing filtered columns in proper order, for example,
    the sahadow mask for RAW_X, RAW_Y filter columns should be initialized as {("RAW_X", "RAW_Y"): correspondingInversedIndexMask}
    """
    def __and__(self, other):
        """
        return the striktest possible filter, containing conditions from both independent filters set
        """
        allfields = set(list(self.keys()) + list(other.keys()))
        return IndependentFilters({k: self[k] & other[k] if (k in self and k in other) else self.get(k, other[k]) for k in allfields})

    def __or__(self, other):
        commonfields = set(self.keys()).intersection(set(other.keys()))
        return IndependentFilters({k: self[k] | other[k]  for k in commonfields})

    @property
    def filter(self):
        """
        the idea behind this method is to use urddata containers instead of the separate filters
        this approach protect against error, arising due to inconsistent data selection and produced products
        """
        return self

    @property
    def volume(self):
        return np.prod([f.size for f in self.values()])

    def apply(self, valsset):
        mask = np.ones(valsset.size, np.bool)
        for k, f in self.items():
            if type(f) == InversedIndexMask:
                mask[mask] = f.apply(valsset[mask])
                continue
            if k not in valsset.dtype.names:
                continue
            mask[mask] = f.apply(valsset[mask][k])

        return mask #np.all([f.apply(valsset if type(f) == InversedIndexMask else valsset[k]) for k, f in self.items() if k in valsset.dtype.names], axis=0)

    def meshgrid(self, keys, arrays):
        """
        produces a multidimensionall mask for a list of keys and corresponding grids(arrays), which corresponds to the IndependetFilters

        signature:
            keys( names of columns, i.e. TIME, RAW_X... etc.),
            arrays - corresponding to key array of possible values
        return:
            boolean mask of shape [a.size for a in arrays]

        Q: why one should provide array?
        A: all of the filters doesn't define the specific volume of the permissible variables, therefore they should be defined externaly

        Q: where I could need this method,
        A: this method is used to produce mock data, one could easiliy estimate the probability volume of the specific value with such mask
        """
        ud = np.meshgrid(*[a for a in arrays])
        shape = ud[0].shape
        data = np.recarray(ud[0].size, [(k, a.dtype) for k, a in zip(keys, arrays)])
        """
        data = np.column_stack([a.ravel() for a in ud[::-1]]).ravel().view(
                                    [(name, np.int) for name in keys[::-1]])
        """
        for i in range(len(keys)):
            data[keys[i]][:] = ud[i].ravel()
        return self.apply(data).reshape(shape)


    def __eq__(self, other):
        if len(self) != len(other):
            return False
        if set(self.keys()) != set(other.keys()):
            return False
        return np.all([self[k] == other[k] for k in self])

    @classmethod
    def from_fits(cls, ffile):
        """
        read fits file and extract IndependentFilter
        TODO: decide and implement on the format to write and read filters in fits compatible format
        """
        return cls({})


#======================================================================================================================

def get_shadowmask_filter(urdn):
    """
    specific method with constructs InversedIndexMask based on the stored in the caldb shadowmask
    """
    shmask = get_shadowmask_by_urd(urdn)
    if urdn == 22:
        shmask[[45, 46], :] = False
    return InversedIndexMask(zip(["RAW_X", "RAW_Y"], [strightindex, strightindex]), shmask)


DEFAULTFILTERS = {urdn: IndependentFilters({"ENERGY": Intervals([4., 12.]),
                                            "GRADE": RationalSet(range(10)),
                                            ("RAW_X", "RAW_Y"): get_shadowmask_filter(urdn)}) for urdn in URDNS}

DEFAULTBKGFILTER = IndependentFilters({"ENERGY": Intervals([40., 100.]), "RAW_X": InversedRationalSet([0, 47]), "RAW_Y":InversedRationalSet([0, 47])})
