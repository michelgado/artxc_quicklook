import numpy as np

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



