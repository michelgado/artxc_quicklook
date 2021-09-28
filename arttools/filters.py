import numpy as np
from collections import OrderedDict
from .caldb import get_shadowmask_by_urd
from .telescope import URDNS
from .interval import Intervals

"""
filters is a set of classes which can be used for data filtering

as I see it there are three main types of filters:
    Intervals : checks weather the provided numbers are inside intervals
    Set: checks weather the provided numbers are in set
    and multidimensional indexed mask
"""


def strightindex(x):
    return x.astype(np.int)


class RationalSet(set):
    def apply(self, vals):
        return np.isin(vals, list(self))

class InversedRationalSet(set):
    def apply(self, vals):
        return np.isin(vals, list(self), invert=True)

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

    def __and__(self, other):
        if type(other) == type(self):
            if self.mask.shape != other.mask.shape:
                raise ValueError("incompatible masks")
            if not all([key in other for key in self]) or not all([ket in self for key in other]):
                raise ValueError("imcompatible fields")
            return InversedIndexMask(self.items(), np.logical_and(mask))
        if type(other) == RationalMultiSet:
            mask = np.copy(self.mask)

    def __or__(self, other):
        if self.mask.shape != other.mask.shape:
            raise ValueError("incompatible masks")
        if not all([key in other for key in self]) or not all([ket in self for key in other]):
            raise ValueError("imcompatible fields")
        return InversedIndexMask(self.items(), np.logical_or(mask))

    def meshgrid(self, arrays):
        keys = list(self.keys())[::-1]
        ud = np.meshgrid(*[self[name](arrays[name]) for name in keys])
        shape = ud[0].shape
        data = np.column_stack([a.ravel() for a in ud[::-1]]).ravel().view(
                                    [(name, np.int) for name in keys[::-1]]).reshape(shape)

        return self.apply(data)

class IndependentFilters(dict):
    """
    multidimensional named interval
    """
    def __and__(self, other):
        allfieds = set(list(self.keys()) + list(other.keys()))
        return MultiInterval({k: self.get(k, other[k]) & other.get(k, self[k]) for k in allfields})

    def __or__(self, other):
        allfieds = set(list(self.keys()) + list(other.keys()))
        return MultiInterval({k: self.get(k, other[k]) | other.get(k, self[k]) for k in allfields})

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
        ud = np.meshgrid(*[a for a in arrays])
        shape = ud[0].shape
        print(shape)
        data = np.recarray(ud[0].size, [(k, a.dtype) for k, a in zip(keys, arrays)])
        """
        data = np.column_stack([a.ravel() for a in ud[::-1]]).ravel().view(
                                    [(name, np.int) for name in keys[::-1]])
        """
        for i in range(len(keys)):
            data[keys[i]][:] = ud[i].ravel()
        return self.apply(data).reshape(shape)


def get_shadowmask_filter(urdn):
    shmask = get_shadowmask_by_urd(urdn)
    if urdn == 22:
        shmask[[45, 46], :] = False
    return InversedIndexMask(zip(["RAW_X", "RAW_Y"], [strightindex, strightindex]), shmask)

DEFAULTFILTERS = {urdn: IndependentFilters({"ENERGY": Intervals([4., 12.]),
                                            "GRADE": RationalSet(range(10)),
                                            "RAWXY": get_shadowmask_filter(urdn)}) for urdn in URDNS}
