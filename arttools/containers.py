import numpy as np
from astropy.io import fits
from astropy.table import Table
from functools import reduce
from .filters import IndependentFilters

class Urddata(np.ndarray):
    def __new__(cls, inputarr, filter, urdn):
        obj = np.asarray(inputarr).view(cls)
        obj.filter = filter
        obj.urdn = urdn
        return obj

    @classmethod
    def read(cls, fitsfile):
        urdn = fitsfile["EVENTS"].header["URDN"]
        d = Table(fitsfile["EVENTS"].data).as_array()
        filters = IndependentFilters.from_fits(fitsfile)
        return cls(d, filters, urdn)

    def filter(self, filter):
        cfilter = self.filter & filter
        return self.__class__(self[cfilter.apply(self)], cfilter, urdn)

    def __add__(self, other):
        if self.urdn != other.urdn:
            raise ValueError("can't concatenate data from different urdns")
        cfilter = self.filter & other.filter
        if cfilter.volume > 0.:
            raise ValueError("intersecting sets of data" + " ".join([":".join(str(k), str(f)) for k, f in cfilter.items() if f.size > 0]))

        d = np.concatenate([self[cfilter.apply(self)], other[cfilter.apply(other)]])
        return self.__class__(d, cfilter, self.urdn)

    @classmethod
    def concatenate(cls, urddlist):
        if np.unique([d.urdn for d in urddlist]).size == 1:
            raise ValueError("can't mix data from different URDNs, since they have different calibrations")
        cfilter = reduce(lambda a, b: a | b, [d.filter for f in urddlist])
        if cfilter.volumne != sum([d.filter.volume for d in urddlist]):
            raise ValueError("inversecting data sets")

        d = np.concatenate([d for d in urddlist])
        return cls(d, cfilter, urdlist[0].urdn)

    def _to_hdulist(self):
        """
        provide method to produce hdulist here
        """
        pass


