import numpy as np
from astropy.io import fits
from astropy.table import Table
from functools import reduce
from .filters import IndependentFilters, Intervals
from .time import get_gti, make_hv_gti

class Urddata(np.ndarray):
    """
    container for the urddata, which includes filter
    """
    def __new__(cls, inputarr, filters=IndependentFilters({}), urdn=None):
        if urdn != None:
            print("initialize as container")
            obj = np.asarray(inputarr).view(cls)
            obj.filters = filters
            obj.urdn = urdn
            return obj
        else:
            print("intialize as array")
            return np.asarray(inputarr).view(np.ndarray)

    @classmethod
    def read(cls, fitsfile):
        urdn = fitsfile["EVENTS"].header["URDN"]
        d = Table(fitsfile["EVENTS"].data).as_array()
        filters = IndependentFilters.from_fits(fitsfile)
        filters["TIME"] = get_gti(fitsfile, usehkgti=False, excludebki=True)
        return cls(d, filters, urdn)

    def apply_filters(self, filters):
        """
        apply new filter to data
        """
        cfilter = self.filters & filters
        return self.__class__(self[cfilter.apply(self)], cfilter, self.urdn)

    """
    def __add__(self, other):
        if self.urdn != other.urdn:
            raise ValueError("can't concatenate data from different urdns")
        cfilter = self.filter & other.filter
        if cfilter.volume > 0.:
            raise ValueError("intersecting sets of data" + " ".join([":".join(str(k), str(f)) for k, f in cfilter.items() if f.size > 0]))

        d = np.concatenate([self[cfilter.apply(self)], other[cfilter.apply(other)]])
        return self.__class__(d, cfilter, self.urdn)
    """

    @classmethod
    def concatenate(cls, urddlist):
        if np.unique([d.urdn for d in urddlist]).size != 1:
            raise ValueError("can't mix data from different URDNs, since they have different calibrations")
        cfilter = reduce(lambda a, b: a | b, [d.filters for d in urddlist])
        if cfilter.volume != sum([d.filters.volume for d in urddlist]):
            raise ValueError("inversecting data sets")

        d = np.concatenate([d for d in urddlist])
        return cls(d[np.argsort(d["TIME"])], cfilter, urddlist[0].urdn)

    def _to_hdulist(self):
        """
        provide method to produce hdulist here
        """
        pass

    def __array_finalize__(self, obj):
        if obj is None: return
        self = self.view(np.ndarray)

def read_urdfiles(urdflist, filterslist={}):
    urddata = {}
    urdhk = {}
    for urdfile in urdflist:
        ffile = fits.open(urdfile)
        udata = Urddata.read(ffile)
        urdhk[udata.urdn] = urdhk.get(udata.urdn, []) + [np.copy(ffile["HK"].data),]
        gti = Intervals([]) if udata.urdn not in urddata else reduce(lambda a, b: a | b, [f.filters["TIME"] for f in urddata[udata.urdn]])
        udata.apply_filters(IndependentFilters({"TIME": gti}))
        udata.apply_filters(filterslist.get(udata.urdn, IndependentFilters({})))
        urddata[udata.urdn] = urddata.get(udata.urdn, []) + [udata,]

    for urdn in urddata:
        urdhk[urdn] = np.unique(np.concatenate(urdhk[urdn]))
        hkgti = make_hv_gti(urdhk[urdn])
        hktimefilter = IndependentFilters({"TIME": hkgti})
        udata = Urddata.concatenate(urddata[urdn])
        udata.apply_filters(hktimefilter)
        urddata[urdn] = udata

    return urddata, urdhk



