import numpy as np
from astropy.io import fits
from astropy.table import Table
from functools import reduce
from .filters import IndependentFilters, Intervals
from .time import get_gti, make_hv_gti

class Urddata(object):
    def __init__(self, data, urdn, filters):
        self.data = data
        self.urdn = urdn
        self.filters = filters

    def __getitem__(self, args):
        return self.data[args]

    @classmethod
    def read(cls, fitsfile, excludebki=True):
        urdn = fitsfile["EVENTS"].header["URDN"]
        d = np.unique(Table(fitsfile["EVENTS"].data).as_array())
        filters = IndependentFilters.from_fits(fitsfile)
        filters["TIME"] = get_gti(fitsfile, usehkgti=True, excludebki=excludebki)
        return cls(d, urdn, filters)

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def size(self):
        return self.data.size

    def apply_filters(self, filters):
        """
        apply new filter to data
        """
        cfilter = self.filters & filters
        return self.__class__(np.copy(self.data[cfilter.apply(self)]), self.urdn, cfilter)


    @classmethod
    def concatenate(cls, urddlist):
        if np.unique([d.urdn for d in urddlist]).size != 1:
            raise ValueError("can't mix data from different URDNs, since they have different calibrations")
        #print('filters', [d.filters for d in urddlist])
        print("exposure", sum([d.filters["TIME"].length for d in urddlist]))
        print("crossed exposure", reduce(lambda a, b: a | b, [d.filters["TIME"] for d in urddlist]).length)
        cfilter = reduce(lambda a, b: a | b, [d.filters for d in urddlist])
        """
        import pickle
        pickle.dump([cfilter, [d.filters for d in urddlist]], open("/srg/a1/work/andrey/ART-XC/lp20/filttest.pkl","wb"))
        print("tot filter", cfilter)
        print("volume", [d.filters.volume for d in urddlist], sum([d.filters.volume for d in urddlist]), cfilter.volume - sum([d.filters.volume for d in urddlist]))
        """

        if abs(cfilter.volume - sum([d.filters.volume for d in urddlist])) > 1e-1:
            raise ValueError("intersecting data sets")

        d = np.concatenate([d.data for d in urddlist])
        return cls(d[np.argsort(d["TIME"])], urddlist[0].urdn, cfilter)

    def _to_hdulist(self):
        """
        provide method to produce hdulist here
        """
        pass

def read_urdfiles(urdflist, filterslist={}):
    urddata = {}
    urdhk = {}
    for urdfile in urdflist:
        ffile = fits.open(urdfile)
        udata = Urddata.read(ffile)
        urdhk[udata.urdn] = urdhk.get(udata.urdn, []) + [np.copy(ffile["HK"].data),]
        gti = Intervals([]) if udata.urdn not in urddata else reduce(lambda a, b: a | b, [f.filters["TIME"] for f in urddata[udata.urdn]])
        udata = udata.apply_filters(IndependentFilters({"TIME": ~gti}))
        #print("filter", udata.filters["TIME"])
        udata = udata.apply_filters(filterslist.get(udata.urdn, IndependentFilters({})))
        #print("filter", udata.filters["TIME"])
        urddata[udata.urdn] = urddata.get(udata.urdn, []) + [udata,]

    for urdn in urddata:
        urdhk[urdn] = np.unique(np.concatenate(urdhk[urdn]))
        hkgti = make_hv_gti(urdhk[urdn])
        hktimefilter = IndependentFilters({"TIME": hkgti})
        udata = Urddata.concatenate(urddata[urdn])
        udata.apply_filters(hktimefilter)
        urddata[urdn] = udata

    return urddata, urdhk
