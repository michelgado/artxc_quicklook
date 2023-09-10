from .caldb import get_totevt_during_bkg, get_deadtime_for_dev
from .background import get_background_surface_brigtnress
from .filters import DEFAULTBKGFILTER, IndependentFilters, Intervals, RationalSet
from .aux import interp1d
from .time import GTI
from functools import reduce
import numpy as np

def get_src_frac(urddata, filters):
    """
    urddata shoulb be unfilterred
    the procedure finds excess over stored background spectrum and attribute it to the source, after that a fraction of excess signal contained within filters is found
    """
    dfilt = {urdn: d.filters & IndependentFilters({"GRADE": RationalSet(range(16)), "ENERGY": Intervals([0., np.inf])}) for urdn, d in urddata.items()}
    bkgdata = {urdn: DEFAULTBKGFILTER.apply(d).sum() for urdn, d in urddata.items()}
    print("bkgdata", bkgdata)
    totdata = {urdn: d.size for urdn, d in urddata.items()}
    print("totdata", totdata)
    fildata = {urdn: filters[urdn].apply(d).sum() for urdn, d in urddata.items()}
    print("filtdata", fildata)
    tbrate = {urdn: get_background_surface_brigtnress(urdn, dfilt[urdn], fill_value=0.).sum() for urdn in urddata}
    r1 = {urdn: tbrate[urdn]/get_background_surface_brigtnress(urdn, DEFAULTBKGFILTER, fill_value=0.).sum() for urdn in urddata}
    r2 = {urdn: tbrate[urdn]/get_background_surface_brigtnress(urdn, filters[urdn], fill_value=0.).sum() for urdn in urddata}
    fracs = {urdn: (fildata[urdn] - bkgdata[urdn]/r2[urdn])/(totdata[urdn] - bkgdata[urdn]/r1[urdn]) for urdn in urddata}
    return fracs


def bright_source_deadtime(srcevt, bkgevt, dt=1., dtbkg=1000., srcfrac={}, deadtime={}):
    """
    typycally we would control very slow variations of deadtime, arrising due to particle background variations, however, in case of bright sources
    the deadtime is defined by the sources itself. For all sky survey due to fast movement of the optical axis, a quate fast tracking of count rate
    variations should be checked (at least 1 s if we want to optical axis moved not more then 1.5' from its position

    lets consider, that the overall count rate consists of background and source, what we observe is several lightcurves i.e. in 4-30 and 40--100 kev and overall number of events, accumulated over 10 s
    assuming that the background spectrum conserves (which might be not true for the source due to intrinsic spectral variations or movement inside the fov with changing arf) we can introduce
    linear coefficient, connecting background count rate in any energy bands
    lets introudce 2 coefficients
    connecting overall background count rate with count rates in two selected energy bands

    one, which includes all the events from source and other, clean from source photons

    1) lc1 = s_f s' + k_1 b'     # where s_f is src fraction in measured lightcurve --> s' = (lc1 - k_1*lc2/k_2)/s_f
    2) lc2 = k_2 b'    -> b' = lc2/k_2

    lets consider the specific case of bright source with fast flux variations (for example flux changes for per cents during 10 s)
    and therefore we should accurately trace this count rate changes over time intervals shorter then 10 s

    we will assume that the background count rate (not corrected for the dead time) changes much slower (and therefore it can be assesed from the longer time intervals)
    b(t) = b (for t in [T0, T1] T1 - T0 >> 1s)
    the deadtime correction is as usual
    3) f(t) = 1 - (s' + b') \tau
    where \tau - is dead time
    we now can measure some values:
        B' = \int b' dt over long period of time (since the background count rate is releatively low
        and s' (in short time intervals)

    the b' = f(t) b  (where b is background count rate not affected by the dead time)

    4) \f(t) (1 + b\tau) = 1 - s'\tau
    then
    5) B' = \int b' dt = b \int (1 - s'(t)\tau)/(1 + b\tau) dt = b/(1 + b\tau) (T - \tau S')
    where S' is \int s' dt
    after that we obtain b (T - \tau (S' + B')) = B'
    and
    6) b = B'/(T - \tau (S' + B'))
    where we can measure S' + B' and B' with linear scaling coefficients from 1) and 2)
    after we obtain b
    we esimate fast variations of f with
    f = (1 - s'\tau)/(1 + b\tau)

    """

    bgtitot = reduce(lambda a, b: a & b, [d.filters["TIME"] for d in bkgevt.values()])
    bsimevt = {urdn: np.sum(bgtitot.mask_external(bkgevt[urdn]["TIME"])) for urdn in bkgevt}
    bsimevt = {urdn: d/np.sum(list(bsimevt.values())) for urdn, d in bsimevt.items()}
    urddtc = {}
    bkglc = {}

    for urdn in srcevt:
        dtloc = deadtime.get(urdn, get_deadtime_for_dev(urdn))
        te, gaps = srcevt[urdn].filters["TIME"].arange(dt)
        dts = np.diff(te)[gaps]
        tc = (te[1:] + te[:-1])[gaps]/2.
        glong = srcevt[urdn].filters["TIME"] + [-dtbkg/4., dtbkg/4.] + [dtbkg/4., -dtbkg/4.] # fill gaps smoller then dtbkg/2. i.e. if we have a series of short observations separated by short gaps
        teb, gapsb = glong.arange(dtbkg)
        shortidx = np.searchsorted(teb[1:][gapsb], tc)
        gif = srcevt[urdn].filters["TIME"].get_interpolation_function()
        dtb = gif.integrate_in_intervals(np.array([teb[:-1], teb[1:]]).T[gapsb])

        s1 = get_background_surface_brigtnress(urdn, bkgevt[urdn].filters, fill_value=0.).sum()
        s2 = get_background_surface_brigtnress(urdn, srcevt[urdn].filters, fill_value=0.).sum()
        s3 = get_totevt_during_bkg(urdn)

        scs = np.diff(np.searchsorted(srcevt[urdn]["TIME"], te))[gaps]
        bcsb = np.diff(np.searchsorted(bkgevt[urdn]["TIME"], teb))[gapsb]
        scsb = np.diff(np.searchsorted(srcevt[urdn]["TIME"], teb))[gapsb]
        blong = s3/s1*bcsb/(dtb - dtloc*(bcsb*s3/s1 + (scsb - bcsb*s2/s1)/srcfrac.get(urdn, 1.)))
        bkglc[urdn] = interp1d(teb[1:][gapsb], blong, bounds_error=False, fill_value=tuple(blong[[0, -1]]), kind="next")

        dtc = (1 - (scs - blong[shortidx]*dts*s2/s1)/srcfrac.get(urdn, 1)/dts*dtloc)/(1 + blong[shortidx]*dtloc)
        urddtc[urdn] = interp1d(te[1:][gaps], dtc, bounds_error=False, fill_value=tuple(dtc[[0, -1]]), kind="next")
    return urddtc, bkglc


def hkrate(hkdata, te, forcete=False):
    dcs = np.diff(hkdata["events"])
    mask = dcs > 0
    dcs = dcs[mask]
    dt = np.diff(hkdata["TIME"])[mask]
    tf = np.copy(hkdata["TIME"][1:])[mask]

    idx, ui, uc = np.unique(np.searchsorted(hkdata, te), return_index=True, return_counts=True)
    dcsc = np.cumsum(dcs)
    dtc = np.cumsum(dt)
    rates = dcsc[idx]/dtc[idx]
    if forcete:
        res = interp1d(te, np.repeat(rates, uc))
    else:
        res = interp1d(tf[idx], rates)
    return res



def overall_deadtime(srcevt, hk, dt=1., dtbkg=1000., fixedsrc=True):
    if not fixesrc:
        raise NotImplementedError("considered only the case, when all events are tide up")
    """
    lets now consider, that all srcsevts and bkg evets are correlated linearly between each other
    """
    bratesscales = {urdn: get_background_surface_brigtnress(urdn, bkgevt[urdn].filters, fill_value=0.).sum()/get_totevt_during_bkg(urdn) for urdn in srcevt}
    gtot = reduce(lambda a, b: a | b, [d.filters["TIME"] for d in srcevt.values()])
    teb, gapsb = gtot.arange(dtbkg)

    hkrates = {urdn: hkrate(d, teb, forcete=True) for urdn, d in hk.items()}
    """
    hkrates now contains both source and background
    csrc = int (k1*src + k2*bkg)*f dt
    ltot = int (k3*src + k4*bkg)*f dt

    lcs sum k_i*(b_i/k_i)
    dttot = (1 - k*
    """
    kbtot = np.ones(len(srcevt))
    bkfrc = np.array([get_background_surface_brigtnress(urdn, bkgevt[urdn].filters, fill_value=0.).sum()/get_totevt_during_bkg(urdn) for urdn in srcevt])
    sctcs = np.full(bktot.size, 1/bktot.size)
