from .caldb import get_crabspec, get_telescope_crabrates, get_crabspec_for_filters, get_arf, get_optical_axis_offset_by_device
from .filters import Intervals
from .energy  import get_arf_energy_function
from scipy.integrate import quad

import numpy as np

def get_filtered_crab_spectrum(filters, collapsegrades=False):
    grid, datacube = get_crabspec()
    menergy = filters["ENERGY"].apply(grid["ENERGY"])
    menergys = np.logical_and(menergy[1:], menergy[:-1])
    mgrade = filters["GRADE"].apply(grid["GRADE"])
    rgrid = {"ENERGY": grid["ENERGY"][menergy], "GRADE": grid["GRADE"][mgrade]}
    if collapsegrades:
        return {"ENERGY":rgrid["ENERGY"]}, datacube[menergys, :][:, mgrade].sum(axis=1)
    else:
        return rgrid, datacube[menergys, :][:, mgrade]

class Spec(object):
    def __init__(self, el, eh, flux):
        self.el = el
        self.eh = eh
        self.flux = flux
        self.defined_range = Intervals(np.array([el, eh]).T)


    def integrate_in_bins(self, bins):
        ee = np.sort(np.concatenate([self.el, self.eh, bins.ravel()]))
        ec = (ee[1:] + ee[:-1])/2.
        de = (ee[1:] - ee[:-1])
        computeband = Intervals(bins) & self.defined_range
        de = de[computeband.mask_external(ec)]
        ec = ec[computeband.mask_external(ec)]
        idxout = np.searchsorted(self.el, ec) - 1
        idxin = np.searchsorted(bins[:, 0], ec) -1
        res = np.zeros(bins.shape[0])
        np.add.at(res, idxin, self.flux[idxout]*de)
        return res

def get_specweights(imgfilter, ebins=np.array([-np.inf, np.inf]), cspec=None):
    w = np.zeros(ebins.size - 1, np.double)
    if not cspec is None:
        egloc, egaps = imgfilter["ENERGY"].make_tedges(ebins)
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        arf = get_arf_energy_function(get_arf())
        cspec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(egloc[:-1][egaps], egloc[1:][egaps])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
        cspec = cspec/cspec.sum()
        np.add.at(w, np.searchsorted(ebins, ec) - 1, cspec)
    else:
        rgrid, cspec = get_filtered_crab_spectrum(imgfilter, collapsegrades=True)
        crabspec = Spec(rgrid["ENERGY"][:-1], rgrid["ENERGY"][1:], cspec)
        egloc, egaps = imgfilter["ENERGY"].make_tedges(np.unique(np.concatenate([ebins, rgrid["ENERGY"]])))
        ec = (egloc[1:] + egloc[:-1])[egaps]/2.
        cspec = crabspec.integrate_in_bins(np.array([egloc[:-1], egloc[1:]]).T[egaps])
        cspec = cspec.sum()
        np.add.at(w, np.searchsorted(ebins, ec) - 1, cspec)
    return w

def get_spec_filters_ratios(cspec, imgfilter1, imgfilter2):
    arf = get_arf_energy_function(get_arf())
    rate1 = sum(quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in imgfilter1.filter["ENERGY"].arr)
    rate2 = sum(quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in imgfilter2.filter["ENERGY"].arr)
    return rate1/rate2

def get_crabrate(imgfilters):
    grid, datacube = get_crabspec()
    crates = get_telescope_crabrates()
    ocrate = datacube.sum()
    ofcrate = sum(get_filtered_crab_spectrum(f)[1].sum()*crates[urdn] for urdn, f in imgfilters.items())/ocrate
    return ofcrate


def get_specprate(emin, emax, cspec):
    arf = get_arf_energy_function(get_arf())
    return quad(lambda e: arf(e)*cspec(e), emin, emax)[0]


def get_events_crab_weights(grid, spec, udata):
    #grid, spec = get_background_spectrum(filters)
    gidx = np.zeros(grid["GRADE"].max() + 1, np.int)
    gidx[grid["GRADE"]] = np.arange(grid["GRADE"].size)
    idxe = np.searchsorted(grid["ENERGY"], udata["ENERGY"]) - 1
    idxg = gidx[udata["GRADE"]]
    return spec[idxe, idxg]/np.sum(spec)


def make_mock_events(size, imgfilters, urdweights={}, cspec=None, pixdist=lambda x, y: np.ones(x.size)):
    events = {}
    x, y = np.mgrid[0:48:1, 0:48:1]

    for urdn in imgfilters:
        gridp, specp = get_crabspec_for_filters(imgfilters[urdn])
        specp = (specp/np.diff(gridp["ENERGY"])[:, np.newaxis])/specp.sum()
        shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        xl, yl = x[shmask], y[shmask] #
        xo, yo = get_optical_axis_offset_by_device(urdn)
        #pixcrate = get_pix_overall_countrate_constbkg_ayut(imgfilters[urdn], cspec)
        pixvol = np.cumsum(pixdist((xl - xo).astype(int), (yl - yo).astype(int)))
        if not cspec is None:
            arf = get_arf_energy_function(get_arf())
            cspec = np.array([quad(lambda e: arf(e)*cspec(e), elow, ehi)[0] for elow, ehi in zip(gridp["ENERGY"][:-1], gridp["ENERGY"][1:])]) #np.concatenate([cspec/cspec.sum()/np.diff(egrid), [0, ]])
            cspec = cspec/cspec.sum()
            specp = specp*(cspec/specp.sum(axis=1))[:, np.newaxis]
            specp = specp/specp.sum()

        pvol = specp.ravel().cumsum()

        n = np.random.poisson(size*urdweights.get(urdn, 1./7.))
        idx = np.searchsorted(pvol, np.random.uniform(0, pvol[-1], n))
        i, j = np.unravel_index(idx, specp.shape)
        events[urdn] = np.empty(n, [("TIME", float), ("RAW_X", int), ("RAW_Y", int), ("ENERGY", float), ("GRADE", int)])
        events[urdn]["ENERGY"] = (gridp["ENERGY"][1:] + gridp["ENERGY"][:-1])[i]/2.
        events[urdn]["GRADE"] = gridp["GRADE"][j]
        idx = np.searchsorted(pixvol, np.random.uniform(0, pixvol[-1], n))
        events[urdn]["RAW_X"] = xl[idx]
        events[urdn]["RAW_Y"] = yl[idx]

    return events


def get_crabflux(emin, emax):
    return 9.21*1.6e-9/0.1*(emin**(-0.1) - emax**(-0.1))

def get_crabpflux(emin, emax):
    return 9.21/1.1*(emin**(-1.1) - emax**(-1.1))
