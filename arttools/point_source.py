from astropy.wcs import WCS
from .expmap import make_expmap_for_wcs, make_exposures
from .caldb import get_boresight_by_device
from .background import get_local_bkgrates
from .orientation import  get_events_quats, get_photons_vectors
from .vector import vec_to_pol, normalize
from ._det_spatial import get_qcorr_for_urddata, vec_to_offset_pairs, raw_xy_to_vec
from .mosaic2 import SkyImage
from .psf import get_ipsf_interpolation_func, urddata_to_opaxoffset, unpack_inverse_psf_ayut, photbkg_pix_coeff, select_psf_groups, naive_bispline_interpolation, get_iicore_normalization
from .telescope import URDNS, concat_data_in_order
from .background import get_photon_vs_particle_prob, get_local_bkgrates, get_background_surface_brigtnress, get_photon_to_particle_rate_ratio
from .filters import IndependentFilters, Intervals
from scipy.spatial.transform import Rotation
from scipy.optimize import root, minimize
from functools import reduce
import numpy as np
from math import cos, pi, log


class Observation(object):
    def __init__(self, attdata, urddata, urdfilters, urdbkg, urdgti, urdweights={}):
        bkgrates = {urdn: get_local_bkgrates(urdn, urdbkg[urdn], urdfilters[urdn], urddata[urdn]) for urdn in URDNS}
        bkgrates = concat_data_in_order(bkgrates)
        urdns = URDNS
        qlist = [Rotation(np.empty((0, 4), np.double)) if urddata[urdn].size == 0 else get_events_quats(urddata[urdn], urdn, attdata)*get_qcorr_for_urddata(urddata[urdn]) for urdn in URDNS if urdn in urddata]
        self.qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

        i, j = zip(*[urddata_to_opaxoffset(urddata[urdn], urdn) for urdn in URDNS if urdn in urddata])
        self.i, self.j = np.concatenate(i), np.concatenate(j)
        self.energy = concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urddata.items()})

        photprob = get_photon_vs_particle_prob(urdfilters, urddata, urdweights=urdweights)
        photprob = concat_data_in_order(photprob)
        self.pkoef = photprob/bkgrates

        self.attdata = attdata
        self.urdgti = urdgti
        self.urdweights = urdweights
        self.urdfilters = urdfilters
        self.vecs = self.qlist.apply([1, 0, 0])
        self.tgti = reduce(lambda a, b: a | b, self.urdgti.values())


    def propose_location(self):
        """
        compute location assuming, that majority of events are produced by  the source
        """
        ipsffuncs = []
        for il, jl, en, qc in zip(self.i, self.j, self.energy, self.qlist):
            ipsffuncs.append(get_ipsf_interpolation_func())
            ipsffuncs[-1].values = unpack_inverse_psf_ayut(il, jl, en)

        def comprob(ax):
            return  -np.sum(ifun(vec_to_offset_pairs(qc.apply(ax, inverse=True)))[0] for ifun, qc in zip(ipsffuncs, self.qlist))

        vinit = normalize(np.sum(self.qlist.apply([1, 0, 0]), axis=0))
        vres = minimize(comprob, vinit, method="Nelder-Mead")
        return vres.x


    def get_rate_and_prob(self, ax, app=600., **kwargs):
        ax = normalize(ax)
        te, gaps = self.tgti.arange(1e6)
        te, dtn = make_exposures(ax, te, self.attdata, self.urdgti, self.urdfilters, self.urdweights, app=app, **kwargs)
        texp = np.sum(dtn)
        rate = self.i.size/dtn
        ipsffun = get_ipsf_interpolation_func()
        mask = np.sum(self.vecs*ax, axis=-1) > cos(pi/180.*app/3600.)
        vals = []
        for il, jl, en, qc in zip(self.i[mask], self.j[mask], self.energy[mask], self.qlist[mask]):
            ipsffun.values = unpack_inverse_psf_ayut(il, jl, en)
            vals.append(ipsffun(vec_to_offset_pairs(qc.apply(ax, inverse=True)))[0])
        vals = np.array(vals)
        res = root(lambda x: np.sum(vals*self.pkoef[mask]/(1. + x*vals*self.pkoef[mask])) - texp, self.i.size/texp)
        rate = res.x[0]
        pval = np.sum(np.log(1. + rate*vals*self.pkoef[mask]))
        return rate, pval - rate*texp


def get_events_photweights(ax, attdata, urddata, urdbkg, photbkg=0., app=120., urdweights={}, filters=IndependentFilters({}), cspec=None):
    gti = attdata.circ_gti(ax, 25.*60.)
    times = []
    photprob = []
    ratecoeff = []
    partrate = []
    pbkgrate = []
    energy = []
    iifun = get_ipsf_interpolation_func()
    x, y = np.mgrid[0:48:1, 0:48:1]
    v = raw_xy_to_vec(x.ravel(), y.ravel()).reshape(list(x.shape) + [3,])
    cfilters = IndependentFilters({"TIME": gti}) & filters
    for urdn in urddata:
        urdevt = urddata[urdn].apply_filters(IndependentFilters({"TIME": gti}) & filters)
        if urdevt.data.size == 0:
            continue

        vecs = v[urdevt["RAW_X"], urdevt["RAW_Y"]] #get_photons_vectors(urdevt, urdn, attdata)

        photbkgpixprofile = photbkg_pix_coeff(urdn, urdevt.filters, cspec) #*urdweights.get(urdn, 1/7.)
        partbkgpixprofile = get_background_surface_brigtnress(urdn, urdevt.filters, fill_value=0., normalize=True)
        srcvec = (attdata(urdevt["TIME"])*get_boresight_by_device(urdn)).apply(ax, inverse=True)

        mask = np.sum(vecs*srcvec, axis=1) > cos(pi/180.*app/3600.)

        urdevt.data = urdevt.data[mask]
        srcvec = srcvec[mask]
        imask, weights = naive_bispline_interpolation(urdevt["RAW_X"], urdevt["RAW_Y"], srcvec, urdevt["ENERGY"], urdn)
        urdevt.data = urdevt.data[imask]
        mask[mask] = imask

        phottoparticleprob = get_photon_to_particle_rate_ratio(urdevt)*urdweights.get(urdn, 1./7.)
        times.append(urdevt["TIME"])
        ratecoeff.append(weights)
        photprob.append(phottoparticleprob)
        partrate.append(urdbkg[urdn](urdevt["TIME"])*partbkgpixprofile[urdevt["RAW_X"], urdevt["RAW_Y"]])
        pbkgrate.append(photbkgpixprofile[urdevt["RAW_X"], urdevt["RAW_Y"]]*photbkg)
        energy.append(urdevt["ENERGY"])

    #print(type(energy), type(times), type(ratecoeff), type(photprob), type(partrate), type(pbkgrate))
    #print(len(energy), len(times), len(ratecoeff), len(photprob), len(partrate), len(pbkgrate))
    #print(type(energy[0]), type(times[0]), type(ratecoeff[0]), type(photprob[0]), type(partrate[0]), type(pbkgrate[0]))

    #print([type(e) for e in energy], [type(t) for t in times], [type(r) for r in ratecoeff], [type(p) for p in photprob], [type(p) for p in partrate], [type(b) for b in pbkgrate])

    #print([e.size for e in energy], [t.size for t in times], [r.size for r in ratecoeff], [p.size for p in photprob], [p.size for p in partrate], [b.size for b in pbkgrate])

    edata = np.array([np.concatenate(arr) for arr in [times, ratecoeff, photprob, partrate, pbkgrate, energy]])
    idx = np.argsort(edata[0])
    return edata[:, idx]



def get_localization(qlist, i, j, pkoef, attdata, urdgti, urdfilters):
    """
    also we have not to assume, that count rate is constant, but assuming, that optical axis movement are not correlated with flux variations
    this can be done by comparing the models of point
    """
    return None

"""
def simmulate_source_events(attdata, urdn, urdfilters, ax, r, cspec=None):
    attloc = attdata.apply_gti(ax, 30.*60.)

    tel, gaps, locgti = make_small_steps_quats(attloc)
    tc = (tel[1:] + tel[:-1])[gaps]/2.

    shiftsize = int(300//45 + 1)
    xc, yc = np.mgrid[-shiftsize: shiftsize + 1: 1, -shiftsize: shiftsize + 1: 1] # size of the pixel is 45 arcsec

    dtn = np.zeros(te.size - 1, np.double)

    x, y = np.mgrid[0:48:1, 0:48:1]
    vecs = raw_xy_to_vec(x.ravel(), y.ravel())
    urddata = {}
    for urdn in urdfilters:
        if (urdfilters["TIME"] & locgti).length:
            urddata[urdn] = np.empty(0, dtype=[("RAW_X", int), ("RAW_Y": int), ("GRADE", int), ("ENERGY", float), ("TIME", float)])
            continue

        shmask = urdfilters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        teu, gaps = (urdfilters[urdn]["TIME"] & locgti).make_tedges(tel)
        dtu = np.diff(teu)[gaps]
        tcc = (teu[1:] + teu[:-1])/2.
        tc = tcc[gaps]
        qlist = attdloc(tc)*get_boresight_by_device(urdn)
        vsrc = qlist.apply(direction, inverse=True)

        xs, ys = offset_to_raw_xy(*vec_to_offset(vsrc))
        xs, ys = xs[:, np.newaxis] + xc.ravel()[np.newaxis, :], ys[:, np.newaxis] + yc.ravel()[np.newaxis, :]
        mask = np.logical_and.reduce([xs > -1, xs < 48, ys > -1, ys < 48])
        mask[mask] = shmask[xs[mask], ys[mask]]
        cvals = mask.sum(axis=1)
        xs, ys = xs[mask], ys[mask]
        qlist = Rotation(np.repeat(qlist.as_quat(), cvals, axis=0))

        grid, spec = get_filtered_crab_spectrum(urdfilters[urdn])
        ew = get_specweights(urdfilters[urdn])
        ec = (grid["ENERGY"][1:] + grid["ENERGY"][:-1])/2.
        data = np.sum(get_ayut_inverse_psf_datacube_packed()*ew[np.newaxis, :, np.newaxis, np.newaxis], axis=1)
        imgmax = np.sum([np.sum(unpack_inverse_psf_ayut(i, j)*w[:, np.newaxis, np.newaxis], axis=0)[60 - i*9, 60 - j*9]*8/(1. + (i == j))/(1. + (i == 0.))/(1. + (j == 0.)) for i in range(5) for j in range(5)])
        data = data/imgmax #(data[0, 60, 60] + data[1, 60, 51]*4 + data[2, 51, 51]*4)


        millum = self.check_pixel_in_illumination(urdn, xs, ys, qlist, vecs.reshape((48, 48, 3))[xs, ys])
        mupd, w = naive_bispline_interpolation_specweight(xs[millum], ys[millum], np.repeat(vsrc, cvals, axis=0)[millum], ipsfdata, urdn)

        millum[millum] = mupd

        if np.any(mupd):
            idx = np.searchsorted(te, np.repeat(tc, cvals)[millum]) - 1

            dtu = np.repeat(dtu, cvals)[millum]
            dtc = np.repeat(dtcorr[urdn](tc), cvals)[millum] if urdn in dtcorr else 1.
            print("urdn", urdn, " time", np.sum(dtu*w))
            np.add.at(dtn, idx, w*dtu*dtc*urdweights.get(urdn, 1./7.))
            print("urdweights", urdweights.get(urdn, 1./7.))
    print("dtn sum", dtn.sum())
    return dtn
"""



