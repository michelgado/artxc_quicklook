from .telescope import URDNS, concat_data_in_order
from ._det_spatial import vec_to_offset, get_qcorr_for_urddata
from .background import get_local_bkgrates, get_photon_vs_particle_prob
from .psf import urddata_to_opaxoffset, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, select_psf_groups, photbkg_pix_coeff
from .caldb import get_telescope_crabrates
from .vector import normalize, pol_to_vec, vec_to_pol
from .mosaic2 import WCSSky, get_source_photon_probability, get_zerosource_photstat
from .orientation import get_events_quats
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, root
import numpy as np

urdcrates = get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}


def make_detstat_tasks(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=0.):
    bkgrates = {urdn: get_local_bkgrates(urdevt[urdn], bkglc[urdn]) for urdn in URDNS}
    bkgrates = concat_data_in_order(bkgrates)

    print("quats")
    qlist = [Rotation(np.empty((0, 4), np.double)) if urdevt[urdn].size == 0 else get_events_quats(urdevt[urdn], urdn, attdata)*get_qcorr_for_urddata(urdevt[urdn]) for urdn in URDNS if urdn in urdevt]
    qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

    i, j = zip(*[urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in URDNS if urdn in urdevt])
    i, j = np.concatenate(i), np.concatenate(j)

    eenergy = concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

    photprob = get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
    photprob = concat_data_in_order(photprob)

    pbkgrate = {}
    for urdn in urdevt:
        profile = photbkg_pix_coeff(urdn, urdevt[urdn].filters)
        pbkgrate[urdn] = profile[urdevt[urdn]["RAW_X"], urdevt[urdn]["RAW_Y"]]*photbkgrate
    pbkgrate = concat_data_in_order(pbkgrate)

    pkoef = photprob/(bkgrates + pbkgrate*photprob)

    ije, sidx, ss, sc = select_psf_groups(i, j, eenergy)
    tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]
    return tasks


def make_detmap(locwcs, emap, tasks, mpnum=20, maxit=101):
    vmap = get_ipsf_interpolation_func()
    sky = WCSSky(locwcs, vmap, mpnum=mpnum)

    mask = emap > 1.
    sky.set_mask(mask)

    sky.set_action(get_source_photon_probability)
    ctot = np.ones(emap.shape, float)*np.sum([t[1].size for t in tasks])/2.
    rmap = np.maximum(ctot, 1.)/np.maximum(emap, 1.)
    sky.set_rmap(2./np.maximum(emap, 1.))
    ctasks = tasks

    for _ in range(maxit):
        sky.clean_image()
        ctasks = [(q, s, c) if np.all(m) else (q[m], s[m], c) for (q, s, c), m in zip(ctasks, sky.rmap_convolve_multicore(ctasks, ordered=True, total=len(ctasks))) if np.any(m)]
        sky.accumulate_img()
        mold = np.copy(sky.mask)
        sky.set_mask(np.all([sky.mask, ~((sky.img < ctot) & (sky.img < 0.5)), np.abs(sky.img - ctot) > np.maximum(ctot, 2)*5e-3], axis=0))
        print("img mask", sky.mask.size, sky.mask.sum(), "total events", np.sum([t[1].size for t in ctasks]))
        print("zero photons cts: ", sky.mask.sum(), "conv hist", np.histogram(np.abs(ctot[sky.mask] - sky.img[sky.mask])/ctot[sky.mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 10000.]))
        ctot[mold] = np.copy(sky.img[mold])
        sky.set_rmap(ctot/np.maximum(emap, 1.))
        sky.img[:, :] = 0.
        if not np.any(sky.mask):
            break

    sky.clean_image()
    sky.set_action(get_zerosource_photstat)
    sky.set_rmap(ctot/np.maximum(emap, 1.))
    sky.set_mask(emap > 1.)
    sky.img[:, :] = 0.
    sky.rmap_convolve_multicore(tasks, total=len(tasks))
    return ctot, np.copy(sky.img)

def estimate_rate_for_direction(srcvec, exposure, tasks):
    vmap = get_ipsf_interpolation_func()
    svals = []
    for ql, pk, vcore in tasks:
        vmap.values = vcore
        svals.append(vmap(vec_to_offset(ql.apply(srcvec, inverse=True)))*pk)
    svals = np.concatenate(svals)
    svals = svals[svals > 0.]
    guess = np.sum(np.minimum(svals, 1.))
    solution = root(lambda x: np.sum(svals/(svals*x + 1.)) - exposure, [guess,])
    return solution.x[0], svals


def make_wcs_nearest_interpolator(wcs, scalarmap):
    def nearest_interpolator(ra, dec):
        y, x = (np.array(wcs.all_world2pix([ra, dec], 0)) + 0.5).astype(int)
        return scalarmap[x, y]
    return nearest_interpolator

def get_nosource_likelihood_ratio_for_direction(srcvec, exposure, tasks):
    rate, svals = estimate_rate_for_direction(srcvec, exposure, tasks)
    return np.sum(np.log(svals*rate + 1.) - exp*rate)

def get_local_maxima(urdevt, bkglc, expmap, urdweights=urdcrates):
    """
    expmap is expected to be a function, which returns real exposure corresponding to events, stored in tasks
    """
    tasks = make_detstat_tasks(urdevt, bkglc, urdweights)
    def prepare_likelihood_estimation(ra, dec):
        exposure = expmap(ra, dec)
        srcvec = pol_to_vec(*np.deg2rad([ra, dec]).reshape((2, -1)))[0]
        return get_nosource_likelihood_ratio_for_direction(srcvec, exposure, tasks)

    mvec = normalize(np.sum(np.concatenate([q.apply([1, 0, 0]) for q, _, _ in tasks], axis=0)))
    guess = np.rad2deg(vec_to_pol(mvec))
    likelihood = minimize(lambda x: -prepare_likelihood_estimation(ra, dec), guess)
    srcvec = pol_to_vec(*np.deg2rad(likelihood.x).reshape((2, -1)))[0]
    return likelihood, estimate_rate_for_direction(srcvec, expmap(*likelihood.x), tasks)
