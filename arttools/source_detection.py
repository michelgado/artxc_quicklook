from .telescope import URDNS, concat_data_in_order
from ._det_spatial import vec_to_offset, get_qcorr_for_urddata, F
from math import pi, sin, cos, sqrt, log10
from .background import get_local_bkgrates, get_photon_vs_particle_prob, get_photon_and_particles_rates
from .psf import urddata_to_opaxoffset, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, select_psf_groups, photbkg_pix_coeff, naive_bispline_interpolation
from .aux import DistributedObj
from .containers import Urddata
from .caldb import get_telescope_crabrates
from .planwcs import make_tan_wcs
from .vector import normalize, pol_to_vec, vec_to_pol
from .expmap import make_exposures, make_expmap_for_wcs
from .mosaic2 import WCSSky, get_source_photon_probability, get_zerosource_photstat
from .orientation import get_events_quats, get_photons_vectors
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, root
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock
import numpy as np

urdcrates = get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}


def make_unipix_data(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0., cspec=None):
    bkgrates = {urdn: get_local_bkgrates(urdevt[urdn], bkglc[urdn]) for urdn in URDNS if urdn in urdevt}
    bkgrates = concat_data_in_order(bkgrates)

    qlist = [Rotation(np.empty((0, 4), np.double)) if urdevt[urdn].size == 0 else get_events_quats(urdevt[urdn], urdn, attdata)*get_qcorr_for_urddata(urdevt[urdn]) for urdn in URDNS if urdn in urdevt]
    qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

    i, j = zip(*[urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in URDNS if urdn in urdevt])
    i, j = np.concatenate(i), np.concatenate(j)

    eenergy = concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

    prate, brate = {}, {}
    for urdn in urdevt:
        p, b = get_photon_and_particles_rates(urdevt[urdn], cspec)
        prate[urdn] = p*urdweights.get(urdn, 1./7.)
        brate[urdn] = b

    prate = concat_data_in_order(prate)
    brate = concat_data_in_order(brate)
    """
    photprob = get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
    photprob = concat_data_in_order(photprob)
    """

    pbkgrate = {}
    for urdn in urdevt:
        profile = photbkg_pix_coeff(urdn, urdevt[urdn].filters)
        pbkgrate[urdn] = profile[urdevt[urdn]["RAW_X"], urdevt[urdn]["RAW_Y"]]*photbkgrate(urdevt[urdn], attdata) #attdata, urdevt)
    pbkgrate = concat_data_in_order(pbkgrate) # this is overall photon background rate (without spectarl information)# to add spectral info multiply by prate

    #pkoef = photprob/(bkgrates + pbkgrate*photprob)
    return i, j, qlist, prate, bkgrates*brate + pbkgrate*prate, eenergy


def make_detstat_tasks(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0.):
    i, j, qlist, prate, brate, eenergy = make_unipix_data(urdevt, attdata, bkglc, urdweights, photbkgrate)
    pkoef = prate/brate

    ije, sidx, ss, sc = select_psf_groups(i, j, eenergy)
    tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]
    return tasks


class Single_position_estimator(object):
    def __init__(self, sky, emap):
        self.sky = sky
        self.results = []
        self.emap = emap
        self.stopit = False
        self.thread = Thread(target = lambda : None)
        self.thread.start()

    def run_worker(self):
        for idx in self.idx:
            if self.stopit:
                break
            xi, yi = np.unravel_index(idx, self.sky.img.shape)
            self.results.append(estimate_rate_for_direction_fast(self.sky.vecs[xi, yi],  self.emap[xi, yi], self.i, self.j, self.ee, self.pk, self.qtot))

    def get_results(self, join=True):
        if join:
            self.thread.join()
        return self.results

    def update_order(self, idx):
        self.idx = idx

    def update_tasks(self, tasks):
        self.tasks = tasks


    def stopthread(self):
        self.stopit = True

    def run_new_cycle(self, idx, i, j, ee, pk, qtot, join=False):
        self.stopit = True
        self.idx = idx
        self.i = i
        self.j = j
        self.ee =  ee
        self.pk = pk
        self.qtot = qtot
        res = self.get_results()
        if join:
            self.thread.join()
        self.stopit = False
        self.results = []
        self.thread = Thread(target=self.run_worker)
        self.thread.start()
        return res


def make_detmap(locwcs, emap, tasks, sky=None, mpnum=20, maxit=101, ctot=None, update_mask=True):
    vmap = get_ipsf_interpolation_func()
    if sky is None:
        sky = WCSSky(locwcs, vmap, mpnum=mpnum)
    else:
        sky.set_vmap(vmap)

    mask = emap > 1.
    sky.set_mask(mask)

    sky.set_action(get_source_photon_probability)
    if ctot is None:
        ctot = np.ones(emap.shape, float)*np.sum([t[1].size for t in tasks])/2.
        rmap = np.maximum(ctot, 0.)/np.maximum(emap, 1.)
    else:
        rmap = np.maximum(ctot, 0.)/np.maximum(emap, 1.)
    sky.set_rmap(rmap) #2./np.maximum(emap, 1.))
    ctasks = tasks

    mdec = np.zeros(ctot.shape, bool)
    for _ in range(maxit):
        sky.clean_image()
        ctasks = [(q, s, c) if np.all(m) else (q[m], s[m], c) for (q, s, c), m in zip(ctasks, sky.rmap_convolve_multicore(ctasks, ordered=True, total=len(ctasks))) if np.any(m)]
        sky.img[:, :] = 0.
        sky.accumulate_img()
        mold = np.copy(sky.mask)
        if update_mask:
            mc = sky.img < ctot
            mdec[:, :] = mc & mdec # number of photons has decreased twice in a row
            mtot = np.logical_and.reduce([sky.mask, ~(mdec & (sky.img < 0.5)), np.abs(sky.img - ctot) > np.maximum(ctot, 2)*5e-3])
            sky.set_mask(mtot)
            mdec = mc

        print("img mask", sky.mask.size, sky.mask.sum(), "total events", np.sum([t[1].size for t in ctasks]))
        print("zero photons cts: ", sky.mask.sum(), "conv hist", np.histogram(np.abs(ctot[sky.mask] - sky.img[sky.mask])/ctot[sky.mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 10000.]))
        ctot[mold] = np.copy(sky.img[mold])
        sky.set_rmap(ctot/np.maximum(emap, 1.))
        if not np.any(sky.mask):
            break

    sky.clean_image()
    sky.set_action(get_zerosource_photstat)
    sky.set_rmap(np.maximum(ctot, 0.)/np.maximum(emap, 1.))
    sky.set_mask(emap > 1.)
    sky.img[:, :] = 0.
    sky.rmap_convolve_multicore(tasks, total=len(tasks))
    return ctot, np.copy(sky.img)

def make_detmap_with_conv(locwcs, emap, urde, attdata, bkglc, mpnum=20, maxit=101, sky=None, ctot=None, update_mask=True, urdweights=urdcrates):
    tasks, i, j, qtot, pk, ee, sidx, ss, sc = make_detstat_tasks(urde, attdata, bkglc, urdweights=urdweights, return_aux=True)
    vmap = get_ipsf_interpolation_func()

    if sky is None:
        sky = WCSSky(locwcs, vmap, mpnum=mpnum)
    else:
        sky.set_vmap(vmap)

    mask = emap > 1.
    sky.set_mask(mask)

    sky.set_action(get_source_photon_probability)
    if ctot is None:
        ctot = np.ones(emap.shape, float)*np.sum([t[1].size for t in tasks])/2.
        rmap = np.maximum(ctot, 0.)/np.maximum(emap, 1.)
    else:
        rmap = np.maximum(ctot, 0.)/np.maximum(emap, 1.)
    sky.set_rmap(rmap) #2./np.maximum(emap, 1.))
    mtasks = np.ones(i.size, bool)
    tidx = range(len(tasks))

    single_thread_estimator = Single_position_estimator(sky, emap)

    mdec = np.zeros(ctot.shape, bool)
    for _ in range(maxit):
        sky.clean_image()
        tnew = []
        for tid, m in zip(tidx, sky.rmap_convolve_multicore((tasks[i] for i in tidx), ordered=True, total=len(tidx))):
            #print(tid, m.size, len(m))
            if np.any(m):
                tnew.append(tid)
                mtasks[sidx[ss[tid]:ss[tid] + sc[tid]]] = m

        #ctasks = [(q, s, c) if np.all(m) else (q[m], s[m], c) for (q, s, c), m in zip(ctasks, sky.rmap_convolve_multicore(ctasks, ordered=True, total=len(ctasks))) if np.any(m)]
        single_thread_estimator.stopthread()
        print("thread stopped")
        sky.img[:, :] = 0.
        sky.accumulate_img()
        mold = np.copy(sky.mask)
        if update_mask:
            mc = sky.img < ctot
            mdec[:, :] = mc & mdec # number of photons has decreased twice in a row
            mtot = np.logical_and.reduce([sky.mask, ~(mdec & (sky.img < 0.5)), np.abs(sky.img - ctot) > np.maximum(ctot, 2)*5e-3])

            res = np.array(single_thread_estimator.get_results())
            print("thread result: ", res.size)
            if res.size > 0:
                mtot.ravel()[idx[:res.size]] = False
                ctot.ravel()[idx[:res.size]] = res
            idx = np.where(mtot.ravel())[0][np.argsort(sky.img[mtot])[::-1]]
            single_thread_estimator.run_new_cycle(idx, i[mtasks], j[mtasks], ee[mtasks], pk[mtasks], qtot[mtasks])
            sky.set_mask(mtot)
            mdec = mc

        print("img mask", sky.mask.size, sky.mask.sum(), "total events", mtasks.sum())
        print("zero photons cts: ", sky.mask.sum(), "conv hist", np.histogram(np.abs(ctot[sky.mask] - sky.img[sky.mask])/ctot[sky.mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 10000.]))
        ctot[mold] = np.copy(sky.img[mold])
        sky.set_rmap(ctot/np.maximum(emap, 1.))
        if not np.any(sky.mask):
            break

    sky.clean_image()
    sky.set_action(get_zerosource_photstat)
    sky.set_rmap(np.maximum(ctot, 0.)/np.maximum(emap, 1.))
    sky.set_mask(emap > 1.)
    sky.img[:, :] = 0.
    sky.rmap_convolve_multicore(tasks, total=len(tasks))
    return ctot, np.copy(sky.img)


def create_neighbouring_blocks_tasks(locwcs, emap, urde, attdata, bkglc, photbkgrate=lambda evt, att: 0., urdweights=urdcrates, cspec=None):
    i, j, qtot, prate, brate, ee = make_unipix_data(urde, attdata, bkglc, photbkgrate=photbkgrate, urdweights=urdweights, cspec=cspec)
    pk = prate/brate
    vmap = get_ipsf_interpolation_func()
    sizex = int(np.arctan(max(np.max(np.abs(vmap.grid[0][[0, -1]])), np.max(np.abs(vmap.grid[1][[0, -1]])))/F)*180/pi/np.min(locwcs.wcs.cdelt[1])*sqrt(2.)) + 2
    sizey = int(np.arctan(max(np.max(np.abs(vmap.grid[0][[0, -1]])), np.max(np.abs(vmap.grid[1][[0, -1]])))/F)*180/pi/np.min(locwcs.wcs.cdelt[0])*sqrt(2.)) + 2
    xy = (locwcs.all_world2pix(np.rad2deg(vec_to_pol(qtot.apply([1, 0, 0]))).T, 0) + 0.5).astype(int)[:, ::-1]
    mx = (emap.shape[0] + sizex - 1)//sizex
    srcidx = xy[:, 0]//sizex + mx*(xy[:, 1]//sizey)
    sidx = np.argsort(srcidx)
    i, j, qtot, pk, ee, srcidx = i[sidx], j[sidx], qtot[sidx], pk[sidx], ee[sidx], srcidx[sidx]

    siu, sus, suc = np.unique(srcidx, return_index=True, return_counts=True)
    ssorter = np.arange(siu.size)
    #print("unique blocks", siu.size)
    sue = sus + suc

    mask = emap > 1.
    ii, jj = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]
    #jj, ii = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]
    ii, jj = ii[mask], jj[mask]
    ipix = ii//sizex + mx*(jj//sizey)
    sidx = np.argsort(ipix)
    ii, jj, ipix = ii[sidx], jj[sidx], ipix[sidx]
    piu, pus, puc = np.unique(ipix, return_index=True, return_counts=True)
    ishift = np.array([(k%3 -1) + (k//3 - 1)*mx for k in range(9)])

    mask = np.logical_or.reduce([np.isin(piu + s, siu) for s in ishift])
    piu, pus, puc = piu[mask], pus[mask], puc[mask]
    print("number of wcs blocks to solve", piu.size)
    pue = pus + puc
    #print("upix and uevts", piu, siu)
    #no = np.searchsorted(siu, piu)

    x, y = ii[pus[0]:pue[0]], jj[pus[0]:pue[0]],
    exp = emap[x, y]
    nsl = piu[0] + ishift
    #print(piu[k], nsl)
    nsl = np.searchsorted(siu, nsl[np.isin(nsl, siu, assume_unique=True)], sorter=ssorter)
    idxg = np.concatenate([np.arange(sus[nl], sue[nl]) for nl in nsl])
    for k, n in enumerate(piu[1:]):
        yield x, y, exp, i[idxg], j[idxg], ee[idxg], pk[idxg], qtot[idxg] #, srcidx[idxg]
        x, y = ii[pus[k + 1]:pue[k+ 1]], jj[pus[k + 1]:pue[k + 1]]
        exp = emap[x, y]
        nsl = n + ishift
        #print(piu[k], nsl)
        nsl = np.searchsorted(siu, nsl[np.isin(nsl, siu, assume_unique=True)], sorter=ssorter)
        idxg = np.concatenate([np.arange(sus[nl], sue[nl]) for nl in nsl])
    yield x, y, exp, i[idxg], j[idxg], ee[idxg], pk[idxg], qtot[idxg] #, srcidx[idxg]

class BlockEstimator(DistributedObj):
    def __init__(self, locwcs, mpnum=4, barrier=None):
        self.locwcs = locwcs
        super().__init__(mpnum, barrier, locwcs=locwcs)


    @DistributedObj.for_each_argument
    def get_nphot_and_theta(self, x, y, exp, i, j, ee, pk, qtot):
        return estimate_rate_for_direction_iterate(self.locwcs, x, y, exp, i, j, ee, pk, qtot)


def estimate_rate_for_direction_fast(vec, exposure, i, j, ee, pk, qtot):
    m, bw = naive_bispline_interpolation(i, j, qtot.apply(vec, inverse=True), ee)
    svals = bw*pk[m]
    guess = np.sum(svals/(svals*m.sum()/exposure + 1.))/exposure
    return root(lambda x: np.sum(svals/(svals*x + 1.)) - exposure, guess).x[0]


def estimate_rate_for_direction_iterate(locwcs, x, y, exposure, i, j, ee, pk, qtot):
    vt = pol_to_vec(*np.deg2rad(locwcs.all_pix2world(np.array([y, x]).T, 0)).T)
    ssize = 2000000 # number of event for split
    csplit = (x.size*i.size)//2000000 + 1
    csize = x.size//csplit
    csplit = x.size//csize + (1 if x.size%csize > 0 else 0)
    ntot = np.zeros(x.size, float)
    thet = np.zeros(x.size, float)
    ic = np.tile(i, csize)
    jc = np.tile(j, csize)
    eec = np.tile(ee, csize)
    pkc = np.tile(pk, csize)
    qtotc = Rotation(np.tile(qtot.as_quat(), (csize, 1)))

    for sl in range(csplit):
        v = vt[sl*csize: x.size if sl == csplit - 1 else (sl + 1)*csize]
        nphot = ntot[sl*csize: x.size if sl == csplit -1 else (sl + 1)*csize]
        qest = thet[sl*csize: x.size if sl == csplit -1 else (sl + 1)*csize]
        expl = exposure[sl*csize: x.size if sl == csplit -1 else (sl + 1)*csize]
        if ic.size != nphot.size*i.size:
            ic = ic[:nphot.size*i.size] #np.tile(i, nphot.size)
            jc = jc[:ic.size] #np.tile(i, nphot.size)
            eec = eec[:ic.size] #np.tile(i, nphot.size)
            pkc = pkc[:ic.size] #np.tile(i, nphot.size)
            qtotc = qtotc[:ic.size]

        vr = np.repeat(v, ic.size//nphot.size, axis=0)

        m, bw = naive_bispline_interpolation(ic, jc, qtotc.apply(vr, inverse=True), eec)
        if m.sum() == 0:
            continue
        #print("bw.size", bw.size)
        cs = m.reshape((nphot.size, -1)).sum(axis=1)
        #print(cs.size, bw.size, np.sum(cs > 0), np.sum(cs))
        bw = bw*pkc[m]
        bwc = np.copy(bw)
        mtot = cs > 0
        csc = cs[mtot]
        css = np.cumsum(csc) - 1
        nc = csc.astype(float)
        nn = np.empty(nc.size, float)
        explc = expl[mtot]
        #print("before check", bw.size, csc.sum(), csc.size, nc.size, explc.size)
        for _ in range(200):
            cres = np.cumsum(1./(1. + np.repeat(explc/nc, csc)/bwc))
            nn[1:] = np.diff(cres[css])
            nn[0] = cres[css[0]]
            nphot[mtot] = nn
            mnotdone = ~np.logical_or((nn <= nc) & (nn < 1.), np.abs(nn - nc) < 1e-3)
            #print("converg", mnotdone.sum())
            if ~np.any(mnotdone):
                break
            mtot[mtot] = mnotdone
            bwc = bwc[np.repeat(mnotdone, csc)]
            csc = csc[mnotdone]
            css = np.cumsum(csc) - 1
            explc = explc[mnotdone]
            nc, nn = nn[mnotdone], nc[mnotdone]
        t = np.cumsum(np.log(bw*np.repeat(nphot/expl, cs) + 1))
        qest[1:] = np.diff(t[np.cumsum(cs) - 1])
        qest[0] = t[cs[0] - 1]
        qest[(cs == 0) | (nphot < 0.5)] == 0.
    return x, y, ntot, thet





def estimate_rate_for_direction(srcvec, exposure, tasks):
    vmap = get_ipsf_interpolation_func()
    svals = []
    """
    qtot = Rotation(np.concatenate([q[0].as_quat() for q in tasks], axis=0))
    csize = np.cumsum([0, ] + [q[1].size for q in tasks])
    vall = qtot.apply(srcvec, inverse=True)
    mask =  vall[:, 0] > offset_to_vec(vmap.grid[0][0], vmap.grid[1][0])[0]
    """
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


def get_nearest_local_maxima(ra0, dec0, urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0., cspec=None, rsearcharound=10.):
    """
    expmap is expected to be a function, which returns real exposure corresponding to events, stored in tasks
    """
    ax0 = pol_to_vec(ra0*pi/180., dec0*pi/180.)
    u1 = {urdn: Urddata(d.data[np.sum(get_photons_vectors(d, urdn, attdata)*ax0, axis=1) > cos(pi/180.*rsearcharound/60.)], urdn, d.filters) for urdn, d in urdevt.items()}
    i, j, qtot, prate, brate, ee = make_unipix_data(u1, attdata, bkglc, urdweights, photbkgrate)
    attloc = attdata.apply_gti(attdata.circ_gti(ax0, 1800 + rsearcharound*60.))
    sx = int(rsearcharound*60.)
    sx = sx + sx*2 - 1
    lwcs = make_tan_wcs(ra0, dec0, sizex=sx, sizey=sx, pixsize=1./3600.)
    eml = make_expmap_for_wcs(lwcs, attloc, urdevt, urdweights=urdweights)
    def lklfun(var):
        print(var)
        ax = pol_to_vec(*var)
        yl, xl = lwcs.all_world2pix([[var[0]*180/pi, var[1]*180/pi],], 0)
        expl = eml[int(xl + 0.5), int(yl + 0.5)]
        #expl = make_exposures(ax, np.array([-np.inf, np.inf]), attloc, {urdn: d.filters for urdn, d in urdevt.items()}, urdweights=urdcrates)[1] #, illum_filters=ifilters)
        mask, w = naive_bispline_interpolation(i, j, qtot.apply(ax, inverse=True), ee)
        pk = w*prate[mask]/brate[mask]
        res = root(lambda x: np.sum(1./(x + 1./pk)) - expl, 1.)
        return expl*res.x[0] - np.sum(np.log(res.x[0]*pk + 1.))

    likelihood = minimize(lklfun, [ra0*pi/180, dec0*pi/180.], method="Nelder-Mead")
    ax = pol_to_vec(*likelihood.x)
    yl, xl = lwcs.all_world2pix([[likelihood.x[0]*180/pi, likelihood.x[1]*180/pi],], 0)
    expl = eml[int(xl + 0.5), int(yl + 0.5)]
    #expl = make_exposures(ax, np.array([-np.inf, np.inf]), attdata, {urdn: d.filters for urdn, d in urdevt.items()}, urdweights=urdcrates)[1] #, illum_filters=ifilters)
    mask, w = naive_bispline_interpolation(i, j, qtot.apply(ax, inverse=True), ee)
    res = root(lambda x: np.sum(1./(x + brate[mask]/(w*prate[mask]))) - expl, 1.)
    return likelihood, exp, res
