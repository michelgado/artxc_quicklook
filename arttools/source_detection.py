from .telescope import URDNS, concat_data_in_order
from ._det_spatial import vec_to_offset, get_qcorr_for_urddata, F
from math import pi, sin, cos, sqrt, log10
from .background import get_local_bkgrates, get_photon_vs_particle_prob
from .psf import urddata_to_opaxoffset, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, select_psf_groups, photbkg_pix_coeff, naive_bispline_interpolation
from .aux import DistributedObj
from .caldb import get_telescope_crabrates
from .vector import normalize, pol_to_vec, vec_to_pol
from .mosaic2 import WCSSky, get_source_photon_probability, get_zerosource_photstat
from .orientation import get_events_quats
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize, root
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock
import numpy as np

urdcrates = get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}


def make_detstat_tasks(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0., return_aux=False):
    bkgrates = {urdn: get_local_bkgrates(urdevt[urdn], bkglc[urdn]) for urdn in URDNS if urdn in urdevt}
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
        pbkgrate[urdn] = profile[urdevt[urdn]["RAW_X"], urdevt[urdn]["RAW_Y"]]*photbkgrate(urdevt[urdn], attdata) #attdata, urdevt)
    pbkgrate = concat_data_in_order(pbkgrate)

    pkoef = photprob/(bkgrates + pbkgrate*photprob)

    ije, sidx, ss, sc = select_psf_groups(i, j, eenergy)
    tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]
    if return_aux:
        return tasks, i, j, qlist, pkoef, eenergy, sidx, ss, sc
    else:
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


def create_neighbouring_blocks_tasks(locwcs, emap, urde, attdata, bkglc, photbkgrate=lambda evt, att: 0., urdweights=urdcrates):
    tasks, i, j, qtot, pk, ee, sidx, ss, sc = make_detstat_tasks(urde, attdata, bkglc, photbkgrate=photbkgrate, urdweights=urdweights, return_aux=True)
    vmap = get_ipsf_interpolation_func()
    sizex = int(np.arctan(max(np.max(np.abs(vmap.grid[0][[0, -1]])), np.max(np.abs(vmap.grid[1][[0, -1]])))/F)*180/pi/np.min(locwcs.wcs.cdelt[1])*sqrt(2.)) + 2
    sizey = int(np.arctan(max(np.max(np.abs(vmap.grid[0][[0, -1]])), np.max(np.abs(vmap.grid[1][[0, -1]])))/F)*180/pi/np.min(locwcs.wcs.cdelt[0])*sqrt(2.)) + 2
    xy = (locwcs.all_world2pix(np.rad2deg(vec_to_pol(qtot.apply([1, 0, 0]))).T, 0) + 0.5).astype(int)[:, ::-1]
    mx = (emap.shape[0] + sizex - 1)//sizex
    print(sizex, sizey, mx)
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
        x, y = ii[pus[k + 1]:pue[k+ 1]], jj[pus[k + 1]:pue[k + 1]],
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
            cres = np.cumsum(1./(1. + np.repeat(explc, csc)/bwc/np.repeat(nc, csc)))
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
        qest[cs < 0.5] == 0.

        """
        bww = np.zeros(ic.size, float)
        bww[m] = bw
        bww = bww.reshape((nphot.size, -1))*pk[np.newaxis, :]
        bww = np.sort(bww, axis=1)
        bww = bww[:, np.any(bww > 0, axis=0)]
        nphot[:] = np.sum(bww > 0, axis=1).astype(float)
        mdone = np.ones(bww.shape[0], bool)
        for _ in range(100):
            #npnew = np.sum(bww[mdone, :]*nphot[mdone, np.newaxis]/(bww[mdone, :]*nphot[mdone, np.newaxis] + exposure[mdone, np.newaxis]), axis=1)
            npnew = np.sum(1./(1. + expl[mdone, np.newaxis]/bww[mdone, :]/nphot[mdone, np.newaxis]), axis=1)
            mred = ~((npnew < nphot[mdone]) & (npnew < 1.))
            nphot[mdone] = npnew
            if ~np.any(mred):
                break
            mdone[mdone] = mred
        qest[:] = np.sum(np.log(bww*(nphot/expl)[:, np.newaxis] + 1.), axis=1)
        """
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
