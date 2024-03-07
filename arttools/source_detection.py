from .telescope import URDNS, concat_data_in_order
from ._det_spatial import vec_to_offset, get_qcorr_for_urddata, F
from math import pi, sin, cos, sqrt, log10
from .background import get_local_bkgrates, get_photon_vs_particle_prob, get_photon_and_particles_rates
from .psf import urddata_to_opaxoffset, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, select_psf_groups, photbkg_pix_coeff, naive_bispline_interpolation, psf_nearest_value, unpack_pix_index, ayutee
from .aux import DistributedObj
from .containers import Urddata
from .caldb import get_telescope_crabrates, get_ayut_inverse_psf_datacube_packed
from .planwcs import make_tan_wcs
from .vector import normalize, pol_to_vec, vec_to_pol
from .expmap import make_exposures, make_expmap_for_wcs
from .mosaic2 import WCSSky, get_source_photon_probability, get_zerosource_photstat
from .orientation import get_events_quats, get_photons_vectors
from .psf_functions import solve_for_locations, optimal_filter
from scipy.spatial.transform import Rotation
import tqdm

from scipy.optimize import minimize, root
from scipy.stats import chi2
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock
import numpy as np
from .src_rate_solvers import get_phc_solution, get_brent_solution
from time import time


urdcrates = get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}


def make_unipix_data(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0., cspec=None, urddtc={}):
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

    dtc = {}
    for urdn in urdevt:
        dtloc = urddtc.get(urdn, lambda x: np.ones(x.size, float))
        dtc[urdn] = dtloc(urdevt[urdn]["TIME"])
    dtc = concat_data_in_order(dtc)
    prate = prate*dtc

    """
    photprob = get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
    photprob = concat_data_in_order(photprob)
    """

    pbkgrate = {}
    for urdn in urdevt:
        profile = photbkg_pix_coeff(urdn, urdevt[urdn].filters)
        pbkgrate[urdn] = profile[urdevt[urdn]["RAW_X"], urdevt[urdn]["RAW_Y"]]*photbkgrate(urdevt[urdn], attdata) #attdata, urdevt)
    pbkgrate = concat_data_in_order(pbkgrate) # this is overall photon background rate (without spectarl information)# to add spectral info multiply by prate

    return i, j, qlist, prate, bkgrates*brate + pbkgrate*prate, eenergy


def make_detstat_tasks(urdevt, attdata, bkglc, urdweights=urdcrates, photbkgrate=lambda evt, att: 0., cspec=None):
    i, j, qlist, prate, brate, eenergy = make_unipix_data(urdevt, attdata, bkglc, urdweights, photbkgrate, cspec=cspec)
    pkoef = prate/brate

    ije, sidx, ss, sc = select_psf_groups(i, j, eenergy)
    tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]
    return tasks


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

def create_neighboring_blocks(locwcs, emap, i, j, qtot, pk, ee, rmap=None):
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
    sue = sus + suc

    mask = emap > 1.
    ii, jj = np.mgrid[0:mask.shape[0]:1, 0:mask.shape[1]:1]
    ii, jj = ii[mask], jj[mask]
    ipix = ii//sizex + mx*(jj//sizey)
    sidx = np.argsort(ipix)
    ii, jj, ipix = ii[sidx], jj[sidx], ipix[sidx]
    piu, pus, puc = np.unique(ipix, return_index=True, return_counts=True)
    ishift = np.array([(k%3 -1) + (k//3 - 1)*mx for k in range(9)])

    mask = np.logical_or.reduce([np.isin(piu + s, siu) for s in ishift])
    piu, pus, puc = piu[mask], pus[mask], puc[mask]
    pue = pus + puc
    #print("max events size per block", puc.max())

    def get_idxg_vals(k):
        x, y = ii[pus[k]:pue[k]], jj[pus[k]:pue[k]]
        exp = emap[x, y]
        nsl = piu[k] + ishift
        nsl = np.searchsorted(siu, nsl[np.isin(nsl, siu, assume_unique=True)], sorter=ssorter)
        idxg = np.concatenate([np.arange(sus[nl], sue[nl]) for nl in nsl])
        if rmap is None:
            return x, y, exp, i[idxg], j[idxg], ee[idxg], pk[idxg], qtot[idxg]
        else:
            return x, y, exp, rmap[x, y], i[idxg], j[idxg], ee[idxg], pk[idxg], qtot[idxg]

    return pus.size, get_idxg_vals


class BlockEstimator(DistributedObj):
    def __init__(self, locwcs, mpnum=4, barrier=None):
        self.locwcs = locwcs
        super().__init__(mpnum, barrier, locwcs=locwcs)


    @DistributedObj.for_each_argument
    def get_nphot_and_theta(self, x, y, exp, i, j, ee, pk, qtot):
        #return estimate_rate_for_direction_iterate(self.locwcs, x, y, exp, i, j, ee, pk, qtot)
        return estimate_rate_for_direction_c(self.locwcs, x, y, exp, i, j, ee, pk, qtot)

    @DistributedObj.for_each_argument
    def get_nbkg_in_s_step(self, x, y, exp, i, j, ee, pk, qtot):
        vt = pol_to_vec(*np.deg2rad(self.locwcs.all_pix2world(np.array([y, x]).T, 0)).T)
        ssize = 1000000 # number of event for split
        csplit = (x.size*i.size)//1000000 + 1
        csize = x.size//csplit + 1
        csplit = x.size//csize + (1 if x.size%csize > 0 else 0)
        ic = np.tile(i, csize)
        jc = np.tile(j, csize)
        eec = np.tile(ee, csize)
        qtotc = Rotation(np.tile(qtot.as_quat(), (csize, 1)))
        ntot = np.zeros(x.size, float)

        for sl in range(csplit):
            v = vt[sl*csize: x.size if sl == csplit - 1 else (sl + 1)*csize]
            nphot = ntot[sl*csize: x.size if sl == csplit -1 else (sl + 1)*csize]
            if ic.size != nphot.size*i.size:
                ic = ic[:nphot.size*i.size] #np.tile(i, nphot.size)
                jc = jc[:ic.size] #np.tile(i, nphot.size)
                eec = eec[:ic.size] #np.tile(i, nphot.size)
                qtotc = qtotc[:ic.size]

            vr = np.repeat(v, ic.size//nphot.size, axis=0)
            m, bw = naive_bispline_interpolation(ic, jc, qtotc.apply(vr, inverse=True), eec)
            nphot[:] = m.reshape((nphot.size, -1)).sum(axis=1)
        return x, y, ntot

def make_srccount_and_detmap(locwcs, emap, urde, attdata, bkglc, photbkgrate=lambda evt, att: 0., urdweights=urdcrates, cspec=None, mpnum=4):
    i, j, qtot, prate, brate, ee = make_unipix_data(urde, attdata, bkglc, photbkgrate=photbkgrate, urdweights=urdweights, cspec=cspec)
    cmap = np.zeros(emap.shape, float)
    pmap = np.zeros(emap.shape, float)
    pk = prate/brate

    psfdata = get_ayut_inverse_psf_datacube_packed()
    ntasks, feeder = create_neighboring_blocks(locwcs, emap, i, j, qtot, pk, ee)
    iifun = get_ipsf_interpolation_func()
    dx = iifun.grid[0][1] - iifun.grid[0][0]
    dy = iifun.grid[1][1] - iifun.grid[1][0]
    xsize = iifun.grid[0].size
    ysize = iifun.grid[1].size

    def worker(args):
        x, y, exp, i, j, ee, pk, qtot = args
        eidx = np.searchsorted(ayutee, ee) - 1
        rmat = qtot.as_matrix()
        vt = pol_to_vec(*np.deg2rad(locwcs.all_pix2world(np.array([y, x]).T, 0)).T)
        return x, y, solve_for_locations(i,j,eidx,rmat,pk,vt,exp,psfdata,dx,xsize,dy,ysize)
        #return x, y, solve_for_locations(i,j,eidx,rmat,pk,vt,psfdata,dx,xsize,dy,ysize)

    pool = ThreadPool(mpnum)
    for x, y, (cl, pl) in tqdm.tqdm(pool.imap_unordered(worker, (feeder(i) for i in range(ntasks))), total=ntasks):
        cmap[x, y] = cl
        pmap[x, y] = pl
    return cmap, pmap

def make_optimal_filter_solution(locwcs, emap, rmap, urde, attdata, bkglc, photbkgrate=lambda evt, att: 0., urdweights=urdcrates, cspec=None, mpnum=4):
    i, j, qtot, prate, brate, ee = make_unipix_data(urde, attdata, bkglc, photbkgrate=photbkgrate, urdweights=urdweights, cspec=cspec)
    cmap = np.zeros(emap.shape, float)
    pmap = np.zeros(emap.shape, float)
    pk = prate/brate

    psfdata = get_ayut_inverse_psf_datacube_packed()
    ntasks, feeder = create_neighboring_blocks(locwcs, emap, i, j, qtot, pk, ee, rmap=rmap)
    iifun = get_ipsf_interpolation_func()
    dx = iifun.grid[0][1] - iifun.grid[0][0]
    dy = iifun.grid[1][1] - iifun.grid[1][0]
    xsize = iifun.grid[0].size
    ysize = iifun.grid[1].size

    def worker(args):
        x, y, exp, rates, i, j, ee, pk, qtot = args
        eidx = np.searchsorted(ayutee, ee) - 1
        rmat = qtot.as_matrix()
        vt = pol_to_vec(*np.deg2rad(locwcs.all_pix2world(np.array([y, x]).T, 0)).T)
        return x, y, optimal_filter(i,j,eidx,rmat,pk,vt,exp,rates,psfdata,dx,xsize,dy,ysize)
        #return x, y, solve_for_locations(i,j,eidx,rmat,pk,vt,psfdata,dx,xsize,dy,ysize)

    pool = ThreadPool(mpnum)
    for x, y, cl in tqdm.tqdm(pool.imap_unordered(worker, (feeder(i) for i in range(ntasks))), total=ntasks):
        pmap[x, y] = cl
    return pmap

def estimate_rate_for_direction_exact(vec, exposure, i, j, ee, pk, qtot):
    m, bw = naive_bispline_interpolation(i, j, qtot.apply(vec, inverse=True), ee)
    svals = bw*pk[m]
    guess = np.sum(svals/(svals*m.sum()/exposure + 1.))/exposure
    return root(lambda x: np.sum(svals/(svals*x + 1.)) - exposure, guess).x[0]

def ppsolver(nphot, bw, cs, expl, itnum=200):
    bwc = np.copy(bw)
    mtot = cs > 0
    csc = cs[mtot]
    css = np.cumsum(csc) - 1
    nc = csc.astype(float)
    nn = np.empty(nc.size, float)
    explc = expl[mtot]
    for _ in range(itnum):
        cres = np.cumsum(1./(1. + np.repeat(explc/nc, csc)/bwc))
        nn[1:] = np.diff(cres[css])#*np.sign(nphot[mtot][1:])
        nn[0] = cres[css[0]]
        #print("nn0", nn[0])
        nphot[mtot] = nn
        mnotdone = ~np.logical_or((nn <= nc) & (nn < 0.001), np.abs(nn - nc) < 1e-5)
        #mnotdone = ~(np.abs(nn - nc) < 1e-5)
        if ~np.any(mnotdone):
            break
        mtot[mtot] = mnotdone
        bwc = bwc[np.repeat(mnotdone, csc)]
        csc = csc[mnotdone]
        css = np.cumsum(csc) - 1
        explc = explc[mnotdone]
        nc, nn = nn[mnotdone], nc[mnotdone]
    return nphot


def estimate_rate_for_direction_iterate(locwcs, x, y, exposure, i, j, ee, pk, qtot, ratesolver="python"):
    data, mask = None, None
    vt = pol_to_vec(*np.deg2rad(locwcs.all_pix2world(np.array([y, x]).T, 0)).T)
    ssize = 1000000 # number of event for split
    csplit = (x.size*i.size)//1000000 + 1
    csize = x.size//csplit
    csplit = x.size//csize + (1 if x.size%csize > 0 else 0)
    ntot = np.zeros(x.size, float)
    thet = np.zeros(x.size, float)
    ic = np.tile(i, csize)
    jc = np.tile(j, csize)
    eec = np.tile(ee, csize)
    pkc = np.tile(pk, csize)
    mrot = qtot.inv().as_matrix()

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

        m, bw = naive_bispline_interpolation(ic, jc, np.einsum("kj,mij->kmi", v, mrot).reshape((-1, 3)), eec)
        """
        vr = np.einsum("kj,mij->kmi", v, mrot).reshape((-1, 3))
        m, bw, data, mask = psf_nearest_value(ic, jc, vr, kc, energy=eec, data=data, mask=mask)
        """

        if m.sum() == 0:
            continue

        cs = m.reshape((nphot.size, -1)).sum(axis=1)
        bw = bw*pkc[m]
        rates = np.maximum(cs.astype(float)/2., 1.)/expl
        b = np.ones(bw.size, float)
        tstart = time()
        if ratesolver == "python":
            ppsolver(nphot, bw, cs, expl)
            rates = nphot/expl
        if ratesolver == "c":
            get_phc_solution(bw, b, rates, expl, cs)
            nphot[:] = rates*expl
        if ratesolver == "brent":
            get_brent_solution(bw, b, rates, expl, cs)
            nphot[:] = rates*expl
        #print("executed in", time() - tstart)

        cc = np.cumsum(bw/(bw*np.repeat(rates, cs) + 1.))
        ch = np.empty(expl.size, float)
        ch[1:] = np.diff(cc[np.cumsum(cs) - 1]) - expl[1:]
        ch[0] = cc[cs[0] - 1]- expl[0]

        t = np.cumsum(np.log(bw*np.repeat(rates, cs) + 1))
        qest[1:] = np.diff(t[np.cumsum(cs) - 1])
        qest[:1] = t[cs[0] - 1]
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


def get_nearest_local_maxima(ra0, dec0, urdevt, attdata, bkglc, urdweights=urdcrates, illum=None, photbkgrate=lambda evt, att: 0., cspec=None, rsearcharound=10., lwcsemap=None):
    """
    expmap is expected to be a function, which returns real exposure corresponding to events, stored in tasks
    """
    ax0 = pol_to_vec(ra0*pi/180., dec0*pi/180.)
    u1 = {urdn: Urddata(d.data[np.sum(get_photons_vectors(d, urdn, attdata)*ax0, axis=1) > cos(pi/180.*rsearcharound/60.)], urdn, d.filters) for urdn, d in urdevt.items()}
    i, j, qtot, prate, brate, ee = make_unipix_data(u1, attdata, bkglc, urdweights, photbkgrate, cspec=cspec)
    attloc = attdata.apply_gti(attdata.circ_gti(ax0, 1800 + rsearcharound*60.))
    if lwcsemap is None:
        sx = int(rsearcharound*60.)
        sx = sx + sx*2 - 1
        lwcs = make_tan_wcs(ra0*pi/180., dec0*pi/180., sizex=sx, sizey=sx, pixsize=1./3600.)
        eml = make_expmap_for_wcs(lwcs, attloc, urdevt, urdweights=urdweights)
    else:
        lwcs, eml = lwcsemap

    def lklfun(var):
        ax = pol_to_vec(*var)
        yl, xl = lwcs.all_world2pix([[var[0]*180/pi, var[1]*180/pi],], 0).T
        if yl < 0 or xl < 0 or xl > eml.shape[0] - 1 or yl > eml.shape[1] - 1:
            return 0
        expl = eml[int(xl + 0.5), int(yl + 0.5)]
        #expl = make_exposures(ax, np.array([-np.inf, np.inf]), attloc, {urdn: d.filters for urdn, d in urdevt.items()}, urdweights=urdcrates)[1] #, illum_filters=ifilters)
        mask, w = naive_bispline_interpolation(i, j, qtot.apply(ax, inverse=True), ee)
        pk = w*prate[mask]/brate[mask]
        res = root(lambda x: np.sum(1./(x + 1./pk)) - expl, 1.)
        return expl*res.x[0] - np.sum(np.log(res.x[0]*pk + 1.))

    likelihood = minimize(lklfun, [ra0*pi/180, dec0*pi/180.], method="Nelder-Mead")
    if likelihood.success:
        ax = pol_to_vec(*likelihood.x)
        yl, xl = lwcs.all_world2pix([[likelihood.x[0]*180/pi, likelihood.x[1]*180/pi],], 0).T
    else:
        ax = pol_to_vec(lwcs.wcs.crval[0]*pi/180, lwcs.wcs.crval[1]*pi/180.)
        yl, xl = lwcs.wcs.crpix
    expl = eml[int(xl + 0.5), int(yl + 0.5)]
    #expl = make_exposures(ax, np.array([-np.inf, np.inf]), attdata, {urdn: d.filters for urdn, d in urdevt.items()}, urdweights=urdcrates)[1] #, illum_filters=ifilters)
    mask, w = naive_bispline_interpolation(i, j, qtot.apply(ax, inverse=True), ee)
    res = root(lambda x: np.sum(1./(x + brate[mask]/(w*prate[mask]))) - expl, 1.)
    return likelihood, expl, res

def get_rate_confidence_intervals(ra, dec, urdevt, attdata, bkglc, quantiles = [0.68, 0.9], urdweights=urdcrates, photbkgrate=lambda evt, att: 0., cspec=None, srcexp=None):
    ax = pol_to_vec(ra*pi/180., dec*pi/180.)
    dllist = chi2.isf(1 - np.array(quantiles), 1)/2.
    if srcexp is None:
        expl = make_exposures(ax, np.array([-np.inf, np.inf]), attdata, {urdn: d.filters for urdn, d in urdevt.items()}, urdweights=urdcrates)[1]
    else:
        expl = srcexp
    i, j, qtot, prate, brate, ee = make_unipix_data(urdevt, attdata, bkglc, urdweights, photbkgrate, cspec=cspec)
    mask, w = naive_bispline_interpolation(i, j, qtot.apply(ax, inverse=True), ee)
    pk = w*prate[mask]/brate[mask]
    rerr = np.empty((len(quantiles), 2), float)
    rguess = np.sum(pk/(pk + 1.))/expl
    rguess = np.sum(rguess*pk/(pk*rguess + 1.))/expl
    rguess = np.sum(rguess*pk/(pk*rguess + 1.))/expl
    rguess = np.sum(rguess*pk/(pk*rguess + 1.))/expl
    ropt = root(lambda x: np.sum(pk/(pk*x + 1.)) - expl, rguess).x[0]
    print("resulted optimized rate solution", pk.size, "guess", rguess, ropt, "suggested exposure", expl)
    if ropt < 0:
        ropt = 0.
        rerr[:, 0] = 0.
        for q, dlkl in enumerate(dllist):
            rerr[q, 1] = root(lambda x: np.sum(np.log((pk*ropt + 1.)/(pk*x + 1))) + expl*(x - ropt) - dlkl, 1.).x[0]
    else:
        for q, dlkl in enumerate(dllist):
            rerr[q, 0] = root(lambda x: np.sum(np.log((pk*ropt + 1.)/(pk*x + 1))) + expl*(x - ropt) - dlkl, ropt/2.).x[0]
            rerr[q, 1] = root(lambda x: np.sum(np.log((pk*ropt + 1.)/(pk*x + 1))) + expl*(x - ropt) - dlkl, ropt*2.).x[0]
    rerr[rerr < 0.] = 0.
    return np.sum(np.log(pk*ropt + 1.)) - ropt*expl, ropt, rerr
