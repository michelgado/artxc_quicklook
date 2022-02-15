import numpy as np
from _det_spatial import offset_to_qcorr

import arttools
import os

from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos, sqrt, pi
from arttools.time import get_gti, GTI, tGTI, emptyGTI, deadtime_correction
from arttools.planwcs import make_wcs_for_attdata
from arttools.caldb import get_energycal, get_shadowmask, get_energycal_by_urd, get_shadowmask_by_urd, urdbkgsc, OPAXOFFSET
from arttools.energy import get_events_energy
from arttools.telescope import URDNS
from arttools.orientation import AttDATA, get_attdata
from arttools.lightcurve import make_overall_lc, weigt_time_intervals
from arttools.vignetting import load_raw_wignetting_function
from arttools.plot import make_energies_flags_and_grades, make_sky_image
from astropy.io import fits
from math import pi, cos, sin, tan
from multiprocessing import Pool, cpu_count, Queue, Process, Pipe
from threading import Thread
from functools import reduce
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import pyds9

from arttools.expmap import make_expmap_for_wcs
from arttools.background import make_bkgmap_for_wcs
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

import sys
import pyds9


PSFSCALE = 300 #arcsec
MPNUM=20

def make_rate_map(bkglcfun, detbmap, x, y, t):
    return detbmap[x, y]*bkglcfun(t)


def prepare_data(attdata, elist, urdn):
    xi0 = int(arttools.caldb.OPAXOFFSET[urdn][0])
    yi0 = int(arttools.caldb.OPAXOFFSET[urdn][1])
    x = elist["RAW_X"] - xi0 + 26 #magic number  26 is from created invert PSF which computed for 53x53 pix matrix with center of detecting pixel in the center of  [26, 26] pixel
    y = elist["RAW_Y"] - yi0 + 26
    qlist = attdata(elist["TIME"])*arttools.caldb.get_boresight_by_device(urdn)*offset_to_qcorr(elist["RAW_X"], elist["RAW_Y"])
    return x, y, qlist


iPSFscale = pi/180.*5./60.
iPSFcorners = np.array([[1, -tan(iPSFscale), -tan(iPSFscale)],
                        [1, -tan(iPSFscale), tan(iPSFscale)],
                        [1, tan(iPSFscale), -tan(iPSFscale)],
                        [1, tan(iPSFscale), tan(iPSFscale)]])


def get_source_photon_probability(core, bkg, rate):
    return rate*core/(bkg + rate*core)

def get_zerosource_photstat(core, bkg, rate):
    return np.log(bkg/(bkg + core*rate))


class PutiPSF(object):

    def __init__(self, qin, qout, locwcs, rmap, rfun):
        self.qin = qin
        self.qout  = qout
        self.rmap = rmap
        self.rfun = rfun
        self.locwcs = locwcs
        self._set_corners()
        self.img = np.zeros((int(self.locwcs.wcs.crpix[1]*2 + 1), int(self.locwcs.wcs.crpix[0]*2 + 1)), np.double)
        self.y, self.x = np.mgrid[1:self.img.shape[0] + 1:1, 1:self.img.shape[1] + 1:1]
        self.ra, self.dec = self.locwcs.all_pix2world(np.array([self.x.ravel(), self.y.ravel()]).T, 1).T
        self.ra, self.dec = self.ra.reshape(self.x.shape), self.dec.reshape(self.x.shape)
        self.vecs = arttools.vector.pol_to_vec(*np.deg2rad([self.ra, self.dec]))

    def _set_corners(self, vals=iPSFcorners):
        self.corners = vals

    def __call__(self):
        xvmap = np.tan(np.arange(-300, 301, 5)*pi/180/3600)*arttools._det_spatial.F
        yvmap = np.tan(np.arange(-300, 301, 5)*pi/180/3600)*arttools._det_spatial.F

        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            qvals, vm, bkg = vals
            vmap = RegularGridInterpolator((xvmap, yvmap), vm, bounds_error=False, fill_value=0.)

            for i in range(len(qvals)):
                qval = qvals[i]
                ra, dec = arttools.vector.vec_to_pol(qval.apply(self.corners))
                x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T
                jl, jr = max(int(x.min()), 0), min(self.img.shape[1] - 1, int(x.max()+1))
                il, ir = max(int(y.min()), 0), min(self.img.shape[0] - 1, int(y.max()+1))

                #x, y = self.x[il:ir + 1, jl: jr + 1], self.y[il: ir + 1, jl: jr + 1]
                #ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T, 1).T
                vecs = np.copy(self.vecs[il:ir + 1, jl: jr + 1].reshape((-1, 3)))
                xl, yl = arttools._det_spatial.vec_to_offset(qval.apply(vecs, inverse=True))
                self.img[il:ir+1, jl:jr+1] += self.rfun(vmap((xl, yl)).reshape((ir + 1 - il, jr + 1 - jl)),
                                                bkg[i], self.rmap[il:ir+1, jl: jr + 1])
        self.qout.put(self.img)

    @staticmethod
    def trace_and_collect(qvals, vmaps, bkgrate, u, ui, uc, qin, qout, pool, accumulate, *args):
        for proc in pool:
            proc.start()

        for i in range(uc.size):
            sl = slice(ui[i], ui[i] + uc[i])
            #print(u[i], bkgrate[sl])
            qin.put([qvals[sl], vmaps[u[i, 0], u[i, 1]], bkgrate[sl]])
            sys.stderr.write('\rdone {0:%}'.format(i/uc.size))

        for p in pool:
            qin.put(-1)

        res = accumulate(qout, len(pool), *args)

        for p in pool:
            p.join()
        return res

    @staticmethod
    def accumulate(qout, size, *args):
        return sum(qout.get() for i in range(size))

    @classmethod
    def make_mp(cls, locwcs, vmaps, qvals, x, y, bkgrates, rmap, rfun, *args, mpnum=MPNUM, **kwargs):
        pairs = np.array([x, y]).T
        idxsort = np.argsort(x*53 + y)
        pairs = pairs[idxsort]
        qvals = qvals[idxsort]
        u, ui, uc = np.unique(pairs, axis=0, return_counts=True, return_index=True)
        qin = Queue(100)
        qout = Queue(10)
        pool = [Process(target=cls(qin, qout, locwcs, rmap, rfun, **kwargs)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(qvals, vmaps, bkgrates, u, ui, uc, qin, qout, pool, cls.accumulate, *args)
        return resimg


if __name__ == "__main__":
    """
    gti = pickle.load(open("/srg/a1/work/andrey/ART-XC/gc/lp20gti.pickle", "rb"))
    ddirs = [os.path.dirname(fname) for fname in gti.keys()]
    urdflist = []
    for dname in ddirs:
        urdflist += [os.path.join(dname, l.rstrip()) for l in os.listdir(dname)]

    attdata = arttools.orientation.AttDATA.concatenate([arttools.plot.get_attdata(fname) for fname in gti.keys()])
    gti = reduce(lambda a, b: a| b, gti.values())
    attdata = attdata.apply_gti(gti + [-50, 50])
    pickle.dump([gti, attdata, urdflist], open("lp20_srcdet.pickle", "wb"))
    """
    gti, attdata, urdflist = pickle.load(open("lp20_srcdet.pickle", "rb"))
    ax = arttools.vector.pol_to_vec(*np.deg2rad([278.38688, -10.571942]))
    gti = gti & arttools.time.GTI([6.24269e+8 + 6800, np.inf]) #& attdata.circ_gti(ax, pi/180.) #attdata.get_axis_movement_speed_gti(lambda x: x > pi/180.*10./3600.)
    urdflist = [name for name in urdflist if "urd.fits" in name]


    pixsize = 10./3600.
    locwcs = arttools.planwcs.make_wcs_for_attdata(attdata, gti, pixsize) #produce wcs for accumulated atitude information
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)

    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdbkge = {}
    urdhk = {}
    urdevt = {}
    imgdata = 0

    gti = gti & attdata.gti
    """

    for urdfname in urdflist[:]:
        print(urdfname)
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]

        locgti = (get_gti(urdfile, "STDGTI") if "STDGTI" in urdfile else get_gti(urdfile)) & gti & ~urdgti.get(urdn, emptyGTI)
        #locgti = locgti & ~urdbti.get(urdn, emptyGTI)
        if locgti.exposure == 0.:
            continue
        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        urddata = urddata[locgti.mask_external(urddata["TIME"])]

        hkdata = np.copy(urdfile["HK"].data)
        hkdata = hkdata[(locgti + [-30, 30]).mask_external(hkdata["TIME"])]
        urdhk[urdn] = urdhk.get(urdn, []) + [hkdata,]

        energy, grade, flag = make_energies_flags_and_grades(urddata, hkdata, urdn)
        timemask = locgti.mask_external(urddata["TIME"])
        pickimg = np.all([energy > 4., energy < 12., grade > -1, grade < 10,
                            flag == 0, locgti.mask_external(urddata["TIME"])], axis=0)
        if np.any(pickimg):
            urdevt[urdn] = urdevt.get(urdn, []) + [urddata[pickimg],]
            timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, 1)
            imgdata += timg

        pickbkg = np.all([energy > 40., energy < 100., grade > -1, grade < 10,
                          urddata["RAW_X"] > 0, urddata["RAW_X"] < 47, urddata["RAW_Y"] > 0, urddata["RAW_Y"] < 47], axis=0)
        bkgevts = urddata["TIME"][pickbkg]
        urdbkge[urdn] = urdbkge.get(urdn, []) + [bkgevts,]

    urdevt = {urdn: np.concatenate(dval) for urdn, dval in urdevt.items()}
    urdhk = {urdn:np.unique(np.concatenate(hklist)) for urdn, hklist in urdhk.items()}
    urddtc = {urdn: deadtime_correction(hk) for urdn, hk in urdhk.items()}

    tevts = np.sort(np.concatenate([np.concatenate(e) for e in urdbkge.values()]))
    tebkg, mgapsbkg, cratebkg, crerrbkg, bkgrate = make_overall_lc(tevts, urdgti, 25.)
    urdbkg = {urdn: constscale(urdbkgsc[urdn], bkgrate) for urdn in urdbkgsc}
    emap = make_expmap_for_wcs(locwcs, attdata, urdgti, phot_index=2., emin=4., emax=12.)
    ra, dec = zip(*[arttools.orientation.vec_to_pol((attdata(urdevt[urdn]["TIME"])*arttools.caldb.get_boresight_by_device(urdn)).apply(arttools._det_spatial.raw_xy_to_vec(urdevt[urdn]["RAW_X"], urdevt[urdn]["RAW_Y"]))) for urdn in urdevt])
    ra, dec = np.concatenate(ra), np.concatenate(dec)
    y, x = locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T.astype(np.int)
    cmap = np.zeros(emap.shape, np.int)
    u, uc = np.unique(np.array([x, y]), axis=1, return_counts=True)
    cmap[u[0], u[1]] = uc
    pickle.dump([emap, bmap, cmap, attdata, locgti, urdhk, urdgti, urdevt, tevts, urdbkge], open("/srg/a1/work/andrey/ART-XC/lp20_ddata.pickle", "wb"))
    """
    emap, bmap, cmap, attdata, locgti, urdhk, urdgti, urdevt, tevts, urdbkge = pickle.load(open("/srg/a1/work/andrey/ART-XC/lp20_ddata.pickle", "rb"))
    tevts = np.sort(np.concatenate([np.concatenate(e) for e in urdbkge.values()]))
    tebkg, mgapsbkg, cratebkg, crerrbkg, bkgrate = make_overall_lc(tevts, urdgti, 25.)


    #urdbkg = {urdn: constscale(urdbkgsc[urdn], bkgrate) for urdn in urdbkgsc}
    #bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
    #pickle.dump(bmap, open(

    urdns = arttools.telescope.URDNS

    xi0, yi0, qlist = zip(*[prepare_data(attdata, urdevt[urdn], urdn) for urdn in urdns])
    def filter_func(energy=None, grade=None):
        if not energy is None:
            return (energy > 4.) & (energy < 12.)
        if not grade is None:
            return (grade > -1) & (grade < 10)

    def bkgfilter_func(energy=None, grade=None):
        if not energy is None:
            return (energy > 40.) & (energy < 100.)
        if not grade is None:
            return (grade > -1) & (grade < 10)

    #bkgcalib = {urdn: prepare_calibration(urdn, filter_func, bkgfilter_func) for urdn in urdns}
    bkgcalib = {urdn: arttools.caldb.make_background_brightnes_profile(urdn, filter_func)/arttools.caldb.make_background_brightnes_profile(urdn, bkgfilter_func).sum() for urdn in urdns}
    bkgrates = np.concatenate([make_rate_map(bkgrate, bkgcalib[urdn], urdevt[urdn]["RAW_X"]-1, urdevt[urdn]["RAW_Y"] - 1, urdevt[urdn]["TIME"]) for urdn in urdns])
    x01 = np.concatenate(xi0)
    y01 = np.concatenate(yi0)
    qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

    coord = np.rad2deg(arttools.vector.vec_to_pol(qlist.apply([1, 0, 0])))
    x, y = locwcs.all_world2pix(coord.T, 1).T.astype(np.int)
    img = np.zeros((xsize, ysize), np.int)
    ij, c = np.unique(np.array([x, y]), axis=1, return_counts=True)
    img[ij[0], ij[1]] = c
    #plt.imshow(img)

    mask = np.all([x01 > -1, y01 > -1, x01 < 53, y01 < 53], axis=0)
    bkgrates, x01, y01, qlist = bkgrates[mask], x01[mask], y01[mask], qlist[mask]

    rmap0 = np.maximum(gaussian_filter(cmap.astype(np.double), 6.)*2.*pi*36*2, 1)/np.maximum(emap, 1.)
    emap = emap/7.
    rmap = np.copy(rmap0)
    ipsf = pickle.load(open("/srg/a1/work/andrey/ART-XC/PSF/invert_psf_v7_53pix.pickle", "rb"))
    ppsf = []
    m = emap > 0
    for it in range(20):
        ppsf.append(PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, default_rfun, mpnum=20))
        rmap[m] = np.maximum(ppsf[-1], 1)[m]/emap[m] #np.maximum(ppsf[-1][m]/emap[m], 0.)

    phrt = PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, zeroct_fun, mpnum=10)
    tcts = PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, total_counts, mpnum=10)
    nzcts = PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, nonzero_cts, mpnum=10)
    pickle.dump([emap, ppsf, tcts, nzcts, phrt],  open("/srg/a1/work/andrey/ART-XC/ppsf4_4.pickle", "wb"))
