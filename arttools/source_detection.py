import numpy as np
import pickle
import arttools
import os
from scipy.signal import convolve
from scipy import ndimage
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import RegularGridInterpolator
from math import sin, cos, sqrt, pi
from arttools.time import get_gti, GTI, tGTI, emptyGTI, deadtime_correction
from arttools.atthist import hist_orientation_for_attdata
from arttools.planwcs import make_wcs_for_attdata
from arttools.caldb import get_energycal, get_shadowmask, get_energycal_by_urd, get_shadowmask_by_urd, urdbkgsc, OPAXOFFSET
from arttools.energy import get_events_energy
from arttools.telescope import URDNS
from arttools.orientation import get_photons_sky_coord, read_gyro_fits, read_bokz_fits, AttDATA, define_required_correction
from arttools.lightcurve import make_overall_lc, weigt_time_intervals
from arttools.vignetting import load_raw_wignetting_function
from arttools.plot import make_energies_flags_and_grades, make_sky_image
from astropy.io import fits
from math import pi, cos, sin
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

import asyncio
from aiopipe import aiopipe
import sys
import pyds9


PSFSCALE = 300 #arcsec
MPNUM=20

#es, eg = pickle.load(open("bkg_sampling.pickle", "rb"))
el = pickle.load(open("/srg/a1/work/andrey/ART-XC/gc/bkg_sampling2.pickle", "rb"))

def prepare_calibration(urdn, filter_func, bkgfilter_func):
    x = np.arange(1, 47)
    y = np.arange(1, 47)
    e = np.arange(4.5, 250., 1.)
    g = np.arange(0, 17)

    eidx = np.arange(e.size)[filter_func(energy=e)]
    gidx = np.arange(g.size)[filter_func(grade=g)]
    e1idx = np.arange(e.size)[bkgfilter_func(energy=e)]
    g1idx = np.arange(g.size)[bkgfilter_func(grade=g)]
    scale = el[urdn][:, :, eidx, gidx].sum()/el[urdn][:, :, e1idx, g1idx].sum()
    return el[urdn][:, :, eidx, gidx].sum(axis=[2, 3])*scale

def constscale(const, func):
    def newfunc(val):
        return func(val)*const
    return newfunc

def make_rate_map(bkglcfun, detbmap, x, y, t):
    return detbmap[x, y]*bkglcfun(t)

def make_ratemap(wcs, cmap, bmap, emap):
    """
    make zero step estimation of the sources rate
    """
    side = 1./60/wcs.wcs.cdelt[0]
    rates = np.maximum(gaussian_filter(cmap, side)*side*sqrt(2.*pi) - gaussian_filter(bmap, side)*side*sqrt(2.*pi), 0.01)/np.maximum(1, emap)
    return rates

def put_on_map(attdata, event, wcs, img, profile):
    '''
    assuming random rotatino of the img defined by wcs to the detectors coordinate frame
    puts an inverse PSF profile over the image on place of correspongind cmap
    '''
    qloc = attdata(event["TIME"])
    ra, dec = qloc.apply(arttools._det_spatial.urd_to_vec(event))
    xc, yc = wcs.all_world2pix(np.rad2deg([ra, dec]).T, 1.)
    x, y = wcs.all_world2pix(np.array([ra*180./pi, dec*180./pi]).T, 1.).T.astype(np.int)
    side = 212//wcs.wcs.cdelt[0]
    xslice = slice(max(x - side, 0), min(x + side + 1, img.shape[0] - 1))
    yslice = slice(max(y - side, 0), min(y + side + 1, img.shape[0] - 1))
    X, Y = np.mgrid[xslice.start: xslice.stop: 1, yslice.start : yslice.stop: 1]
    pixradecy = wcs.all_pix2world(np.array([X.ravel(), Y.ravel()]).T, 1)
    vecs = qloc.inv().apply(arttools.orientation.pol_to_vec(np.deg2rad(pixradec)))
    x, y = arttools._det_spatial.vec_to_offset(vecs)
    img[xslice, yslice] += profile(x, y)

def refine_ratemap(wcs, emap, elist, detbmap, rates):
    pass

def offset_to_qcorr(x, y):
    v1 = np.array([0., 1., 0.])
    if type(x) is np.ndarray:
        v1 = np.tile(v1, x.shape + (1,))
    else:
        x, y = np.array([x,]), np.array([y,])
    q1 = Rotation.from_rotvec(v1*np.arctan2(-(23.5 - y)*arttools._det_spatial.DL, arttools._det_spatial.F)[:, np.newaxis])
    v1 = np.roll(v1, 1, axis=-1)
    q2 = Rotation.from_rotvec(v1*np.arctan2(-(23.5 - x)*arttools._det_spatial.DL, arttools._det_spatial.F)[:, np.newaxis])
    v1 = np.roll(v1, 1, axis=-1)
    return q1*q2*Rotation.from_rotvec(v1*pi)

def prepare_data(attdata, elist, urdn):
    xi0 = int(arttools.caldb.OPAXOFFSET[urdn][0])
    yi0 = int(arttools.caldb.OPAXOFFSET[urdn][1])
    x = elist["RAW_X"] - xi0 + 26 #magic number  26 is from created invert PSF which computed for 53x53 pix matrix with center of detecting pixel in the center of  [26, 26] pixel
    y = elist["RAW_Y"] - yi0 + 26
    qlist = attdata(elist["TIME"])*arttools.caldb.get_boresight_by_device(urdn)*offset_to_qcorr(elist["RAW_X"], elist["RAW_Y"])
    return x, y, qlist


iPSFscale = pi/180.*5./60.
iPSFcorners = np.array([[1, -sin(iPSFscale), -sin(iPSFscale)],
                        [1, -sin(iPSFscale), sin(iPSFscale)],
                        [1, sin(iPSFscale), sin(-iPSFscale)],
                        [1, sin(iPSFscale), -sin(iPSFscale)]])


def default_rfun(core, bkg, rate):
    return core/(bkg + rate*core)

def rsq_fun(core, bkg, rate):
    return core/(bkg + core*rate)**2.

def zeroct_fun(core, bkg, rate):
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
        self.vecs = arttools.orientation.pol_to_vec(*np.deg2rad([self.ra, self.dec]))

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
                ra, dec = arttools.orientation.vec_to_pol(qval.apply(self.corners))
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



    def make_one_step(self, vmaps, qvals, x, y, bkgrtes):
        #plt.ion()
        #im = plt.imshow(self.img, norm=LogNorm())
        #plt.show(block=False)

        xvmap = np.tan(np.arange(-300, 301, 5)*pi/180/3600)*arttools._det_spatial.F
        yvmap = np.tan(np.arange(-300, 301, 5)*pi/180/3600)*arttools._det_spatial.F
        pairs = np.array([x, y]).T
        idxsort = np.argsort(x*53 + y)
        pairs = pairs[idxsort]
        qvalst = qvals[idxsort]
        u, ui, uc = np.unique(pairs, axis=0, return_counts=True, return_index=True)
        for i in range(ui.size):
            sl = slice(ui[i], ui[i] + uc[i])
            print(u[i], bkgrates[sl])
            qvals, vm, bkg = qvalst[sl], vmaps[u[i, 0], u[i, 1]], bkgrates[sl]
            vmap = RegularGridInterpolator((xvmap, yvmap), vm, bounds_error=False, fill_value=0.)

            for j in range(len(qvals)):
                qval = qvals[j]
                ra, dec = arttools.orientation.vec_to_pol(qval.apply(self.corners))
                x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T
                jl, jr = max(int(x.min()), 0), min(self.img.shape[1] - 1, int(x.max()+1))
                il, ir = max(int(y.min()), 0), min(self.img.shape[0] - 1, int(y.max()+1))

                #x, y = self.x[il:ir + 1, jl: jr + 1], self.y[il: ir + 1, jl: jr + 1]
                #ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T, 1).T

                vecs = np.copy(self.vecs[il:ir + 1, jl: jr + 1].reshape((-1, 3)))
                xl, yl = arttools._det_spatial.vec_to_offset(qval.apply(vecs, inverse=True))
                self.img[il:ir+1, jl:jr+1] += self.rfun(vmap((xl, yl)).reshape((ir + 1 - il, jr + 1 - jl)),
                                                bkg[j], self.rmap[il:ir+1, jl: jr + 1])

                #x, y = self.locwcs.all_world2pix(np.rad2deg(arttools.orientation.vec_to_pol(qval.apply([1, 0, 0]))).T.reshape((1, 2)), 1).astype(np.int).T
                #self.img[x, y] += 1
                #im.set_data(self.img)
            plt.imshow(self.img, norm=LogNorm(vmin=1e-6, vmax=0.12251))
            plt.show()
            #plt.draw()



    @classmethod
    def make_mp(cls, locwcs, vmaps, qvals, x, y, bkgrates, rmap, rfun, *args, mpnum=MPNUM, **kwargs):
        pairs = np.array([x, y]).T
        idxsort = np.argsort(x*53 + y)
        pairs = pairs[idxsort]
        qvals = qvals[idxsort]
        u, ui, uc = np.unique(pairs, axis=0, return_counts=True, return_index=True)
        qin = Queue(100)
        qout = Queue(2)
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
    ax = arttools.orientation.pol_to_vec(*np.deg2rad([278.38688, -10.571942]))
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
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
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

    coord = np.rad2deg(arttools.orientation.vec_to_pol(qlist.apply([1, 0, 0])))
    x, y = locwcs.all_world2pix(coord.T, 1).T.astype(np.int)
    img = np.zeros((xsize, ysize), np.int)
    ij, c = np.unique(np.array([x, y]), axis=1, return_counts=True)
    img[ij[0], ij[1]] = c
    plt.imshow(img)




    mask = np.all([x01 > -1, y01 > -1, x01 < 53, y01 < 53], axis=0)
    bkgrates, x01, y01, qlist = bkgrates[mask], x01[mask], y01[mask], qlist[mask]

    rmap = np.maximum(gaussian_filter(cmap, 3.) - gaussian_filter(bmap, 3.), 0)/np.maximum(gaussian_filter(emap, 3.), 1.)
    ipsf = pickle.load(open("/srg/a1/work/andrey/ART-XC/PSF/invert_psf_v7_53pix.pickle", "rb"))
    ppsf = []
    m = emap > 0
    for it in range(20):
        ppsf.append(PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, default_rfun, mpnum=20))
        rmap[m] = ppsf[-1][m]/emap[m]

    phrt = PutiPSF.make_mp(locwcs, ipsf, qlist, x01, y01, bkgrates, rmap, zeroct_fun, mpnum=10)
    pickle.dump([emap, ppsf, phrt],  open("/srg/a1/work/andrey/ART-XC/ppsf4_2.pickle", "wb"))
