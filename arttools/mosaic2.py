import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Process, Queue, RawArray, cpu_count, Barrier, current_process
from threading import Thread
from multiprocessing.pool import ThreadPool
import tqdm
from .vector import vec_to_pol, pol_to_vec
from .orientation import OPAX, ra_dec_roll_to_quat
from .planwcs import get_wcs_roll_for_qval, make_quat_for_wcs, wcs_roll
from ._det_spatial import offset_to_vec, vec_to_offset, vec_to_offset_pairs
from .sphere import ConvexHullonSphere, PRIME_NUMBERS
from .psf import unpack_inverse_psf, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, ayutee
from .aux import DistributedObj
from math import sin, cos, sqrt, pi, log
from ctypes import c_bool
import pickle

MPNUM = cpu_count()//4

def put_stright_on(core, scale, rmap):
    """
    return convolve core multiplied by scale faactor
    """
    return core*scale

def get_source_photon_probability(core, scale, rate):
    """
    core (is expected to be inversed psf) is multiplied by the current ratemap, individual energy and grade based photon probability and background rate to produce probability of individual event to be relevant to the  source at corresponding possiotion
    here scale is equal to the photon vs background probability devided over pixel background rate  P(Photon vs background | E & G) * rate/bkgrate; where rate is provided from the rates map
    """
    return scale*rate*core/(1. + scale*rate*core)

def get_zerosource_photstat(core, scale, rate):
    """
    produce a probability
    """
    return -np.log(1. + scale*core*rate)

def surface_brightness_mixtues(core, scale, rate):
    return rate*scale*core/(1. + scale*np.sum(core*rate))

def get_split_side(val, guess=None):
    guess = int(sqrt(val)) if guess is None else guess
    while guess*(val//guess) != val:
        guess = guess - 1
    return guess


class MockPool(object):
    _processes = 1
    @staticmethod
    def map(func, args):
        return list(map(func, args))

    @staticmethod
    def imap(func, args):
        return list(map(func, args))

    @staticmethod
    def imap_unordered(func, args):
        return (func(a) for a in args)


class SkyImage(DistributedObj):

    @staticmethod
    def spread_events(img, locwcs, vecs, weights=None, shape=None):
        xy = (locwcs.all_world2pix(np.rad2deg(vec_to_pol(vecs)).T, 1) - 0.5).astype(int) - ([0, 0] if shape is None else [shape[1][0], shape[0][0]])
        np.add.at(img, (xy[:, 1], xy[:, 0]), 1 if weights is None else weights)

    """
    stores an image of the sky and allows to put specific core on the sky coordinates
    """
    def __init__(self, locwcs, vmap=None, shape=None, mpnum=4, barrier=None):
        kwargs = locals()
        kwargs.pop("self")
        kwargs.pop("__class__")
        initpool = Thread(target=super().__init__, kwargs=kwargs) #don't wait for pool initialization since it takes approximately the same time
        initpool.start()
        #super().__init__(**kwargs)

        self.locwcs = locwcs
        self.shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
        self.img = np.zeros(np.diff(self.shape, axis=1).ravel(), np.double)
        self.mask = np.ones(self.img.shape, np.bool)
        self.rmap = np.zeros(self.img.shape, np.double)
        y, x = np.mgrid[self.shape[0][0] + 1:self.shape[0][1] + 1:1, self.shape[1][0] + 1: self.shape[1][1] + 1:1]
        ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T, 1).T
        self.vecs = pol_to_vec(*np.deg2rad([ra, dec])).reshape(list(self.img.shape) + [3, ])
        self.action = put_stright_on #set_action(put_stright_on)
        self.corners = None

        if not vmap is None:
            self.set_core(vmap.grid[0], vmap.grid[1], vmap.values)

        initpool.join()

    def _set_corners(self, vals):
        """
        expect to receive four vectors at the corners of interpolation map
        """
        self.corners = ConvexHullonSphere(vals)
        self.corners = self.corners.expand(2./3660.*pi/180.)


    def set_core(self, x, y, core):
        self.vmap = RegularGridInterpolator((x, y), core, bounds_error=False, fill_value=0.)
        self._set_corners(offset_to_vec(x[[0, 0, -1, -1]], y[[0, -1, -1, 0]]))


    @DistributedObj.for_each_process
    def set_vmap(self, vmap):
        self.set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        list(self.for_each_process("set_core", vmap.grid[0], vmap.grid[1], vmap.values))


    @DistributedObj.for_each_process
    def update_interpolation_core_values(self, core):
        self.vmap.values = core


    @DistributedObj.for_each_process
    def clean_image(self):
        self.img[:, :] = 0.

    def _get_vmap_edges(self, qval):
        radec = vec_to_pol(qval.apply(self.corners.corners))
        return (self.locwcs.all_world2pix(np.rad2deg(radec).T, 1).T - 0.5).astype(np.int)

    def _get_quat_rectangle(self, qval):
        x, y = self._get_vmap_edges(qval)
        x = x - self.shape[1][0]
        y = y - self.shape[0][0]
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
        return il, ir, jl, jr

    @DistributedObj.for_each_process
    def set_actions(self, func):
        self.action = func

    @DistributedObj.for_each_process
    def get_img(self):
        return self.img

    def accumulate_img(self):
        self.img += sum(self.get_img)

    def set_mask(self, mask):
        self.mask[:, :] = mask[:, :]

    def set_rmap(self, rmap):
        self.rmap[:, :] = rmap[:, :]

    def interpolate_vmap_for_qval(self, qval, scale, img=None, rmap=None, mask=None):
        if rmap is None:
            rmap = self.rmap
        if img is None:
            img = self.img
        if mask is None:
            mask = self.mask

        il, ir, jl, jr = self._get_quat_rectangle(qval)
        vecs = self.vecs[il:ir, jl:jr][mask[il:ir, jl:jr]]
        xyl = vec_to_offset_pairs(qval.apply(vecs, inverse=True))
        img[il:ir, jl:jr][mask[il:ir, jl:jr]] += self.action(self.vmap(xyl), scale, rmap[il:ir, jl:jr][mask[il:ir, jl:jr]])


    def interpolate_vmap_for_qvals(self, qvals, scales, vmap):
        self.vmap.values = vmap
        for q, s in zip(qvals, scales):
            self.interpolate_vmap_for_qval(q, s)

    def accumulate_img(self):
        self.img += sum(self.for_each_process("get_img"))

    def cores_for_rolls(self, rolls):
        ra, dec = self.locwcs.wcs.crval
        vcentral = pol_to_vec(*np.deg2rad([ra, dec]).reshape((2, -1)))

        q0 = ra_dec_roll_to_quat(*np.array([ra, dec, 0.]).reshape((3, 1)))[0]

        lroll = rolls - np.repeat(wcs_roll(self.locwcs, Rotation(q0.as_quat().reshape((-1, 4)))), np.asarray(rolls).size)
        q0 = ra_dec_roll_to_quat(*np.array([np.full(lroll.size, ra, float), np.full(lroll.size, dec, float), np.rad2deg(lroll)]))
        x, y = np.concatenate([self._get_vmap_edges(q) for q in q0], axis=1)

        xsize = int(np.max(np.abs(x - self.locwcs.wcs.crpix[1])) + 0.5)
        ysize = int(np.max(np.abs(y - self.locwcs.wcs.crpix[0])) + 0.5)
        shape = np.asarray(self.locwcs.wcs.crpix, int)[::-1] + [[-xsize, xsize +1], [-ysize, ysize + 1]]
        lsky = self.__class__(self.locwcs, shape=shape, vmap=self.vmap, mpnum=1, barrier=False)

        x, y = np.mgrid[-0.45:0.46:0.1, -0.45:0.46:0.1]
        ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T + self.locwcs.wcs.crpix, 1).T
        for r in lroll:
            q = ra_dec_roll_to_quat(ra, dec, np.full(ra.size, r*180/pi, float))
            lsky.clean_image()
            for qv in q:
                lsky.interpolate_vmap_for_qval(qv, 1./len(q))
            yield np.copy(lsky.img)

    def convolve_with_core(self, core, qvals, scales):
        tmpimg = np.zeros((self.img.shape), double)
        radec = np.rad2deg(vec_to_pol(qloc.apply([1, 0, 0])))
        xy = (self.locwcs.all_world2pix(radec.T, 1) - 0.5).astype(int)
        np.add.at(tmpimg, (xy[:, 1], xy[:, 0]), sloc)
        self.img += convolve(tmpimg, lsky.img, mode="same")

    def fft_convolve(self, qvals, scales):
        list(self.for_each_process("clean_image"))
        rolls = wcs_roll(self.locwcs, qvals)
        urolls, uiidx = np.unique((rolls*180./pi*2).astype(int), return_inverse=True)
        list(self.for_each_argument("convolve_with_core", zip(cores_for_rolls((urolls + 0.5)/360.*pi), (qvals[k == uiidx] for k in range(urolls.size)), (scales[k == uiidx] for k in range(urolls.size)))))
        self.accumulate_img()

    def direct_convolve(self, qvals, scales):
        list(self.for_each_process("clean_image"))
        list(self.for_each_argument("interpolate_vmap_for_qval", zip(qvals, scales), total=len(qvals)))
        self.accumulate_img()

    def rmap_convolve_multicore(self, tasks, total, size=None):
        list(self.for_each_process("clean_image"))
        list(self.for_each_argument("interpolate_vmap_for_qvals", tasks, total=total))
        self.accumulate_img()
