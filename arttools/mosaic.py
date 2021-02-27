import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
#from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Process, Queue, RawArray, cpu_count
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock, current_thread
from .orientation import vec_to_pol, pol_to_vec, OPAX, make_quat_for_wcs, get_wcs_roll_for_qval
from ._det_spatial import offset_to_vec, vec_to_offset
from .psf import unpack_inverse_psf
from itertools import cycle
from math import sin, cos, sqrt, pi
import asyncio
import matplotlib.pyplot as plt
from copy import copy
from ctypes import c_bool
import sys


MPNUM = cpu_count()

def put_stright_on(vals, bkg, rmap):
    return vals

def get_source_photon_probability(core, bkg, rate):
    return rate*core/(bkg + rate*core)

def get_zerosource_photstat(core, bkg, rate):
    return np.log(bkg/(bkg + core*rate))


class SkyImage(object):
    """
    stores an image of the sky and allows to put specific core on the sky coordinates
    """
    def __init__(self, locwcs, vmap=None, shape=None, rfun=get_source_photon_probability):
        self.locwcs = locwcs
        self.shape = shape if not shape is None else [int(locwcs.wcs.crpix[1]*2 + 1), int(locwcs.wcs.crpix[0]*2 + 1)]
        self.img = np.zeros(self.shape, np.double)
        self.rfun = rfun
        self.lock = Lock()

        self.y, self.x = np.mgrid[1:self.img.shape[0] + 1:1, 1:self.img.shape[1] + 1:1]
        self.ra, self.dec = self.locwcs.all_pix2world(np.array([self.x.ravel(), self.y.ravel()]).T, 1).T
        self.vecs = pol_to_vec(*np.deg2rad([self.ra, self.dec])).reshape(list(self.img.shape) + [3, ])
        if not vmap is None:
            self._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        self.idx = np.arange(self.img.size).reshape(self.img.shape)

    def _set_corners(self, vals):
        self.corners = vals

    def _set_core(self, x, y, core):
        self.vmap = RegularGridInterpolator((x, y), core, bounds_error=False, fill_value=0.)
        self._set_corners(offset_to_vec(x[[0, 0, -1, -1]], y[[0, -1, -1, 0]]))

    def _update_interpolation_core_values(self, core):
        self.vmap.values = core

    def _get_quat_rectangle(self, qval):
        ra, dec = vec_to_pol(qval.apply(self.corners))
        x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T - 0.5
        #print("x", x)
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
        return il, ir, jl, jr


    def interpolate_vmap_for_qval(self, qval, norm, img=None):
        img = self.img if img is None else img
        ra, dec = vec_to_pol(qval.apply(self.corners))
        x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T - 0.5
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
        vecs = self.vecs[il:ir, jl:jr]
        xl, yl = vec_to_offset(qval.apply(vecs.reshape((-1, 3)), inverse=True))
        img[il:ir, jl:jr] += norm*self.vmap((xl, yl)).reshape((ir - il, jr - jl))

    def collector(self, qout):
        val = qout.get()
        self.lock.acquire()
        self.img = self.img + val
        self.lock.release()

    @staticmethod
    def worker(locwcs, shape, x, y, vmapvals, qin, qout):
        """
        note: no self in the multiprocessing processes since they will be sent to other process after pickling the overall instance of the class,
        which is several times heavier then actually required for iniitialization informaion
        """

        vmap = RegularGridInterpolator((x, y), np.copy(np.frombuffer(vmapvals).reshape(x.size, y.size)), bounds_error=False, fill_value=0.)
        sky = SkyImage(locwcs, vmap, shape)
        while True:
            val = qin.get()
            if val == -1:
                break
            quat, norm = val
            sky.interpolate_vmap_for_qval(qval, norm)
        qout.put(sky.image)

    def interpolate_mp(self, qvals, norms, mpnum=None):
        if mpnum is None:
            for i in range(norms.size):
                self.interpolate_vmap_for_qval(qvals[i], norms[i])
        else:
            vmapvals = RawArray(self.vmap.values.dtype.char, self.vmap.values.size)
            np.copyto(np.frombuffer(vmapvals).reshape(self.vmap.values.shape), self.vmap.values)
            lock = Lock()
            qin = Queue(100)
            qout = Queue()
            res = [self.img, ]
            collectors = [Thread(target=self.collector, args=(qout,)) for _ in range(mpnum)]
            for collector in collectors:
                collector.start()

            pool = [Process(target=self.worker, args = \
                            (self.locwcs, self.shape, self.vmap.grid[0], self.vmap.grid[1], vmapvals, qin, qout)) \
                            for _ in range(mpnum)]

            for worker in pool:
                worker.start()

            for i in range(norms.size):
                qin.put([qvals[i], norms[i]])
                sys.stderr.write('\rdone {0:%}'.format(i/(exptime.size - 1)))

            for worker in pool:
                qin.put(-1)

            for collector in collectors:
                collector.join()

    @staticmethod
    def init_imgs_buffer(cache, numpy_buffer):
        cache[current_thread()] = next(numpy_buffer)


    def interpolate(self, qvals, norms, mpnum=2):
        imgs = [np.zeros(self.img.shape, np.double) for _ in range(mpnum)]
        cache = {}
        pool = ThreadPool(mpnum, initializer=SkyImage.init_imgs_buffer,
                          initargs=(cache, iter(imgs)))
        pool.map(lambda args: self.interpolate_vmap_for_qval(*args, img=cache.get(current_thread())),
                 zip(qvals, norms))
        self.img += sum(imgs)

    @staticmethod
    def spread_events(img, locwcs, vecs, weights=None):
        xy = locwcs.all_world2pix(np.rad2deg(vec_to_pol(vecs)).T, 1) - 0.5
        xy = xy.astype(np.int)
        np.add.at(img, (xy[:, 1], xy[:, 0]), 1 if weights is None else weights)

    def convolve(self, qvals, norm, shape, dalpha=pi/180./10., img=None):
        img = self.img if img is None else img
        angles = get_wcs_roll_for_qval(self.locwcs, qvals)
        u, ii = np.unique((angles/dalpha).astype(np.int), return_inverse=True)
        for i, angle in enumerate(u*dalpha + dalpha/2.):
            idx = np.where(ii == i)[0]
            tmpimg = np.zeros(img.shape, np.double)
            q = make_quat_for_wcs(self.locwcs, self.img.shape[1]//2, self.img.shape[0]//2, angle)
            il, ir, jl, jr = self._get_quat_rectangle(q)
            self.interpolate_vmap_for_qval(q, 1, tmpimg)
            core = np.copy(tmpimg[il: ir, jl: jr])
            tmpimg[il: ir, jl: jr] = 0.
            self.spread_events(tmpimg, self.locwcs, qvals[idx].apply(OPAX), norm[idx])
            img += convolve(tmpimg, core, mode="same")


    def permute_with_rmap(self, qval, bkg, scale, vmap=None, img=None):
        vmap = self.vmap if vmap is None else vmap
        img = self.img if img is None else img
        ra, dec = vec_to_pol(qval.apply(self.corners))
        x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T - 0.5
        jl, jr = max(int(x.min()), 0), min(img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(img.shape[0], int(y.max()+1))
        m = self.mask[il:ir, jl: jr]
        vecs = self.vecs[il:ir, jl:jr][m]
        xl, yl = vec_to_offset(qval.apply(vecs, inverse=True))
        aval = self.rfun(vmap((xl, yl)), bkg, scale*self.rmap[il:ir, jl:jr][m])
        img[il:ir, jl:jr][m] += aval

    def permute_thread_pool_worker(self, args):
        sl, i, j = args
        img = self.imgbuffer.pop(0)
        vmap = self.vmapbuffer.pop(0)
        vmap.values = unpack_inverse_psf(i, j)
        for q, b, s in zip(self.qvals[sl], self.bkgs[sl], self.scales[sl]):
            self.permute_with_rmap(q, b, s, vmap, img)
        self.imgbuffer.append(img)
        self.vmapbuffer.append(vmap)

    def permute_thread(self, rmap, qvals, x01, y01, bkgs, scales, mask, mpnum=MPNUM):
        self.rmap = rmap
        self.mask = mask
        self.imgbuffer = [np.zeros(self.img.shape, np.double) for _ in range(mpnum)]
        self.vmapbuffer = [RegularGridInterpolator((self.vmap.grid[0], self.vmap.grid[1]), self.vmap.values, bounds_error=False, fill_value=0.0) for _ in range(mpnum)]
        pool = ThreadPool(mpnum)

        ijpairs, iidx, counts = np.unique(np.array([x01, y01]), axis=1, return_counts=True, return_inverse=True)
        isidx = np.argsort(iidx)
        ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
        self.bkgs = bkgs[isidx]
        self.scales = scales[isidx]
        self.qvals = qvals[isidx]
        slices = [slice(s, e) for s, e in zip(ii, ii + counts)]

        pool.map(self.permute_thread_pool_worker, zip(slices, ijpairs[0], ijpairs[1]))
        return np.sum(self.imgbuffer, axis=0)

    @staticmethod
    def permute_mp_worker(rmapv, maskv, shape, locwcs, vmapgrid, qin, qout, rfun):
        rmap = np.empty(shape, np.double)
        mask = np.empty(shape, np.bool)
        vmap = RegularGridInterpolator(vmapgrid, np.empty((vmapgrid[0].size, vmapgrid[1].size), np.double), fill_value=0., bounds_error=False)
        np.copyto(mask, np.frombuffer(maskv, np.bool).reshape(shape))
        rt = np.empty(mask.sum(), np.double)
        np.copyto(rt, np.frombuffer(rmapv))
        rmap[mask] = rt
        sky = SkyImage(locwcs, vmap, rfun=rfun)
        sky.rmap = rmap
        sky.mask = mask
        while True:
            vals = qin.get()
            if vals == -1:
                break
            i, j, quats, bkgs, scales = vals
            sky.vmap.values = unpack_inverse_psf(i, j)
            #print(i, j, bkgs.sum(), scales.sum(), sky.vmap.values.sum())
            for q, b, s in zip(quats, bkgs, scales):
                sky.permute_with_rmap(q, b, s)
        qout.put(sky.img)

    def permute_mp(self, rmap, qvals, x01, y01, bkgs, scales, mask, mpnum=MPNUM):
        self.img[:, :] = 0.
        maskv = RawArray(c_bool, mask.size)
        rmapv = RawArray(rmap.dtype.char, int(mask.sum()))
        np.copyto(np.frombuffer(rmapv), rmap[mask])
        np.copyto(np.frombuffer(maskv, np.bool).reshape(rmap.shape), mask)
        qin = Queue(100)
        qout = Queue()
        res = [self.img, ]
        collectors = [Thread(target=self.collector, args=(qout,)) for _ in range(mpnum)]
        for collector in collectors:
            collector.start()

        pool = [Process(target=self.permute_mp_worker, args = \
                        (rmapv, maskv, rmap.shape, self.locwcs, self.vmap.grid, qin, qout, self.rfun)) \
                        for _ in range(mpnum)]

        ijpairs, iidx, counts = np.unique(np.array([x01, y01]), axis=1, return_counts=True, return_inverse=True)
        isidx = np.argsort(iidx)
        ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
        bkgs = bkgs[isidx]
        scales = scales[isidx]
        qvals = qvals[isidx]
        b = [bkgs[s:e] for s, e in zip(ii, ii + counts)]
        s = [scales[s:e] for s, e in zip(ii, ii + counts)]
        q = [qvals[s:e] for s, e in zip(ii, ii + counts)]

        for worker in pool:
            worker.start()

        for i in range(counts.size):
            #print(ijpairs[0, i], ijpairs[1, i], b[i].sum(), s[i].sum())
            qin.put([ijpairs[0, i], ijpairs[1, i], q[i], b[i], s[i]])
            sys.stderr.write('\rdone {0:%}'.format(i/(counts.size - 1)))

        for worker in pool:
            qin.put(-1)

        for collector in collectors:
            collector.join()

        for worker in pool:
            worker.join()

        return np.copy(self.img)
