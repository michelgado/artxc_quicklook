import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max
#from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Process, Queue, RawArray, cpu_count, Barrier
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock, current_thread
import tqdm
from .orientation import vec_to_pol, pol_to_vec, OPAX, make_quat_for_wcs, get_wcs_roll_for_qval
from ._det_spatial import offset_to_vec, vec_to_offset
from .planwcs import ConvexHullonSphere, PRIME_NUMBERS
from .psf import unpack_inverse_psf, unpack_inverse_psf_ayut, get_ipsf_interpolation_func, ayutee
from itertools import cycle, repeat
from math import sin, cos, sqrt, pi, log
import asyncio
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from ctypes import c_bool
import sys
import asyncio

import time


MPNUM = cpu_count()//4

def put_stright_on(vals, bkg, rmap):
    return vals

def get_source_photon_probability(core, bkg, rate):
    return rate*core/(bkg + rate*core)

def get_zerosource_photstat(core, bkg, rate):
    return np.log(bkg/(bkg + core*rate))

def get_split_side(val, guess=None):
    guess = int(sqrt(val)) if guess is None else guess
    while guess*(val//guess) != val:
        guess = guess - 1
    return guess

class MockPool(object):
    @staticmethod
    def map(func, args):
        return list(map(func, args))

    @staticmethod
    def imap(func, args):
        return list(map(func, args))


class SkyImage(object):
    """
    stores an image of the sky and allows to put specific core on the sky coordinates
    """
    def __init__(self, locwcs, vmap=None, shape=None):
        self.locwcs = locwcs
        self.shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
        self.img = np.zeros(np.diff(self.shape, axis=1).ravel(), np.double)
        self.rmap = np.zeros(self.img.shape, np.double)
        self.mask = np.ones(self.img.shape, np.bool)
        self.idx = np.arange(self.img.size).reshape(self.img.shape)
        self.lock = Lock()

        y, x = np.mgrid[self.shape[0][0] + 1:self.shape[0][1] + 1:1, self.shape[1][0] + 1: self.shape[1][1] + 1:1]
        ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T, 1).T
        self.vecs = pol_to_vec(*np.deg2rad([ra, dec])).reshape(list(self.img.shape) + [3, ])
        if not vmap is None:
            self._set_core(vmap.grid[0], vmap.grid[1], vmap.values)

    def _set_corners(self, vals):
        """
        expect to receive four vectors at the corners of interpolation map
        """
        self.corners = ConvexHullonSphere(vals)
        self.corners = self.corners.expand(2./3660.*pi/180.)

    def _set_core(self, x, y, core):
        self.vmap = RegularGridInterpolator((x, y), core, bounds_error=False, fill_value=0.)
        self._set_corners(offset_to_vec(x[[0, 0, -1, -1]], y[[0, -1, -1, 0]]))

    def _clean_image(self):
        self.img[:, :] = 0.

    def _update_interpolation_core_values(self, core):
        self.vmap.values = core

    def _get_quat_rectangle(self, qval):
        ra, dec = vec_to_pol(qval.apply(self.corners.corners))
        x, y = (self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T - 0.5).astype(np.int)
        x = x - self.shape[1][0]
        y = y - self.shape[0][0]
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
        return il, ir, jl, jr

    def _get_quats_rectangles(self, qvals):
        radec = np.array([vec_to_pol(qvals.apply(v)) for v in self.corners.corners])
        xy = (self.locwcs.all_world2pix(radec.reshape((-1, 2)), 1) - 0.5).astype(int).reshape(radec.shape)
        il = np.maximum(np.min(xy[..., 0], axis=0), 0)
        jl = np.maximum(np.min(xy[..., 1], axis=0), 0)
        iu = np.minimum(np.max(xy[..., 0], axis=0) + 1, self.shape[0])
        ju = np.minimum(np.max(xy[..., 1], axis=0) + 1, self.shape[1])
        return il, iu, jl, ju


    def interpolate_vmap_for_qval(self, qval, norm, img):
        il, ir, jl, jr = self._get_quat_rectangle(qval)
        vecs = self.vecs[il:ir, jl:jr]
        xl, yl = vec_to_offset(qval.apply(vecs.reshape((-1, 3)), inverse=True))
        img[il:ir, jl:jr] += norm*self.vmap((xl, yl)).reshape((ir - il, jr - jl))

    def collector(self, qout):
        val = qout.get()
        self.lock.acquire()
        self.img += val
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
            sky.interpolate_vmap_for_qval(quat, norm, sky.img)
        qout.put(sky.img)

    def interpolate_mp(self, qvals, norms, mpnum=None):
        if mpnum is None:
            for i in range(norms.size):
                self.interpolate_vmap_for_qval(qvals[i], norms[i], self.img)
        else:
            vmapvals = RawArray(self.vmap.values.dtype.char, self.vmap.values.size)
            np.copyto(np.frombuffer(vmapvals).reshape(self.vmap.values.shape), self.vmap.values)
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
                sys.stderr.write('\rdone {0:%}'.format((i + 1.)/norms.size))

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



    def interpolate_bunch(self, qvals, norm, vmapvals=None, img=None):
        if not vmapvals is None:
            self.vmap.values = vmapvals
        if img is None: img = self.img
        """
        for q, n in zip(qvals, norm):
            self.interpolate_vmap_for_qval(q, n, img)
        """
        il, ir, jl, jr = self._get_quat_rectangle(qvals)
        #s = (ir - il - 1)*(jr - jl - 1)
        """
        sx = ir - il
        js = np.cumsum(np.repeat(jr - jl, ir - il))
        a = np.arange(s.sum())

        a[js[0]:] -= np.repeat(s[np.cumsum(js[:-1])], js[1:])
        b = np.arange(np.sum(sx))
        b[sx[0]:] -= np.repeat(b[sx[:-1]], sx[1:])
        idx = a + np.repeat(il*img.shape[0] + jl, s) + np.repeat(b*img.shpae[0], np.repeat(jr - jl, sx))
        #mx = np.max(ir - il)
        #my = np.max(jr - jl)
        """
        idx = np.concatenate([self.idx[xl:xr, xl, xr].ravel() for xl, xr, yl, yz in zip([il ,ir, jl, jr])])
        x, y = vec_to_offset(Rotation(np.repeat(q.as_quat(), (ir - il)*(jr - jl))).apply(self.vecs.reshape((-1, 3))[idx], inverse=True))
        m = np.all([x > self.vmap.grid[0][0], x < self.vmap.grid[0][-1], y > self.vmap.grid[1][0], y < self.vmap.grid[0][-1]], axis=0)
        np.add.at(img.ravel(), idx[m], self.vmap(np.array([x, y]).T[m])*np.repeat(norm, s)[m])

    @staticmethod
    def spread_events(img, locwcs, vecs, weights=None):
        xy = locwcs.all_world2pix(np.rad2deg(vec_to_pol(vecs)).T, 1) - 0.5
        xy = xy.astype(np.int)
        np.add.at(img, (xy[:, 1], xy[:, 0]), 1 if weights is None else weights)

    def convolve_worker(self, args):
        angle, qvals, norms = args
        tmpimg = np.zeros(self.img.shape, np.double)
        q = make_quat_for_wcs(self.locwcs, self.img.shape[1]//2 + 1, self.img.shape[0]//2 + 1, angle)
        il, ir, jl, jr = self._get_quat_rectangle(q)
        self.interpolate_vmap_for_qval(q, 1, tmpimg)
        xsize = max(self.img.shape[0]//2 - il, ir - self.img.shape[0]//2 - 1)
        ysize = max(self.img.shape[1]//2 - jl, jr - self.img.shape[1]//2 - 1)
        core = np.copy(tmpimg[tmpimg.shape[0]//2 - xsize: tmpimg.shape[0]//2 + xsize + 1,
                              tmpimg.shape[1]//2 - ysize: tmpimg.shape[1]//2 + ysize + 1])

        tmpimg[il: ir, jl: jr] = 0.
        self.spread_events(tmpimg, self.locwcs, qvals.apply(OPAX), norms)
        return convolve(tmpimg, core, mode="same")

    def convolve(self, qvals, norm, dalpha=pi/180./2., img=None, mpnum=MPNUM):
        img = self.img if img is None else img
        angles = get_wcs_roll_for_qval(self.locwcs, qvals)
        u, ii = np.unique((angles/dalpha).astype(np.int), return_inverse=True)
        pool = ThreadPool(mpnum)
        idxs = pool.map(lambda i: np.where(ii == i)[0], range(u.size))
        #for a, idx in zip(u*dalpha + dalpha/2., idxs):
        #    self.img[:, :] += self.convolve_worker((a, qvals[idx], norm[idx]))
        for tmpimg in pool.imap(self.convolve_worker, zip(u*dalpha + dalpha/2., [qvals[idx] for idx in idxs], [norm[idx] for idx in idxs])):
            self.img[:, :] += tmpimg

    def permute_with_rmap(self, qval, bkg, scale, rfun=None, vmap=None, img=None):
        vmap = self.vmap if vmap is None else vmap
        img = self.img if img is None else img
        rfun = self.rfun if rfun is None else rfun
        il, ir, jl, jr = self._get_quat_rectangle(qval)
        if ir - il > 0 and jr - jl > 0:
            mt = self.mask[il:ir, jl:jr]
            md = self.detm[il:ir, jl:jr]
            if np.any(md):
                mt = mt | md
                ml = md[mt]
                vecs = self.vecs[il:ir, jl:jr][mt]
                xl, yl = vec_to_offset(qval.apply(vecs, inverse=True))
                core = vmap((xl, yl))
                rl = self.rmap[il:ir, jl:jr][mt]*scale
                bkg = bkg + np.sum(core[ml]*rl[ml]) - core*ml*rl
            else:
                vecs = self.vecs[il:ir, jl:jr][mt]
                xl, yl = vec_to_offset(qval.apply(vecs, inverse=True))
                core = vmap((xl, yl))
                rl = self.rmap[il:ir, jl:jr][mt]*scale
            aval = rfun(core, bkg, rl)
            img[il:ir, jl:jr][mt] += aval

    def update_rates_for_qvals(self, bkgs, scales, qvals, core):
        self.vmap.core = core
        lidx = list(map(self.get_arr_idx_slice, zip(qvals, cycle([idx,]), cycle([mask,]))))
        csize = np.array([l.size for l in lidx])
        lvecs = np.concatenate(list(map(lambda args: args[0].apply(args[1], inverse=True), zip(qvals[sl], [rvecs[l] for l in lidx]))))
        lidx = np.concatenate(lidx)
        core = self.vmap(vec_to_offset(lvecs))
        np.add.at(self.img.ravel(), lidx, rfun(core, np.repeat(bkgs[sl], csize), np.repeat(scales[sl], csize)*rmap.ravel()[lidx]))

    def permute_thread_pool_worker(self, args):
        sl, i, j = args
        img = self.imgbuffer.pop(0)
        vmap = self.vmapbuffer.pop(0)
        vmap.values = unpack_inverse_psf(i, j)
        for q, b, s in zip(self.qvals[sl], self.bkgs[sl], self.scales[sl]):
            self.permute_with_rmap(q, b, s, get_source_photon_probability, vmap, img)
        self.imgbuffer.append(img)
        self.vmapbuffer.append(vmap)

    def get_arr_idx_slice(self, args):
        q, idx, mask = args
        il, ir, jl, jr = self._get_quat_rectangle(q)
        return idx[il: ir, jl: jr][mask[il: ir, jl: jr]]

    def permute_banch(self, rfun, rmap, mask, qvals, x01, y01, bkgs, scales, energy=None):
        if energy is None:
            ijpairs, iidx, counts = np.unique(np.array([x01, y01]), axis=1, return_counts=True, return_inverse=True)
        else:
            eidx = np.searchsorted(ayutee, energy) - 1
            ijpairs, iidx, counts = np.unique(np.array([x01, y01, eidx]), axis=1, return_counts=True, return_inverse=True)
        isidx = np.argsort(iidx)
        ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
        idx = np.arange(self.img.size).reshape(self.img.shape)
        slices = [slice(s, e) for s, e in zip(ii, ii + counts)]
        rvecs = self.vecs.reshape((-1, 3))
        for k, sl in enumerate(slices):
            if energy is None:
                i, j = ijpairs[:, k]
                self.vmap.values = unpack_inverse_psf(i, j)
            else:
                i, j, ke = ijpairs[:, k]
                self.vmap.values = unpack_inverse_psf_ayut(i, j)[ke]
            #an attempt to boost this part
            lidx = list(map(self.get_arr_idx_slice, zip(qvals[sl], cycle([idx,]), cycle([mask,]))))
            csize = np.array([l.size for l in lidx])
            lvecs = np.concatenate(list(map(lambda args: args[0].apply(args[1], inverse=True), zip(qvals[sl], [rvecs[l] for l in lidx]))))
            lidx = np.concatenate(lidx)
            core = self.vmap(vec_to_offset(lvecs))
            np.add.at(self.img.ravel(), lidx, rfun(core, np.repeat(bkgs[sl], csize), np.repeat(scales[sl], csize)*rmap.ravel()[lidx]))

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

    def sender(self, qin, qout):
        while True:
            self.qin.get()
            qout.put(np.copy(self.img[mask]))

    @staticmethod
    def permute_mp_worker(rmapv, maskv, maskd, shape, locwcs, vmapgrid, qin, qout, rfun):
        rmap = np.empty(shape, np.double)
        mask = np.empty(shape, np.bool)
        detm = np.empty(shape, np.bool)
        vmap = RegularGridInterpolator(vmapgrid, np.empty((vmapgrid[0].size, vmapgrid[1].size), np.double), fill_value=0., bounds_error=False)
        np.copyto(mask, np.frombuffer(maskv, np.bool).reshape(shape))
        np.copyto(detm, np.frombuffer(maskd, np.bool).reshape(shape))
        rt = np.empty(mask.sum(), np.double)
        np.copyto(rt, np.frombuffer(rmapv))
        rmap[mask] = rt
        sky = SkyImage(locwcs, vmap)
        sky.rfun = rfun
        sky.rmap = rmap
        sky.mask = mask
        sky.detm = detm
        while True:
            vals = qin.get()
            if vals == -1:
                break
            i, j, ek, quats, bkgs, scales = vals
            if ek is None:
                sky.vmap.values = unpack_inverse_psf(i, j)
            else:
                sky.vmap.values = unpack_inverse_psf_ayut(i, j)[ek]
            for q, b, s in zip(quats, bkgs, scales):
                sky.permute_with_rmap(q, b, s, rfun)
        qout.put(np.copy(sky.img[mask]))

    def permute_mp(self, rmap, mask, qvals, x01, y01, bkgs, scales, mdet=None, energy=None, rfun=get_source_photon_probability, mpnum=MPNUM):
        self.img[:, :] = 0.
        maskv = RawArray(c_bool, mask.size)
        maskd = RawArray(c_bool, mask.size)
        rmapv = RawArray(rmap.dtype.char, int(mask.sum()))
        np.copyto(np.frombuffer(rmapv), rmap[mask])
        np.copyto(np.frombuffer(maskv, np.bool).reshape(rmap.shape), mask)
        np.copyto(np.frombuffer(maskd, np.bool).reshape(rmap.shape), np.zeros(mask.shape, np.bool) if mdet is None else mdet)
        qin = Queue(100)
        qout = Queue()
        img = np.copy(self.img)
        self.img = img[mask]
        collectors = [Thread(target=self.collector, args=(qout,)) for _ in range(mpnum)]
        for collector in collectors:
            collector.start()

        pool = [Process(target=self.permute_mp_worker, args = \
                        (rmapv, maskv, maskd, rmap.shape, self.locwcs, self.vmap.grid, qin, qout, rfun)) \
                        for _ in range(mpnum)]


        if energy is None:
            ijpairs, iidx, counts = np.unique(np.array([x01, y01]), axis=1, return_counts=True, return_inverse=True)
        else:
            eidx = np.searchsorted(ayutee, energy) - 1
            ijpairs, iidx, counts = np.unique(np.array([x01, y01, eidx]), axis=1, return_counts=True, return_inverse=True)
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
            if energy is None:
                qin.put([ijpairs[0, i], ijpairs[1, i], None, q[i], b[i], s[i]])
            else:
                qin.put([ijpairs[0, i], ijpairs[1, i], ijpairs[2, i], q[i], b[i], s[i]])
            sys.stderr.write('\rdone {0:%}'.format((i + 1.)/counts.size))

        for worker in pool:
            qin.put(-1)

        for collector in collectors:
            collector.join()
        for worker in pool:
            worker.join()
        img[mask] = self.img
        self.img = img

        return self.img

    def permute_ratemap_until_convergence(self, emap, qvals, x01, y01, bkgs, scales, energy=None, rmap=None, mdet=None):
        self.img[:, :] = 0.
        qmask = np.ones(len(qvals), np.bool)
        pix = (self.locwcs.all_world2pix(np.rad2deg(vec_to_pol(qvals.apply([1, 0, 0]))).T, 1) - 0.5).astype(np.int)
        cmap = np.zeros(self.img.shape)
        u, uc = np.unique((pix[:, [1, 0]] - [self.shape[0][0], self.shape[1][0]]).T, axis=1, return_counts=True)
        cmap[u[0], u[1]] = uc
        minit = emap > 0.1 #np.median(emap[emap > 0.])*0.001
        mask = np.copy(minit)
        cinit = gaussian_filter(cmap, (30./3600.)/self.locwcs.wcs.cdelt[0])*2.*pi*((30./3600.)/self.locwcs.wcs.cdelt[0])**2.
        if rmap is None:
            rmap = np.ones(self.img.shape, np.double)
            rmap[mask] = cinit[mask]/emap[mask]
        convsize = max(int(sqrt(self.corners.area)/self.locwcs.wcs.cdelt[0] + 2), 1)
        convsize = convsize + 1 - convsize%2
        print("locwcs convolutions size", convsize)
        """
        import pyds9
        from astropy.io import fits
        import time
        ds9 = pyds9.DS9("test")
        ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=cinit, header=self.locwcs.to_header())]))
        ds9.set("frame new")
        ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=emap, header=self.locwcs.to_header())]))
        ds9.set("frame new")
        ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=mask.astype(np.int), header=self.locwcs.to_header())]))
        ds9.set("frame new")
        """
        ctot = np.zeros(self.img.shape, np.double)
        ctot[mask] = cinit[mask]
        """
        ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=ctot, header=self.locwcs.to_header())]))
        ds9.set("frame new")
        """
        for k in range(51):
            print("iteration %d, compute %f" % (k, mask.sum()/mask.size))
            self.permute_mp(rmap, mask, qvals[qmask], x01[qmask], y01[qmask], bkgs[qmask], scales[qmask], mdet, energy=None if energy is None else energy[qmask])
            print(self.img.sum(), self.img[mask].sum())
            ctot[mask] = self.img[mask]
            """
            ds9.set("frame 2")
            ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=mask.astype(np.int), header=self.locwcs.to_header())]))
            ds9.set("frame 3")
            ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=ctot, header=self.locwcs.to_header())]))
            """
            conv = (ctot - rmap*emap)/np.maximum(ctot, 0.5)

            conv[~mask] = 0.
            print("\nconvergence: ", np.histogram(np.abs(conv), np.array([0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 10.])))
            mup = (np.abs(conv[mask]) > 0.005)
            rmap[mask] = ctot[mask]/emap[mask]
            mask[mask] = mup
            """
            ds9.set("frame 4")
            ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=conv, header=self.locwcs.to_header())]))
            time.sleep(15)
            """
            if not np.any(mask):
                break
            #qm = convolve(mask, np.ones([convsize, convsize], np.double), mode="same") > 0
            #qmask = qm[pix[:, 1], pix[:, 0]]
            print("qmask size", qmask.sum(), qmask.size)
        self.img = ctot
        return self.img

    def permute_worker_2(self, ):
        allevts = qvals.apply(v)
        mloc = self.corners.check_inside_polygon(allevts)
        qloc, xloc, yloc, bloc, sloc = self.qvals[mask]

    @classmethod
    def permute_mp_2_worker(cls, args):
        shape, rfun, rmap, emap, mask, qvals, x01, y01, bkgs, scales, locwcs, energy = args
        print(shape, x01.size)
        vmap = get_ipsf_interpolation_func()
        sky = cls(locwcs, vmap, shape)
        ctot = np.zeros(sky.img.shape, np.double)
        rmap = np.ones(ctot.shape, np.double)
        for i in tqdm.tqdm(range(50)):
            sky.img[:, :] = 0.
            sky.permute_banch(rfun, rmap, mask, qvals, x01, y01, bkgs, scales, energy)
            ctot[mask] = sky.img[mask]
            rmap[mask], mask[mask] = sky.img[mask]/emap[mask], np.abs(sky.img[mask] - emap[mask]*rmap[mask]) > np.maximum(sky.img[mask], 1.)*0.01
            if not np.any(mask):
                break
        return sky.shape, ctot

    def permute_mp_2(self, rmap, emap, mask, qvals, x01, y01, bkgs, scales, energy=None, rfun=get_source_photon_probability, mpnum=MPNUM):
        self.img[:, :] = 0.
        imgarea = self.img.shape[0]*self.locwcs.wcs.cdelt[0]*self.img.shape[1]*self.locwcs.wcs.cdelt[1]
        snum = min(10*mpnum, int(imgarea*9)) #curreent inverse psf core has 10 arcmin side, so area 1/9 sq.arcdeg will double the amount of events sent to processes (smaller area will increase that number)
        while snum in PRIME_NUMBERS:
            snum += 1
        smallside = get_split_side(snum)
        bigside = snum//smallside
        if self.img.shape[0] < self.img.shape[1]: smallside, bigside = bigside, smallside

        x = np.linspace(0., self.img.shape[0], smallside + 1).astype(np.int)
        y = np.linspace(0., self.img.shape[1], bigside + 1).astype(np.int)
        grid = []
        shapes = []
        rslice = []
        eslice = []
        mslice = []
        expandsize = sqrt(self.corners.area)*pi/180./1.8
        for k in range(snum):
            i = k%smallside
            j = k//smallside
            shapes.append([x[[i, i + 1]], y[[j, j + 1]]])
            c = ConvexHullonSphere(pol_to_vec(*np.deg2rad(self.locwcs.all_pix2world(np.array([y[[j, j, j + 1, j + 1]], x[[i, i + 1, i + 1, i]]]).T + 1., 1)).T))
            c = c.expand(expandsize)
            grid.append(c)
            rslice.append(np.copy(rmap[x[i]:x[i + 1], y[j]: y[j + 1]]))
            eslice.append(np.copy(emap[x[i]:x[i + 1], y[j]: y[j + 1]]))
            mslice.append(np.copy(mask[x[i]:x[i + 1], y[j]: y[j + 1]]))

        qvecs = qvals.apply([1, 0, 0])
        masks = ThreadPool(mpnum).map(lambda g: g.check_inside_polygon(qvecs), grid)
        pool = Pool(mpnum)
        print(self.locwcs)
        for shape, r in tqdm.tqdm(pool.imap(self.permute_mp_2_worker, zip(shapes, repeat(rfun), rslice, eslice, mslice,
                                                                (qvals[m] for m in masks), (np.copy(x01[m]) for m in masks), (np.copy(y01[m]) for m in masks),
                                                                (np.copy(bkgs[m]) for m in masks), (np.copy(scales[m]) for m in masks), repeat(self.locwcs),
                                                                repeat(None) if energy is None else (np.copy(energy[m]) for m in masks))), total=snum):
            np.copyto(self.img[shape[0][0]:shape[0][1], shape[1][0]: shape[1][1]], r)


def run_in_endlessloop(func, q):
    while True:
        func(q.get())



localsky = None
syncbarrier = None


class SkyImageMP:

    def __init__(self, locwcs, vmap=None, shape=None, mpnum=MPNUM):
        self.barrier = Barrier(mpnum)
        self.pool = Pool(mpnum, initializer=self.initializer, initargs=(locwcs, vmap, shape, self.barrier))
        self.shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
        self.img = np.zeros(np.diff(self.shape, axis=1).ravel(), np.double)


    @staticmethod
    def initializer(locwcs, vmap, shape, barrier):
        global localsky
        global syncbarrier
        syncbarrier = barrier
        localsky = SkyImage(locwcs, vmap, shape)

    @staticmethod
    def _interpolate_worker(args): #qval, norm):
        qval, norm = args
        localsky.interpolate_vmap_for_qval(qval, norm, localsky.img)

    def interpolate(self, qvals, norm, finalize=True):
        for _ in tqdm.tqdm(self.pool.imap_unordered(self._interpolate_worker, zip(qvals, norm)), total=len(qvals)):
            pass
        #self.pool.map(self._interpolate_worker, zip(qvals, norm))
        if finalize:
            self._accumulate_img()
            self._clean_img()

    @staticmethod
    def _accumulate_worker(args):
        syncbarrier.wait()
        return localsky.img

    def _accumulate_img(self):
        limg = np.sum(self.pool.map(self._accumulate_worker, [None for _ in range(self.pool._processes)]), axis=0)
        self.img += limg

    @staticmethod
    def _cleanimg_worker(args):
        syncbarrier.wait()
        localsky.img[:, :] = 0.

    def _clean_img(self):
        self.pool.map(self._cleanimg_worker, [None for _ in range(self.pool._processes)])

    @staticmethod
    def _set_vmap_worker(args):
        vmap = args
        localsky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        syncbarrier.wait()

    def _set_core(self, vmap):
        self.pool.map(self._set_vmap_worker, [vmap for _ in range(self.pool._processes)])


    @staticmethod
    def _set_vmapval_worker(args):
        vmapvals = args
        localsky.vmap.values = vmapvals #_set_core(vmap.grid[0], vmap.grid[1], vmap.values)
        syncbarrier.wait()

    def _set_core_values(self, vmapvals):
        self.pool.map(self._set_vmapval_worker, [vmapvals for _ in range(self.pool._processes)])


    @staticmethod
    def _interpolate_bunch_worker(args):
        qvals, norms, vmapvals = args
        localsky.vmap.values = vmapvals
        for q, n in zip(qvals, norms):
            localsky.interpolate_vmap_for_qval(q, n, localsky.img)

    @staticmethod
    def _interpolate_bunch_convolve_worker(args):
        qvals, norms, vmapvals = args
        #print("qvals size", len(qvals), norm.size)
        localsky.vmap.values = vmapvals
        localsky.convolve(qvals, norms, img=localsky.img, mpnum=1)


    def interpolate_bunch(self, tasks, kind="stright"):
        if kind == "stright":
            for _ in tqdm.tqdm(self.pool.imap_unordered(self._interpolate_bunch_worker, tasks), total=len(tasks)):
                pass
        if kind == "convolve":
            for _ in tqdm.tqdm(self.pool.imap_unordered(self._interpolate_bunch_convolve_worker, tasks), total=len(tasks)):
                pass

        #self.pool.map(self._interpolate_banch_worker, tasks)
