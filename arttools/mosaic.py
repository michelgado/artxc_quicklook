import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import convolve
#from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Process, Queue, RawArray, cpu_count
from multiprocessing.pool import ThreadPool
from threading import Thread, Lock, current_thread
from .orientation import vec_to_pol, pol_to_vec, OPAX, make_quat_for_wcs, get_wcs_roll_for_qval
from ._det_spatial import offset_to_vec, vec_to_offset
from .planwcs import Corners, PRIME_NUMBERS
from .psf import unpack_inverse_psf, get_ipsf_interpolation_func
from itertools import cycle
from math import sin, cos, sqrt, pi, log
import asyncio
import matplotlib.pyplot as plt
from copy import copy
from ctypes import c_bool
import sys

import time


MPNUM = cpu_count()

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

class SkyImage(object):
    """
    stores an image of the sky and allows to put specific core on the sky coordinates
    """
    def __init__(self, locwcs, vmap=None, shape=None, rfun=get_source_photon_probability):
        self.locwcs = locwcs
        self.shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
        self.img = np.zeros(np.diff(self.shape, axis=1).ravel(), np.double)
        self.rfun = rfun
        self.lock = Lock()

        self.y, self.x = np.mgrid[self.shape[0][0] + 1:self.shape[0][1] + 1:1, self.shape[1][0] + 1: self.shape[1][1] + 1:1]
        self.ra, self.dec = self.locwcs.all_pix2world(np.array([self.x.ravel(), self.y.ravel()]).T, 1).T
        self.vecs = pol_to_vec(*np.deg2rad([self.ra, self.dec])).reshape(list(self.img.shape) + [3, ])
        if not vmap is None:
            self._set_core(vmap.grid[0], vmap.grid[1], vmap.values)

    def _set_corners(self, vals):
        """
        expect to receive four vectors at the corners of interpolation map
        """
        self.corners = Corners(vals)

    def _set_core(self, x, y, core):
        self.vmap = RegularGridInterpolator((x, y), core, bounds_error=False, fill_value=0.)
        self._set_corners(offset_to_vec(x[[0, 0, -1, -1]], y[[0, -1, -1, 0]]))

    def _update_interpolation_core_values(self, core):
        self.vmap.values = core

    def _get_quat_rectangle(self, qval):
        ra, dec = vec_to_pol(qval.apply(self.corners.corners))
        x, y = self.locwcs.all_world2pix(np.rad2deg([ra, dec]).T, 1).T - 0.5
        x = x - self.shape[1][0]
        y = y - self.shape[0][0]
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
        return il, ir, jl, jr


    def interpolate_vmap_for_qval(self, qval, norm, img=None):
        img = self.img if img is None else img
        il, ir, jl, jr = self._get_quat_rectangle(qval)
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
            sky.interpolate_vmap_for_qval(quat, norm)
        qout.put(sky.img)

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
            aval = self.rfun(core, bkg, rl)
            """
            print(core)
            print(mt.sum(), mt.size, il, ir, jl, jr)
            print(xl.min(), xl.max(), yl.min(), yl.max())
            #print("add this", aval.sum(),)
            print("comute for", rl.size, aval.sum(), mt.sum())
            """
            img[il:ir, jl:jr][mt] += aval

    def permute_thread_pool_worker(self, args):
        sl, i, j = args
        img = self.imgbuffer.pop(0)
        vmap = self.vmapbuffer.pop(0)
        vmap.values = unpack_inverse_psf(i, j)
        for q, b, s in zip(self.qvals[sl], self.bkgs[sl], self.scales[sl]):
            self.permute_with_rmap(q, b, s, vmap, img)
        self.imgbuffer.append(img)
        self.vmapbuffer.append(vmap)


    def permute_banch(self, rmap, qvals, x01, y01, bkgs, scales, mask):
        ijpairs, iidx, counts = np.unique(np.array([x01, y01]), axis=1, return_counts=True, return_inverse=True)
        isidx = np.argsort(iidx)
        ii = np.concatenate([[0,], np.cumsum(counts[:-1])])
        self.mask = mask
        self.detm = np.zeros(mask.shape, np.bool)
        self.bkgs = bkgs[isidx]
        self.scales = scales[isidx]
        self.qvals = qvals[isidx]
        self.rmap = rmap
        slices = [slice(s, e) for s, e in zip(ii, ii + counts)]
        for k, sl in enumerate(slices):
            i, j = ijpairs[:, k]
            self.vmap.values = unpack_inverse_psf(i, j)
            for q, b, s in zip(self.qvals[sl], self.bkgs[sl], self.scales[sl]):
                ra, dec = np.rad2deg(vec_to_pol(q.apply([1, 0, 0])))
                self.permute_with_rmap(q, b, s)


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
        sky = SkyImage(locwcs, vmap, rfun=rfun)
        sky.rmap = rmap
        sky.mask = mask
        sky.detm = detm
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

    def permute_mp(self, rmap, qvals, x01, y01, bkgs, scales, mask, mdet, mpnum=MPNUM):
        self.img[:, :] = 0.
        maskv = RawArray(c_bool, mask.size)
        maskd = RawArray(c_bool, mask.size)
        rmapv = RawArray(rmap.dtype.char, int(mask.sum()))
        np.copyto(np.frombuffer(rmapv), rmap[mask])
        np.copyto(np.frombuffer(maskv, np.bool).reshape(rmap.shape), mask)
        np.copyto(np.frombuffer(maskd, np.bool).reshape(rmap.shape), mdet)
        qin = Queue(100)
        qout = Queue()
        res = [self.img, ]
        collectors = [Thread(target=self.collector, args=(qout,)) for _ in range(mpnum)]
        for collector in collectors:
            collector.start()

        pool = [Process(target=self.permute_mp_worker, args = \
                        (rmapv, maskv, maskd, rmap.shape, self.locwcs, self.vmap.grid, qin, qout, self.rfun)) \
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
            sys.stderr.write('\rdone {0:%}'.format((i + 1.)/counts.size))

        for worker in pool:
            qin.put(-1)

        for collector in collectors:
            collector.join()

        for worker in pool:
            worker.join()

        return np.copy(self.img)


    def permute_worker_2(self, ):
        allevts = qvals.apply(v)
        mloc = self.corners.check_inside_polygon(allevts)
        qloc, xloc, yloc, bloc, sloc = self.qvals[mask]

    @classmethod
    def permute_mp_2_worker(cls, args):
        shape, rmap, emap, mask, qvals, x01, y01, bkgs, scales, locwcs = args
        #print(shape, x01.size)
        """
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.imshow(emap)
        plt.subplot(132)
        plt.imshow(rmap)
        """
        vmap = get_ipsf_interpolation_func()
        sky = cls(locwcs, vmap, shape)
        #print(sky.img.shape, mask.shape)
        ctot = np.zeros(sky.img.shape, np.double)
        for i in range(2):
            sky.img[:, :] = 0.
            sky.permute_banch(rmap, qvals, x01, y01, bkgs, scales, mask)
            ctot[mask] = sky.img[mask]
            mask[mask] = np.abs(sky.img[mask] - emap[mask]*rmap[mask]) > np.maximum(sky.img[mask], 1.)*0.01
            rmap = np.copy(sky.img/emap)
            if not np.any(mask):
                break
        """
        plt.subplot(133)
        plt.imshow(ctot)
        plt.show()
        """
        return sky.shape, ctot

    def permute_mp_2(self, rmap, emap, qvals, x01, y01, bkgs, scales, mask, mdet, mpnum=MPNUM):
        self.img[:, :] = 0.
        imgarea = self.img.shape[0]*self.locwcs.wcs.cdelt[0]*self.img.shape[1]*self.locwcs.wcs.cdelt[1]
        snum = min(4*mpnum, int(imgarea*9)) #curreent inverse psf core has 10 arcmin side, so area 1/9 sq.arcdeg will double the amount of events sent to processes (smaller area will increase that number)
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
        expandsize = sqrt(self.corners.area)*pi/180./1.95
        for k in range(snum):
            i = k%smallside
            j = k//smallside
            shapes.append([x[[i, i + 1]], y[[j, j + 1]]])
            c = Corners(pol_to_vec(*np.deg2rad(self.locwcs.all_pix2world(np.array([y[[j, j, j + 1, j + 1]], x[[i, i + 1, i + 1, i]]]).T + 1., 1)).T))
            c = c.expand(expandsize)
            grid.append(c)
            rslice.append(rmap[x[i]:x[i + 1], y[j]: y[j + 1]])
            eslice.append(emap[x[i]:x[i + 1], y[j]: y[j + 1]])
            mslice.append(mask[x[i]:x[i + 1], y[j]: y[j + 1]])

        qvecs = qvals.apply([1, 0, 0])
        masks = ThreadPool(mpnum).map(lambda g: g.check_inside_polygon(qvecs), grid)

        """
        print(masks[7].sum())
        sky = SkyImage(self.locwcs, self.vmap, shapes[7])
        sky.mask = mslice[7]
        sky.rmap = rslice[7]
        sky.detm = np.zeros(sky.mask.shape, np.bool)
        import pyds9
        from astropy.io import fits
        import matplotlib.pyplot as plt
        ds9 = pyds9.DS9("test")
        time.sleep(4.)
        ra, dec = np.rad2deg(vec_to_pol(sky.vecs.reshape((-1, 3))))
        print(ra.min(), ra.max(), dec.min(), dec.max())
        print(ra.size, dec.size)
        for r, d in zip(ra[::100], dec[::100]):
            ds9.set("regions", 'fk5; circle %.5f %.5f 50" # color="cyan"' % (r, d))
        ra, dec = np.rad2deg(vec_to_pol(qvals[masks[7]].apply([1, 0, 0])))
        print(ra.min(), ra.max(), dec.min(), dec.max())
        ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(emap, header=self.locwcs.to_header())]))
        for r, d in zip(ra[::50], dec[::50]):
            ds9.set("regions", 'fk5; circle %.5f %.5f 50"' % (r, d))

        ra, dec = np.rad2deg(vec_to_pol(sky.vecs[[0, -1, -1, 0], [0, 0, -1, -1]]))
        print("fk5; polygon " + " ".join(np.array([ra, dec]).T.ravel().astype(np.str)) + ' # color="red"' )
        print("compute on", mslice[7].sum())
        ds9.set("regions", "fk5; polygon " + " ".join(np.array([ra, dec]).T.ravel().astype(np.str))  + ' # color="red"')

        ra, dec = np.rad2deg(vec_to_pol(qvals[masks[7]][6].apply([1, 0, 0])))
        ds9.set("regions", 'fk5; circle %.6f %.6f 60" # color="red"' % (ra, dec))
        sky.vmap.values = unpack_inverse_psf(x01[masks[7]][6], y01[masks[7]][6])
        plt.imshow(sky.vmap.values)
        plt.show()

        sky.permute_with_rmap(qvals[masks[7]][6], bkgs[masks[7]][6], scales[masks[7]][6])
        plt.imshow(sky.img)
        plt.show()
        sky.permute_banch(rslice[7], qvals[masks[7]], x01[masks[7]], y01[masks[7]], bkgs[masks[7]], scales[masks[7]], mslice[7])
        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.imshow(eslice[7])
        plt.subplot(132)
        plt.imshow(rslice[7])
        plt.subplot(133)
        plt.imshow(sky.img)
        plt.show()
        """

        pool = Pool(mpnum)
        """
        import pyds9
        import time
        from astropy.io import fits
        ds9 = pyds9.DS9("tmp")
        time.sleep(3)
        import matplotlib.pyplot as plt
        from copy import copy
        for k in range(snum):
            #print(shapes[0], eslice[0].sum())
            ra, dec = np.rad2deg(vec_to_pol(qvecs[masks[k]]))
            #wcs = copy(self.locwcs)
            #wcs.wcs.crpix = wcs.wcs.crpix - [shapes[k][1][0], shapes[k][0][0]]
            print("events", masks[k].sum())
            r = self.permute_mp_2_worker([shapes[k], rslice[k], eslice[k], mslice[k], qvals[masks[k]],
                                      x01[masks[k]], y01[masks[k]], bkgs[masks[k]], scales[masks[k]], self.locwcs])
            i = k%smallside
            j = k//smallside
            self.img[:, :] = 0.
            self.img[x[i]:x[i + 1], y[j]: y[j + 1]] = 1. #r[:, :]

            ds9.set_pyfits(fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=self.img, header=self.locwcs.to_header())]))
            for r, d in zip(ra, dec):
                ds9.set("regions", 'fk5; circle %.6f %.6f 30"' % (r, d))
            time.sleep(30.)
            #plt.imshow(r)
        pause
        """
        for shape, r in pool.imap(self.permute_mp_2_worker, zip(shapes, rslice, eslice, mslice, (qvals[m] for m in masks),
                                                                (x01[m] for m in masks), (y01[m] for m in masks),
                                                                (bkgs[m] for m in masks), (scales[m] for m in masks), cycle([self.locwcs,]))):
            print(shape, r.shape, type(r))
            #plt.imshow(r)
            #plt.show()
            np.copyto(self.img[shape[0][0]:shape[0][1], shape[1][0]: shape[1][1]], r)
        from matplotlib.colors import LogNorm
        plt.imshow(self.img, norm=LogNorm())
        plt.show()
