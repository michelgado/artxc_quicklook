from .orientation import vec_to_pol, pol_to_vec, ART_det_QUAT
from .atthist import hist_orientation_for_attdata
from ._det_spatial import DL, offset_to_vec
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting
from .time import gti_intersection, gti_difference
from multiprocessing import Pool, cpu_count, Queue, Process
from threading import Thread
import numpy as np
import sys
from functools import reduce

MPNUM = cpu_count()

class AttWCShist(object):
    def __init__(self, wcs, vmap, qin, qout, imgshape=None, subscale=5):
        self.wcs = wcs
        self.qin = qin
        self.qout = qout
        if imgshape is None:
            imgshape = [int(wcs.wcs.crpix[1]*2 + 1), int(wcs.wcs.crpix[0]*2 + 1)]
        self.img = np.zeros(imgshape, np.double)
        xmin, xmax = vmap.grid[0][[0, -1]]
        ymin, ymax = vmap.grid[0][[0, -1]]
        dd = DL/subscale
        dx = dd - dd%(xmax - xmin)
        x = np.linspace(xmin - dx/2., xmax + dx/2., int((xmax - xmin + dx)/dd))
        dy = dd - dd%(ymax - ymin)
        y = np.linspace(ymin - dy/2., ymax + dy/2., int((ymax - ymin + dy)/dd))

        x, y = np.tile(x, y.size), np.repeat(y, x.size)
        #y = np.repeat((np.arange(-24*subscale, 24*subscale) + 0.5)*DL/subscale, 48*subscale)
        vmap = vmap(np.array([x, y]).T)
        mask = vmap > 0.
        x, y = x[mask], y[mask]
        self.vmap = vmap[mask]
        self.vecs = offset_to_vec(x, y)

    def put_vignmap_on_sky(self, quat, exp):
        vec_icrs = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = (self.wcs.all_world2pix(np.degrees(np.array([r, d]).T), 1) + 0.5).T.astype(np.int)
        u, idx = np.unique(np.array([x, y]), return_index=True, axis=1)
        mask = np.all([u[0] > -1, u[1] > -1, u[0] < self.img.shape[1], u[1] < self.img.shape[0]], axis=0)
        u, idx = u[:, mask], idx[mask]
        np.add.at(self.img, (u[1], u[0]), self.vmap[idx]*exp)

    def __call__(self):
        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            q, exp = vals
            self.put_vignmap_on_sky(q, exp)
        self.qout.put(self.img)

    @staticmethod
    def trace_and_collect(exptime, qvals, qin, qout, pool):
        for proc in pool:
            proc.start()

        for i in range(exptime.size):
            qin.put([qvals[i], exptime[i]])
            sys.stderr.write('\rdone {0:%}'.format(i/exptime.size))

        for p in pool:
            qin.put(-1)

        img = sum(qout.get() for p in pool)
        for p in pool:
            p.join()
        return img

    @classmethod
    def make_mp_expmap(cls, wcs, vmap, exptime, qvals, mpnum=MPNUM):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(wcs, vmap, qin, qout)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool)
        return resimg


def make_expmap_for_wcs(wcs, attdata, gti, mpnum=MPNUM):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    overall_gti = reduce(gti_intersection, gti.values())
    exptime, qval = hist_orientation_for_attdata(attdata, overall_gti)
    vmap = make_overall_vignetting()
    print("produce overall urds expmap")
    emap = AttWCShist.make_mp_expmap(wcs, vmap, exptime, qval, mpnum)
    print("\ndone!")
    for urd in gti:
        urdgti = gti_difference(overall_gti, gti[urd])
        if urdgti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval = hist_orientation_for_attdata(attdata, urdgti, ART_det_QUAT[urd])
        vmap = make_vignetting_for_urdn(urd)
        emap = AttWCShist.make_mp_expmap(wcs, vmap, exptime, qval, mpnum) + emap
        print(" done!")
    return emap
