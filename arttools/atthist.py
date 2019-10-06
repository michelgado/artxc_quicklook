from .time import make_ingti_times, deadtime_correction
from .orientation import get_gyro_quat, quat_to_pol_and_roll, pol_to_vec, \
    ART_det_QUAT, ART_det_mean_QUAT, vec_to_pol
from ._det_spatial import DL, offset_to_vec
from .telescope import OPAX
import healpy

from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from math import cos, sin, pi
import numpy as np
import sys

from multiprocessing import Pool, cpu_count, Queue, Process

MPNUM = cpu_count()


DELTASKY = 15./3600./180.*pi #previously I set it to be 5''
"""
optica axis is shifted 11' away of the sattelite x axis, therefore we need some more fine resolution
5'' binning at the edge of the detector, is rotation take place around its center is 2*pi/9/24
(hint: pix size 45'', 5''=45''/9)
"""
DELTAROLL = 1./24./3.

def hist_quat(quat):
    ra, dec, roll = quat_to_pol_and_roll(quat)
    orhist = np.empty((ra.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, np.int)
    orhist[:, 1] = np.asarray(np.cos(dec - dec%(pi/180.*15./3600))*ra/DELTASKY, np.int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, np.int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def hist_orientation(qval, dt):
    oruniq, uidx, invidx = hist_quat(qval)
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt)
    return exptime, qval[uidx]

def make_small_steps_quats(times, quats, gti, timecorrection=lambda x: 1.):
    quatint = Slerp(times, quats)
    tnew, maskgaps = make_ingti_times(times, gti)
    if tnew.size == 0:
        return Rotation(np.empty((0, 4), np.double)), np.array([])
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]

    qval = quatint(ts)
    ra, dec, roll = quat_to_pol_and_roll(quatint(tnew))

    """
    to do:
    formally, this subroutine should not know that optic axis is [1, 0, 0],
    need to fix this
    vec = qval.apply([1., 0, 0])
    """
    vec = pol_to_vec(ra, dec)
    vecprod = np.sum(vec[1:, :]*vec[:-1, :], axis=1)
    """
    this ugly thing appears due to the numerical precision
    """
    vecprod[vecprod > 1.] = 1.
    dalpha = np.arccos(vecprod)[maskgaps]
    cs = np.cos(roll)
    ss = np.sin(roll)
    vecprod = np.minimum(ss[1:]*ss[:-1] + cs[1:]*cs[:-1], 1.)
    droll = np.arccos(vecprod)[maskgaps]

    maskmoving = (dalpha < DELTASKY) & (droll < DELTAROLL)
    qvalstable = qval[maskmoving]
    maskstable = np.logical_not(maskmoving)
    if np.any(maskstable):
        tsm = (ts - dt/2.)[maskstable]
        size = np.maximum(dalpha[maskstable]/DELTASKY, droll[maskstable]/DELTAROLL).astype(np.int)
        dtm = np.repeat(dt[maskstable]/size, size)
        ar = np.arange(size.sum()) - np.repeat(np.cumsum([0,] + list(size[:-1])), size) + 0.5
        tnew = np.repeat(tsm, size) + ar*dtm
        dtn = np.concatenate([dt[maskmoving], dtm]) #*timecorrection(ts[maskmoving]), dtm*timecorrection(tnew)])
        ts = np.concatenate([ts[maskmoving], tnew])
        qval = quatint(ts)
    else:
        dtn = dt
    return qval, dtn*timecorrection(ts)

def hist_orientation_for_attdata(attdata, gti, corr_quat=ART_det_mean_QUAT, timecorrection=lambda x:1.):
    quats = get_gyro_quat(attdata)*corr_quat
    qval, dtn = make_small_steps_quats(attdata["TIME"], quats, gti, timecorrection)
    return hist_orientation(qval, dtn)

def hist_orientation_for_attdata_urdset(attdata, gti):
    """
    gti is expected to be a dictionary with key is urdn and value - corresponding gti
    """
    qval = get_gyro_quat(attdata)
    qtot, dtn = [], []

    for urdn in gti:
        q, dt = make_small_steps_quats(attdata["TIME"], qval*ART_det_QUAT[urdn], gti[urdn])
        qtot.append(q)
        dtn.append(dt)
    qtot = Rotation(np.concatenate([q.as_quat() for q in qtot]))
    dtn = np.concatenate(dtn)
    return hist_orientation(qtot, dtn)

def make_wcs_for_radecs(ra, dec, pixsize=20./3600.):
    radec = np.array([ra, dec]).T
    ch = ConvexHull(radec)
    r, d = radec[ch.vertices].T
    """
    vecs = pol_to_vec(r*pi/180, d*pi/180.)
    mvec = vecs.sum(axis=0)
    """
    def find_bbox(alpha):
        x = r*cos(alpha) - d*sin(alpha)
        y = r*sin(alpha) + d*cos(alpha)
        return (x.max() - x.min())*(y.max() - y.min())
    res = minimize(find_bbox, [pi/4., ], method="Nelder-Mead")
    #res = minimize(find_bbox, [0., ], method="Nelder-Mead")
    #alpha = 0. #res.x
    alpha = res.x
    x, y = r*cos(alpha) - d*sin(alpha), r*sin(alpha) + d*cos(alpha)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xc = (xmax + xmin)/2.
    yc = (ymax + ymin)/2.

    locwcs = WCS(naxis=2)
    locwcs.wcs.cdelt = [pixsize, pixsize]
    cdmat = np.array([[cos(alpha), sin(alpha)], [-sin(alpha), cos(alpha)]])
    locwcs.wcs.cd = cdmat*pixsize
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.crval = [xc*cos(alpha) + yc*sin(alpha), -xc*sin(alpha) + yc*cos(alpha)]
    desize = int((ymax - ymin + 0.7)/pixsize)//2
    desize = desize + 1 - desize%2
    rasize = int((xmax - xmin + 0.7)/pixsize)//2
    rasize = rasize + 1 - rasize%2
    locwcs.wcs.crpix = [rasize, desize]
    #inspect obtained wcs region
    """
    import matplotlib.pyplot as plt
    plt.scatter(ra, dec, color="g")
    plt.scatter(r, d, color="m")
    r1, d1 = locwcs.all_pix2world(np.array([np.arange(1, rasize*2 + 1), np.ones(rasize*2)]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.arange(1, rasize*2 + 1), np.ones(rasize*2)*desize*2]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.ones(desize*2), np.arange(1, desize*2 + 1)]).T, 1).T
    plt.scatter(r1, d1, color="b")
    r1, d1 = locwcs.all_pix2world(np.array([np.ones(desize*2)*rasize*2, np.arange(1, desize*2 + 1)]).T, 1).T
    plt.scatter(r1, d1, color="b")

    plt.scatter(*locwcs.all_pix2world([[1., 1.], [0., locwcs.wcs.crpix[1]*2 + 1],
                                      [locwcs.wcs.crpix[0]*2 + 1, locwcs.wcs.crpix[1]*2 + 1],
                                      [1., locwcs.wcs.crpix[1]*2 + 1]], 1).T, color="r")
    plt.scatter([locwcs.wcs.crval[0],], [locwcs.wcs.crval[1],], color="k")
    plt.show()
    """
    return locwcs

def make_wcs_for_quats(quats, pixsize=20./3600.):
    opaxis = quats.apply(OPAX)
    ra, dec = vec_to_pol(opaxis)
    ra, dec = ra*180/pi, dec*180/pi
    return make_wcs_for_radecs(ra, dec, pixsize)

def make_wcs_for_attdata(attdata, gti=None):
    if not gti is None:
        idx = np.searchsorted(attdata["TIME"], gti)
        mask = np.zeros(attdata.size, np.bool)
        for s, e in idx:
            mask[s:e] = True
        attdata = attdata[mask]
    qvtot = get_gyro_quat(attdata)*ART_det_mean_QUAT
    return make_wcs_for_quats(qvtot)

def make_wcs_for_attsets(attflist, gti=None):
    qvtot = []
    for attname in attflist:
        attdata = np.copy(fits.getdata(attname, 1))
        attdata = clear_att(attdata)
        if not gti is None:
            lgti = gti_intersection(gti, np.array([attdata["TIME"][[0, -1]],]))
            idx = np.searchsorted(attdata["TIME"], lgti)
            mask = np.zeros(attdata.size, np.bool)
            for s, e in idx:
                mask[s:e] = True
            attdata = attdata[mask]
        quats = get_gyro_quat(attdata)*ART_det_mean_QUAT
        qvtot.append(quats) #qvals)

    qvtot = Rotation(np.concatenate([q.as_quat() for q in qvtot]))
    return make_wcs_for_quats(qvtot)

class AttWCShist(object):
    def __init__(self, wcs, vmap, qin, qout, imgshape=None, subscale=4):
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
        vmap = vmap(np.array([x, y]).T)
        mask = vmap > 0.
        x, y = x[mask], y[mask]
        self.vmap = vmap[mask]
        self.vecs = offset_to_vec(x, y)

    def integrate_vmap_on_sky(self, quat, exp):
        pass

    def put_vmap_on_sky(self, quat, exp):
        vec_icrs = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_icrs)
        x, y = (self.wcs.all_world2pix(np.degrees(np.array([r, d]).T), 1) + 0.5).T.astype(np.int)
        #sc = SkyCoord(r*180./pi, d*180/pi, unit=("deg", "deg"), frame="fk5")
        #x, y = (self.wcs.all_world2pix(np.array([sc.galactic.l.value, sc.galactic.b.value]).T, 1) + 0.5).T.astype(np.int)
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
            self.put_vmap_on_sky(q, exp)
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
    def make_mp(cls, wcs, vmap, exptime, qvals, mpnum=MPNUM):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(wcs, vmap, qin, qout)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool)
        return resimg


class AttHealpixhist(object):
    def __init__(self, nside, vmap, qin, qout, imgshape=None, subscale=3):
        self.nside =nside
        self.qin = qin
        self.qout = qout
        xmin, xmax = vmap.grid[0][[0, -1]]
        ymin, ymax = vmap.grid[0][[0, -1]]
        dd = DL/subscale
        dx = dd - dd%(xmax - xmin)
        x = np.linspace(xmin - dx/2., xmax + dx/2., int((xmax - xmin + dx)/dd))
        dy = dd - dd%(ymax - ymin)
        y = np.linspace(ymin - dy/2., ymax + dy/2., int((ymax - ymin + dy)/dd))

        x, y = np.tile(x, y.size), np.repeat(y, x.size)
        vmap = vmap(np.array([x, y]).T)
        mask = vmap > 0.
        x, y = x[mask], y[mask]
        self.vmap = vmap[mask]
        self.vecs = offset_to_vec(x, y)
        self.pixs = {}

    def integrate_vmap_on_sky(self, quat, exp):
        pass

    def put_vmap_on_sky(self, quat, exp):
        vec_icrs = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_icrs)
        sc = SkyCoord(r*180/pi, d*180/pi, unit=("deg", "deg"), frame="fk5")
        hidx = healpy.ang2pix(self.nside, sc.galactic.l.value, sc.galactic.b.value, lonlat=True)
        u, posidx, invidx, counts = np.unique(hidx, return_index=True, return_counts=True, return_inverse=True)
        mv = np.zeros(u.size, np.double)
        np.add.at(mv, invidx, self.vmap)
        mv = mv/counts

        for i in range(u.size):
            self.pixs[u[i]] = self.pixs.get(u[i], 0) + exp*self.vmap[posidx[i]]

    def __call__(self):
        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            q, exp = vals
            self.put_vmap_on_sky(q, exp)
        self.qout.put(self.pixs)

    @staticmethod
    def trace_and_collect(exptime, qvals, nside, qin, qout, pool):
        for proc in pool:
            proc.start()

        for i in range(exptime.size):
            qin.put([qvals[i], exptime[i]])
            sys.stderr.write('\rdone {0:%}'.format(i/exptime.size))

        for p in pool:
            qin.put(-1)

        img = np.zeros(healpy.nside2npix(nside))
        for proc in pool:
            for idx, exptime in qout.get().items():
                img[idx] += exptime
        for p in pool:
            p.join()
        return img

    @classmethod
    def make_mp(cls, nside, vmap, exptime, qvals, mpnum=MPNUM):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(nside, vmap, qin, qout)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, nside, qin, qout, pool)
        return resimg

