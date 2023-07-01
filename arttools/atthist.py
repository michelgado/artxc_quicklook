from .time import make_ingti_times, deadtime_correction, GTI, tGTI
from .vector import pol_to_vec, vec_to_pol, normalize
from .orientation import quat_to_pol_and_roll, align_with_z_quat, minimize_norm_to_survey, get_linear_and_rot_components
from .planwcs import get_wcs_roll_for_qval
from ._det_spatial import DL, offset_to_vec, vec_to_offset_pairs, vec_to_offset
from .telescope import OPAX
from .mask import edges as medges
import healpy

from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from math import cos, sin, pi, sqrt
import numpy as np
from scipy import ndimage
import sys

from multiprocessing import Pool, cpu_count, Queue, Process

MPNUM = cpu_count()

r0 = Rotation([0, 0, 0, 1])

DELTASKY = 15./3600./180.*pi #previously I set it to be 5''
"""
optica axis is shifted 11' away of the sattelite x axis, therefore we need some more fine resolution
5'' binning at the edge of the detector, is rotation take place around its center is 2*pi/9/24
(hint: pix size 45'', 5''=45''/9)
"""
DELTAROLL = 1./24./3.

def hist_quat(quat):
    """
    provided set of quats split them on several groups, corresponding to a limited sets of optical axis direction on sky and roll angles
    the number  of groups is defined by two parameters - DELTASKY and DELTAROLL
    DELTASKY


    params: quat - a set of quats stored in scipy.spatial.transfrom.Rotation class

    return: unique sets of ra, dec, roll values, indices and inverse indeces of quats for corresponding set (see np.unique)
    """
    ra, dec, roll = quat_to_pol_and_roll(quat)
    orhist = np.empty((ra.size, 3), int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, int)
    orhist[:, 1] = np.asarray(np.cos(dec - dec%DELTASKY)*ra/DELTASKY, int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def hist_orientation(qval, dt):
    """
    provided with quats, and time spent* in the direction defined by quat
    produces grouped by ra, dec and roll quaternions and corresponding time, spent in quats

    params: qval  a set of quats stored in scipy.spatial.transfrom.Rotation class
    params: dt corresponding to the set of quaternions, set of time intervals duration (which sc spent in the dirrection defined by quaternion)

    return: exptime, qval - histogramed set of quaterninons with corresponding times
    """
    oruniq, uidx, invidx = hist_quat(qval)
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt)
    return exptime, qval[uidx]


def join_short_rotation_intervals(te, gaps, qvals, ax=OPAX, deltamove=pi/180.*15./3600., deltarot=0.1*pi/180):
    qrot = (qvals[1:]*qvals[:-1].inv()).as_rotvec()
    moda = np.sqrt(np.sum(qrot**2, axis=1))
    caa = np.sqrt(1. - np.sum(qrot*ax, axis=1)**2/moda**2.)
    lienarshift = caa*moda



def make_small_steps_quats(attdata, gti=tGTI, tedges=None, dlin=DELTASKY, drot=DELTAROLL, ax=OPAX):
    """
    provided with AttDATA container (see arttools.orientation.AttDATA)
    produces a set of quaternions, which separated not more than by DELTASY in angles and DELTAROLL in rolls

    params: 'attdata' (AttDATA container, which defines interpolation of quaterninons with time  within attached gti)
            'timecorrection' - function which produce timescaling depending on time (time weights)

    returns: qval, exptime, gti - quatertions, exposure time for this quaternions, and resulted overall gti
    """
    locgti = gti & attdata.gti
    if tedges is None:
        tnew, gaps = locgti.make_tedges(attdata.times)
    else:
        tnew, gaps = locgti.make_tedges(np.unique(np.concatenate([attdata.times, tedges])))

    q = attdata(tnew)
    lin, rot = get_linear_and_rot_components((q[1:].inv()*q[:-1]).as_rotvec())
    splitsize = (np.maximum(lin/dlin, rot/drot) + 1).astype(int)
    splitsize[~gaps] = 1
    dt = np.diff(tnew)
    gaps = np.repeat(gaps, splitsize)
    te = np.empty(gaps.size + 1, float)
    te[:-1] = np.repeat(tnew[:-1], splitsize) + (np.arange(splitsize.sum()) - np.repeat(np.cumsum([0,] + list(splitsize[:-1])), splitsize))*np.repeat(dt/splitsize, splitsize)
    te[-1] = tnew[-1]
    return te, gaps, locgti


def make_wcs_steps_quats(wcs, attdata, gti=tGTI, tedges=None, ax=OPAX):
    if (gti & attdata.gti).exposure != attdata.gti.exposure:
        attloc = attdata.apply_gti(gti)
    else:
        attloc = attdata
    if attloc.gti.exposure == 0:
        return np.empty(0, float), np.empty(0, bool), attloc.gti
    radec = np.rad2deg(vec_to_pol(attloc(attloc.times).apply(ax)))
    xy = wcs.all_world2pix(radec.T, 1).T
    gaps = gti.mask_external((attloc.times[1:] + attloc.times[:-1])/2.)
    #assuming the moving along coordinate is almost linear
    dx, dy = np.diff(xy[0]), np.diff(xy[1])
    xyint = (xy + 0.5).astype(int)
    nshiftx = np.abs(np.diff(xyint[0]))
    nshifty = np.abs(np.diff(xyint[1]))
    nshiftx[~gaps] = 0
    nshifty[~gaps] = 0
    dt = np.diff(attloc.times)
    xte = ((np.arange(nshiftx.sum()) + np.repeat(nshiftx - nshiftx.cumsum(), nshiftx) + 0.5)*np.repeat(np.sign(dx), nshiftx) + np.repeat(xyint[0, :-1] - xy[0, :-1], nshiftx))/np.repeat(dx/dt, nshiftx) + np.repeat(attloc.times[:-1], nshiftx)
    yte = ((np.arange(nshifty.sum()) + np.repeat(nshifty - nshifty.cumsum(), nshifty) + 0.5)*np.repeat(np.sign(dy), nshifty) + np.repeat(xyint[1, :-1] - xy[1, :-1], nshifty))/np.repeat(dy/dt, nshifty) + np.repeat(attloc.times[:-1], nshifty)
    te = np.unique(np.concatenate([attloc.times, xte, yte] if tedges is None else [attloc.times, xte, yte, tedges]))
    return make_small_steps_quats(attloc, gti, tedges=te, ax=ax)

def hist_orientation_for_attdata(attdata, gti=tGTI, timecorrection=lambda x:1., wcs=None):
    """
    given the AttDATA, gti and timecorrection (function which weights each time interval, in case of exposure map it is livetime fraction, or background lightcurve for the background map)

    """
    if wcs is None:
        print("small steps", gti.length)
        te, gaps, locgti = make_small_steps_quats(attdata, gti)
    else:
        print("wcs steps")
        te, gaps, locgti = make_wcs_steps_quats(wcs, attdata, gti)
    tc = (te[1:] + te[:-1])[gaps]/2.
    dtn = np.diff(te)[gaps]*timecorrection(tc)
    exptime, qhist = hist_orientation(attdata(tc), dtn)
    return exptime, qhist, locgti

def hist_by_roll_for_attdata(attdata, gti=tGTI, timecorrection=lambda x:1., wcs=None): #wcsax=[0, 0, 1]):
    te, gaps, locgti = make_small_steps_quats(attdata, gti, timecorrection)
    tc = (te[1:] + te[:-1])[gaps]/2.
    qval = attdata(tc)
    dtn = np.diff(te)[gaps]*timecorrection(tc)
    if wcs is None:
        ra, dec, roll = quat_to_pol_and_roll(qval)
        roll = (roll*180./pi)%360
    else:
        roll = get_wcs_roll_for_qval(wcs, qval)
        ra, dec = vec_to_pol(qval.apply([1, 0, 0]))

    idx = np.argsort(roll)
    ra, dec, qval, dtn, roll = ra[idx], dec[idx], qval[idx], dtn[idx], roll[idx]
    rs = np.linspace(0., 360., 721)
    rsc = (rs[1:] + rs[:-1])/2.
    se = roll.searchsorted(rs)
    return [(ra[se[i]:se[i+1]], dec[se[i]:se[i+1]], dtn[se[i]:se[i+1]], rsc[i]) for i in range(rsc.size) if se[i] != se[i + 1]]

def make_convolve_with_roll(*args):
    ra, dec, dt, roll, profile, locwcs = args
    x, y = locwcs.all_world2pix(np.array([ra*180./pi, dec*180./pi]).T, 1.).T
    timg = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                 np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    locbkg = ndimage.rotate(profile, roll)
    img = ndimage.convolve(timg, locbkg) + img


def convolve_profile(attdata, locwcs, profile, gti=tGTI, timecorrection=lambda x: 1.):
    rolls = hist_by_roll_for_attdata(attdata, gti, timecorrection, locwcs)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    img = 0.
    for i in range(len(rolls)):
        ra, dec, dt = rolls[i]
        if ra.size == 0:
            continue
        roll = i*0.5 + 0.25
        x, y = locwcs.all_world2pix(np.array([ra*180./pi, dec*180./pi]).T, 1.).T
        timg = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                     np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
        locbkg = ndimage.rotate(profile, roll)
        img = ndimage.convolve(timg, locbkg) + img
    return img


class AttHist(object):
    """
    base class which performrs FoV slide over provided wcs defined area
    """

    def __init__(self, vmap, qin, qout, subscale=4):
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
        self.vecs = self.vecs/np.sqrt(np.sum(self.vecs**2., axis=1))[:, np.newaxis]

    def put_vmap_on_sky(self, quat, exp):
        raise NotImplementedError("its a prototype class for vignetting to sky hist, the method is not implemented")

    def __call__(self):
        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            q, exp = vals
            self.put_vmap_on_sky(q, exp)
        self.qout.put(self.res)

    @staticmethod
    def trace_and_collect(exptime, qvals, qin, qout, pool, accumulate, *args):
        for proc in pool:
            proc.start()

        for i in range(exptime.size):
            qin.put([qvals[i], exptime[i]])
            sys.stderr.write('\rdone {0:%}'.format(i/exptime.size))

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
    def make_mp(cls, vmap, exptime, qvals, *args, mpnum=MPNUM, **kwargs):
        raise NotImplementedError("need to be implemented")
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(vmap, qin, qout, **kwargs)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool, cls.accumulate, *args)
        return resimg


class AttWCSHist(AttHist):

    def __init__(self, *args, wcs, imgshape=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.wcs = wcs
        if imgshape is None:
            imgshape = [int(wcs.wcs.crpix[1]*2 + 1), int(wcs.wcs.crpix[0]*2 + 1)]
        self.img = np.zeros(imgshape, np.double)
        self.res = self.img


    def put_vmap_on_sky(self, quat, exp):
        vec_fk5 = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_fk5)
        x, y = (self.wcs.all_world2pix(np.rad2deg(np.array([r, d]).T), 1) - 0.5).T.astype(int)
        u, idx = np.unique(np.array([x, y]), return_index=True, axis=1)
        mask = np.all([u[0] > -1, u[1] > -1, u[0] < self.img.shape[1], u[1] < self.img.shape[0]], axis=0)
        u, idx = u[:, mask], idx[mask]
        np.add.at(self.img, (u[1], u[0]), self.vmap[idx]*exp)

    @classmethod
    def make_mp(cls, vmap, exptime, qvals, wcs, mpnum=MPNUM, **kwargs):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(vmap, qin, qout, wcs=wcs, **kwargs)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool, cls.accumulate)
        return resimg

class AttWCSHistmean(AttWCSHist):

    def put_vmap_on_sky(self, quat, exp):
        vec_fk5 = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_fk5)
        x, y = (self.wcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1) - 0.5).T.astype(int)
        mask = np.all([x > -1, y > -1, x < self.img.shape[1], y < self.img.shape[0]], axis=0)
        x, y, vmap = x[mask], y[mask], self.vmap[mask]
        u, idx, cts = np.unique(np.array([x, y]), return_inverse=True, return_counts=True, axis=1)
        #mask = np.all([u[0] > -1, u[1] > -1, u[0] < self.img.shape[1], u[1] < self.img.shape[0]], axis=0)
        #u, idx = u[:, mask], idx[mask]
        np.add.at(self.img, (y, x), vmap*exp/cts[idx])

class AttWCSHistinteg(AttWCSHist):

    def put_vmap_on_sky(self, quat, exp):
        vec_fk5 = quat.apply(self.vecs)
        r, d = vec_to_pol(vec_fk5)
        x, y = (self.wcs.all_world2pix(np.rad2deg(np.array([r, d]).T), 1) - 0.5).T.astype(int)
        np.add.at(self.img, (y, x), self.vmap*exp)


class AttHealpixHist(AttHist):

    def __init__(self, *args, nside, **kwargs):
        super().__init__(*args, **kwargs)
        self.nside = nside
        self.res = {}

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
            self.res[u[i]] = self.res.get(u[i], 0) + exp*self.vmap[posidx[i]]

    @staticmethod
    def accumulate(qout, size, nside):
        img = np.zeros(healpy.nside2npix(nside))
        for proc in range(size):
            for idx, exptime in qout.get().items():
                img[idx] += exptime
        return img

    @classmethod
    def make_mp(cls, vmap, exptime, qvals, nside, mpnum=MPNUM):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(vmap, qin, qout, nside=nside)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool, nside)
        return resimg


class AttCircHist(AttHist):
    def __init__(self, *args, dvec, rapp, **kwargs):
        super().__init__(*args, **kwargs)
        self.dvec = dvec
        self.rapp = rapp
        self.res = 0.

    def put_vmap_on_sky(self, quat, exp):
        vec_icrs = quat.apply(self.vecs)
        mask = np.arccos(np.minimum(np.sum(vec_icrs*self.dvec, axis=1), 1.)) < self.rapp
        self.res += (0. if not np.any(mask) else np.mean(self.vmap[mask]))

    @classmethod
    def make_mp(cls, vmap, exptime, qvals, dvec, rapp, mpnum=MPNUM):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(vmap, qin, qout, dvec=dvec, rapp=rapp)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool, cls.accumulate)
        return resimg


DET_CORNERS = offset_to_vec(np.array([-24, 24, 24, -24])*DL, np.array([24, 24, -24, -24])*DL)
DET_CORNERS = DET_CORNERS/np.sqrt(np.sum(DET_CORNERS**2, axis=1))[:, np.newaxis]

class AttInvHist(object):

    def __init__(self, vmap, qin, qout, locwcs):
        self.qin = qin
        self.qout = qout
        self.vmap = vmap
        self.locwcs = locwcs
        clist = offset_to_vec(vmap.grid[0][[0, 0, -1, -1]]*1.01, vmap.grid[1][[0, -1, -1, 0]]*1.01) #np.array([np.ones(4, np.double), np.tan(1.01*vmap.grid[0][[0, 0, -1, -1]]*pi/180/60.), np.tan(1.01*vmap.grid[1][[0, -1, -1, 0]]*pi/180/60.)]).T
        clist = clist/np.sqrt(np.sum(clist**2, axis=1))[:, np.newaxis]
        self._set_corners(clist)
        self.img = np.zeros((int(self.locwcs.wcs.crpix[1]*2 + 1),
                             int(self.locwcs.wcs.crpix[0]*2 + 1)), np.double)
        self.y, self.x = np.mgrid[1:self.img.shape[0] + 1:1, 1:self.img.shape[1] + 1:1]
        self.ra, self.dec = self.locwcs.all_pix2world(np.array([self.x.ravel(), self.y.ravel()]).T, 1).T
        self.ra, self.dec = self.ra.reshape(self.x.shape), self.dec.reshape(self.x.shape)
        self.vecs = pol_to_vec(*np.deg2rad([self.ra, self.dec]))

    def _set_corners(self, vals=DET_CORNERS):
        self.corners = vals

    def single_step(self, qval, exptime):
        ra, dec = vec_to_pol(qval.apply(self.corners))
        x, y = self.locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
        jl, jr = max(int(x.min()), 0), min(self.img.shape[1] - 1, int(x.max()+1))
        il, ir = max(int(y.min()), 0), min(self.img.shape[0] - 1, int(y.max()+1))
        vecs = self.vecs[il: ir + 1, jl: jr + 1]
        xl, yl = vec_to_offset(qval.apply(vecs.reshape((-1, 3)), inverse=True))
        self.img[il:ir+1, jl:jr+1] += self.vmap((xl, yl)).reshape(vecs.shape[:2])*exptime

    def __call__(self):
        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            qval, exptime = vals
            self.single_step(qval, exptime)

        self.qout.put(self.img)

    @staticmethod
    def trace_and_collect(exptime, qvals, qin, qout, pool, accumulate, *args):
        for proc in pool:
            proc.start()

        for i in range(exptime.size):
            qin.put([qvals[i], exptime[i]])
            sys.stderr.write('\rdone {0:%}'.format(i/exptime.size))

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
    def make_mp(cls, locwcs, vmap, exptime, qvals, *args, mpnum=MPNUM, **kwargs):
        qin = Queue(100)
        qout = Queue(2)
        pool = [Process(target=cls(vmap, qin, qout, locwcs, **kwargs)) for i in range(mpnum)]
        resimg = cls.trace_and_collect(exptime, qvals, qin, qout, pool, cls.accumulate, *args)
        return resimg
