from .time import make_ingti_times, deadtime_correction, GTI, tGTI
from .orientation import quat_to_pol_and_roll, pol_to_vec, \
    vec_to_pol, get_wcs_roll_for_qval, get_survey_mode_rotation_plane, align_with_z_quat
from .caldb import ARTQUATS
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
    ra, dec, roll = quat_to_pol_and_roll(quat)
    orhist = np.empty((ra.size, 3), np.int)
    orhist[:, 0] = np.asarray((dec + pi/2.)/DELTASKY, np.int)
    orhist[:, 1] = np.asarray(np.cos(dec - dec%DELTASKY)*ra/DELTASKY, np.int)
    orhist[:, 2] = np.asarray(roll/DELTAROLL, np.int)
    return np.unique(orhist, return_index=True, return_inverse=True, axis=0)

def hist_orientation(qval, dt):
    oruniq, uidx, invidx = hist_quat(qval)
    exptime = np.zeros(uidx.size, np.double)
    np.add.at(exptime, invidx, dt)
    return exptime, qval[uidx]

def make_small_steps_quats(attdata, gti=tGTI, timecorrection=lambda x: 1.):
    locgti = gti & attdata.gti
    tnew, maskgaps = locgti.make_tedges(attdata.times)
    if tnew.size == 0:
        return Rotation(np.empty((0, 4), np.double)), np.array([])
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]

    qval = attdata(ts)
    ra, dec, roll = quat_to_pol_and_roll(attdata(tnew))

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
        qval = attdata(ts)
    else:
        dtn = dt
    print("check exposure", dt.sum(), gti.exposure)
    return qval, dtn*timecorrection(ts), locgti

def hist_orientation_for_attdata(attdata, gti=tGTI, timecorrection=lambda x:1.):
    qval, dtn, locgti = make_small_steps_quats(attdata, gti, timecorrection)
    exptime, qhist = hist_orientation(qval, dtn)
    return exptime, qhist, locgti

def hist_by_roll_for_attdata(attdata, gti=tGTI, timecorrection=lambda x:1., wcs=None): #wcsax=[0, 0, 1]):
    qval, dtn, locgti = make_small_steps_quats(attdata, gti, timecorrection)
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
    import matplotlib.pyplot as plt
    #r, d = locwcs.all_pix2world([[0, locwcs.wcs.crpix[1]], [2*locwcs.wcs.crpix[0], locwcs.wcs.crpix[1]]], 1).T
    """
    r, d = locwcs.all_pix2world([[locwcs.wcs.crpix[0], locwcs.wcs.crpix[1]*2], [locwcs.wcs.crpix[0], 0]], 1).T
    vecs = pol_to_vec(r*pi/180., d*pi/180.)
    wcsax = vecs[1, :] - vecs[0, :]
    wcsax = wcsax/np.sqrt(np.sum(wcsax**2.))
    """
    rolls = hist_by_roll_for_attdata(attdata, gti, timecorrection, locwcs)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    img = 0.
    for i in range(len(rolls)):
        ra, dec, dt = rolls[i]
        if ra.size == 0:
            continue
        print(ra.size)
        roll = i*0.5 + 0.25
        print("roll %d exptimes %.2f" % (roll, dt.sum()))
        x, y = locwcs.all_world2pix(np.array([ra*180./pi, dec*180./pi]).T, 1.).T
        print(x, y)
        timg = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                     np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
        #return timg
        print("timg sum", timg.sum())
        #plt.imshow(timg)
        #plt.show()

        print("rotate on roll angle", roll)
        locbkg = ndimage.rotate(profile, roll)
        print("rotate image")
        print("locbkg shape", locbkg.shape)
        #timg = ndimage.convolve(timg, locbkg)
        print(timg.shape)
        print(img)
        img = ndimage.convolve(timg, locbkg) + img
        print("affter adding convolve", img)
    return img


def get_vecs_convex(vecs):
    """
    for a set of vectors (which assumed to cover less the pi along any direction)
    produces set of vectors which is verteces of convex figure, incapsulating all other vectors on  sphere
    """
    cvec = vecs.sum(axis=0)
    cvec = cvec/np.sqrt(np.sum(cvec**2.))
    vrot = np.cross(np.array([0., 0., 1.]), cvec)
    vrot = vrot/np.sqrt(np.sum(vrot**2.))
    alpha = pi/2. - np.arccos(cvec[2])
    quat = Rotation([vrot[0]*sin(alpha/2.), vrot[1]*sin(alpha/2.), vrot[2]*sin(alpha/2.), cos(alpha/2.)])
    r1 = np.array([0, 0, 1])
    r2 = np.cross(quat.apply(cvec), r1)
    vecn = quat.apply(vecs) - quat.apply(cvec)
    l, b = np.sum(quat.apply(vecs)*r2, axis=1), vecn[:,2]
    ch = ConvexHull(np.array([l, b]).T)
    r, d = l[ch.vertices], b[ch.vertices]
    return cvec, r1, r2, quat, vecs[ch.vertices], r, d


def min_area_wcs_for_vecs(vecs, pixsize=20./3600.):
    """
    produce wcs from a set of vectors, the wcs is intendet to cover this vectors
    vecs are expected to be of shape Nx3
    method:
        find mean position vector
        produce quaternion which put this vector at altitude 0, where metric is close to cartesian
        find optimal rectangle
        compute field rotation angle e.t.c.
        fill wcs with rotation angle, field size, estimated image size
        return wcs
    """
    cvec, r1, r2, eqquat, cv_vecs, r, d = get_vecs_convex(vecs)

    def find_bbox(alpha):
        x = r*cos(alpha) - d*sin(alpha)
        y = r*sin(alpha) + d*cos(alpha)
        return (x.max() - x.min())*(y.max() - y.min())
    res = minimize(find_bbox, [pi/8., ], method="Nelder-Mead")
    alpha = res.x
    x, y = r*cos(alpha) - d*sin(alpha), r*sin(alpha) + d*cos(alpha)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xc = (xmax + xmin)/2.
    yc = (ymax + ymin)/2.

    dx = (xmax - xmin)
    dy = (ymax - ymin)

    vec1 = eqquat.apply(cvec) + (xc*cos(alpha) + yc*sin(alpha))*r2 + \
                  (-xc*sin(alpha) + yc*cos(alpha))*r1
    rac, decc = vec_to_pol(eqquat.apply(vec1, inverse=True))


    locwcs = WCS(naxis=2)
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    locwcs.wcs.cdelt = [pixsize, pixsize]
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]
    locwcs.wcs.radesys = "FK5"
    desize = int((ymax - ymin + 0.2*pi/180.)*180./pi/pixsize)//2
    desize = desize + 1 - desize%2
    rasize = int((xmax - xmin + 0.2*pi/180.)*180./pi/pixsize)//2
    rasize = rasize + 1 - rasize%2
    locwcs.wcs.crpix = [rasize, desize]

    return locwcs

def min_roll_wcs_for_quats(quats, pixsize=20./3600.):
    """
    now we want to find coordinate system on the sphere surface, in which most of the attitudes would have 0 roll angle
    """

    vecs = quats.apply([1, 0, 0])
    cvec, r1, r2, eqquat, cv_vecs, r, d = get_vecs_convex(vecs)
    rac, decc = vec_to_pol(cvec)
    locwcs = WCS(naxis=2)
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]
    locwcs.wcs.cdelt = [pixsize, pixsize]
    locwcs.wcs.radesys = "FK5"
    crsize = int(np.max(np.arccos(np.sum(cvec*cv_vecs, axis=1)))*180./pi/pixsize)
    locwcs.wcs.crpix = [crsize, crsize]
    roll = get_wcs_roll_for_qval(locwcs, quats)
    mroll = np.median(roll)*pi/180.
    locwcs.wcs.pc = np.array([[cos(mroll), sin(mroll)], [-sin(mroll), cos(mroll)]])
    rav, decv = vec_to_pol(cv_vecs)
    xy = locwcs.all_world2pix(np.array([rav, decv]).T*180./pi, 1)
    xmax, ymax = np.max(xy, axis=0)
    xmin, ymin = np.min(xy, axis=0)
    locwcs.wcs.crval = locwcs.all_pix2world([[(xmax + xmin + 1)//2, (ymax + ymin + 1)//2],], 1)[0]
    locwcs.wcs.crpix = [(xmax - xmin)//2 + 0.7/pixsize, (ymax - ymin)//2 + 0.7/pixsize]
    return locwcs


def make_wcs_for_quats(quats, pixsize=20./3600.):
    vedges = offset_to_vec(np.array([-26.*DL, 26*DL, 26.*DL, -26.*DL]),
                           np.array([-26.*DL, -26*DL, 26.*DL, 26.*DL]))
    edges = np.concatenate([quats.apply(v) for v in vedges])
    edges = edges/np.sqrt(np.sum(edges**2., axis=1))[:, np.newaxis]
    return min_area_wcs_for_vecs(edges)

def make_wcs_for_attdata(attdata, gti=tGTI):
    locgti = gti & attdata.gti
    qvtot = attdata(attdata.times[locgti.mask_outofgti_times(attdata.times)])
    return make_wcs_for_quats(qvtot)

def split_survey_mode(attdata, gti=tGTI):
    aloc = attdata.apply_gti(gti)
    rpvec = get_survey_mode_rotation_plane(aloc)
    rquat = align_with_z_quat(rpvec)
    amin = np.argmin(vec_to_pol(aloc(aloc.times).apply([1, 0, 0]))[0])
    print(amin)
    """
    zeropoint = [1, 0, 0] - rpvec*rpvec[0]
    print(zeropoint)
    #zeropoint = np.cross(rpvec, [0, 1, 0])
    zeropoint = zeropoint/sqrt(np.sum(zeropoint**2.))
    """
    zeropoint = rquat.apply(aloc([aloc.times[amin],])[0].apply([1, 0, 0]))
    alpha0 = np.arctan2(zeropoint[1], zeropoint[0])

    vecs = aloc(aloc.times).apply([1, 0, 0])
    vecsz = rquat.apply(vecs)
    alpha = (np.arctan2(vecsz[:, 1], vecsz[:, 0]) - alpha0)%(2.*pi)
    edges = np.linspace(0., 2.*pi, 37)
    gtis = [medges((alpha > e1) & (alpha < e2)) + [0, -1] for e1, e2 in zip(edges[:-1], edges[1:])]
    return [GTI(aloc.times[m]) for m in gtis]


def make_wcs_for_survey(attdata, gti=tGTI, pixsize=20./3600.):
    rpvec = get_survey_mode_rotation_plane(attdata, gti)

def make_wcs_for_survey_mode(attdata, gti=None, rpvec=None, pixsize=20./3600.):
    aloc = attdata.apply_gti(gti) if not gti is None else attdata

    if rpvec is None: rpvec = get_survey_mode_rotation_plane(aloc)

    cvec = aloc(aloc.times).apply([1, 0, 0])
    cvec = cvec.mean(axis=0)
    cvec = cvec/np.sqrt(np.sum(cvec**2.))
    xvec = np.cross(cvec, rpvec)
    xvec = xvec/np.sqrt(np.sum(xvec**2.))

    alpha = np.arctan2(xvec[2], rpvec[2])

    locwcs = WCS(naxis=2)
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    locwcs.wcs.cdelt = [pixsize, pixsize]
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    rac, decc = vec_to_pol(cvec)
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]

    vecs = aloc(aloc.times).apply([1, 0, 0])
    yproj = np.sum(rpvec*vecs, axis=1)
    yangles = np.arcsin(yproj)*pi/180./pixsize
    xangles = np.arccos(np.sum(cvec*(vecs - yproj[:, np.newaxis]*rpvec[np.newaxis, :]), axis=1))/pi*180./pixsize
    yangles = np.arccos(np.sum(rpvec*vecs, axis=1))/pi*180./pixsize
    xmin, xmax = xangles.min(), xangles.max()
    ymin, ymax = yangles.min(), yangles.max()
    print(xmin, xmax, ymin, ymax)
    rasize = max(abs(xmin), abs(xmax))
    desize = max(abs(ymin), abs(ymax))
    rasize = rasize - rasize%2 + 1
    desize = desize - desize%2 + 1

    locwcs.wcs.radesys = "FK5"
    locwcs.wcs.crpix = [rasize, desize]

    return locwcs



class AttHist(object):

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
        x, y = (self.wcs.all_world2pix(np.rad2deg(np.array([r, d]).T), 1) - 0.5).T.astype(np.int)
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
        x, y = (self.wcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1) - 0.5).T.astype(np.int)
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
        x, y = (self.wcs.all_world2pix(np.rad2deg(np.array([r, d]).T), 1) - 0.5).T.astype(np.int)
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


class AttInvHist(object):

    def __init__(self, vmap, qin, qout, locwcs):
        self.qin = qin
        self.qout = qout
        self.vmap = vmap
        self.locwcs = locwcs
        corners = np.array([-24, 24, 24, -24])*DL
        self.corners = offset_to_vec(corners, np.roll(corners, 1))
        self.corners = self.corners/np.sqrt(np.sum(self.corners**2., axis=1))[:, np.newaxis]
        self.img = np.zeros((int(self.locwcs.wcs.crpix[1]*2 + 1),
                             int(self.locwcs.wcs.crpix[0]*2 + 1)), np.double)
        self.y, self.x = np.mgrid[1:self.img.shape[0] + 1:1, 1:self.img.shape[1] + 1:1]

    def __call__(self):
        while True:
            vals = self.qin.get()
            if vals == -1:
                break
            qval, exptime = vals
            ra, dec = vec_to_pol(qval.apply(self.corners))
            x, y = self.locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
            jl, jr = max(int(x.min()), 0), min(self.img.shape[1], int(x.max()+1))
            il, ir = max(int(y.min()), 0), min(self.img.shape[0], int(y.max()+1))
            x, y = self.x[il:ir + 1, jl: jr + 1], self.y[il: ir + 1, jl: jr + 1]
            ra, dec = self.locwcs.all_pix2world(np.array([x.ravel(), y.ravel()]).T, 1).T
            vecs = pol_to_vec(ra*pi/180., dec*pi/180.)
            xl, yl = vec_to_offset(qval.apply(vecs, inverse=True))
            self.img[il:ir+1, jl:jr+1] += self.vmap((xl, yl)).reshape(x.shape)*exptime

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
