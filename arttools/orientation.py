from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from copy import copy
from math import pi, cos, sin, sqrt
from ._det_spatial import urd_to_vec, F, DL, raw_xy_to_vec
from .sphere import ConvexHullonSphere, get_vec_triangle_area, FullSphere, ConvexHullonSphere, CVERTICES, SPHERE
from .time import get_hdu_times, GTI, tGTI, emptyGTI
from .vector import vec_to_pol, pol_to_vec, normalize
from .caldb import T0, get_boresight_by_device, get_device_timeshift, relativistic_corrections_gti, MJDREF
from .containers import Urddata
from .telescope import URDNS, OPAX
from .mask import edges as medges
from functools import reduce, lru_cache
from scipy.optimize import minimize
from datetime import datetime
from astropy.time import Time
from astropy.io import fits


#debug
import matplotlib.pyplot as plt

#============================================================================================
"""
temporal solution to the SED1/2 relativistic corrections on board
"""
from astropy.coordinates import SkyCoord
from astropy import time as atime


qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
#qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
qgyro0 = Rotation([0., 0., 0., 1.])

SECPERYR = 3.15576e7
SOLARSYSTEMPLANENRMALEINFK5 = np.array([-9.83858346e-08, -3.97776911e-01,  9.17482168e-01])

#gyrocorrectionbti = GTI([[624390347, 624399808], [6.24410643e+08, 6.30954575e+08]])
gyrocorrectionbti = GTI(relativistic_corrections_gti)
#gyrocorrectionbti = emptyGTI #GTI([[624390347, 624399808], [6.24410643e+08, 6.30954575e+08]])

class SlerpWithNaiveIndexing(Slerp):
    """
    scipy quaternions interpolation class with indexing
    """
    def __getitem__(self, idx):
        return Slerp(self.times[idx], self.rotations[idx])

    def __init__(self, times, q):
        if np.asarray(times).size == 0:
            self.__call__ = self.blank_call
            self.times = np.empty(0, np.double)
        else:
            super().__init__(times, q)

    def __call__(self, tarr):
        if np.asarray(tarr).size == 0:
            return Rotation(np.empty((0, 4), np.double))
        else:
            return super().__call__(tarr)


    """
    def __init__(self, times, q):
        if np.asarray(times).size == 0:
            super().__init__([-np.inf, -np.inf], Rotation([[0, 0, 0, 1], [0, 0, 0, 1]]))
        else:
            super().__init__(times, q)
    """

    def blank_call(self, times):
        raise ValueError("no interpolation provided")

    def __add__(self, other):
        """
        concatenate two set of quaternions, subject to interpolation
        """
        times = np.concatenate([self.times, other.times])
        idxs = np.argsort(times)
        quats = np.concatenate([self(self.times).as_quat(), other(other.times).as_quat()])[idxs]
        times = times[idxs]

        #TODO:
        # there are lots of errors cases to be considered
        # here erorr raised due to inconsistent quaternions:
        # i.e. time points are muxed, and rotations in them significantly different
        # but, one can imagine, that sparse points are completed with more
        idxmix = np.searchsorted(self.times, other.times)
        mask = np.logical_and(idxmix > 0, idxmix < self.times.size)
        print("quat crossing", mask.sum())
        if np.any(mask):
            if not np.allclose(self(other.times[mask]).as_quat(),
                               other(other.times[mask]).as_quat()):
                raise ValueError("concatenated quaternions are incompatible")

        tuniq, uidx = np.unique(times, return_index=True)
        return self.__class__(tuniq, Rotation(quats[uidx]))

    def __mul__(self, val):
        """
        returns the product of the quaterninos interpolation with provided quaternion

        ------
        Params:
            val: quaternion to be applied from the right side
        """
        return Slerp(self.times, self(self.times)*val)


class AttDATA(SlerpWithNaiveIndexing):
    """
    quaternions interpolation associated with
    """
    prec = 10./3600*pi/180.

    def __init__(self, *args, gti=None, hide_bad_interpolations=True, check_interpolation=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bt = np.array([])
        self.bq = np.array([])
        try:
            self.gti = GTI(gti)
        except Exception:
            self.gti = GTI(self.times[[0, -1]])
        if check_interpolation:
            if hide_bad_interpolations:
                # _chekc_interpolation_quality will find outliers
                bti = self._check_interpolation_quality()
                mask = (self.gti & ~bti).mask_external(self.times, True)
                # lets store outliers in a separate array
                self.bt, self.bq = self.times[~mask], self(self.times[~mask])
                #times, gaps = (self.gti & ~bti).make_tedges(self.times)
                times = self.times[mask]
                super().__init__(times, self(times))
                # check interpolation one more time to see wether new neighbouring
                # points provied good interpolation if no, exclude them from GTI
                bti = bti.remove_short_intervals(3.01)
                self.gti = self.gti & ~bti #self._check_interpolation_quality()
            else:
                self.gti = self.gti & ~self._check_interpolation_quality()
        else:
            self.bti = emptyGTI

    def _check_interpolation_quality(self):
        if self.gti.exposure == 0. or self.times.size == 2:
            return emptyGTI
        mgap = self.gti.mask_external((self.times[1:] + self.times[:-1])/2.)
        mgap2 = np.zeros(mgap.size + 1, np.bool)
        mgap2[1:] = ~mgap
        mgap2[:-1] = mgap2[:-1] | ~mgap
        dt = np.diff(self.times, 1)
        dt2 = self.times[2:] - self.times[:-2]
        rot = self(self.times)

        q = rot[1:]*rot[:-1].inv()
        qfov = Rotation.from_rotvec(q[:-1].as_rotvec()*(dt2/dt[:-1])[:, np.newaxis])*rot[:-2]
        qbak = Rotation.from_rotvec(-q[1:].as_rotvec()*(dt2/dt[1:])[:, np.newaxis])*rot[2:]
        mfov = np.zeros(self.times.size, np.bool)
        mbak = np.zeros(self.times.size, np.bool)
        mfov[2:] = np.sum(rot[2:].apply(OPAX)*qfov.apply(OPAX), axis=-1) < cos(pi/180.*10./3600.)
        mfov[2:] = mfov[2:] & mgap[1:]
        mbak[:-2] = np.sum(rot[:-2].apply(OPAX)*qbak.apply(OPAX), axis=-1) < cos(pi/180.*10./3600.)
        mbak[:-2] = mbak[:-2] & mgap[:-1]
        #mbad = mbak & mfov
        edges = medges(mfov[1:] & mbak[:-1]) # + [1, 0]
        edges = edges.reshape((-1, 2))
        return GTI(self.times[edges] + np.minimum(dt[edges], 1e-5)*[1, -1])


    @lru_cache
    def for_urdn(self, urdn):
        return self*get_boresight_by_device(urdn)

    def __add__(self, other):
        base = super().__add__(other)
        base.gti = self.gti | other.gti
        self._check_interpolation_quality()
        return base

    def _get_clean(self):
        mask = self.gti.mask_external(self.times)
        quats = self(self.times[mask])
        return self.__class__(self.times[mask], quats, self.gti)

    def apply_gti(self, gti, **kwargs):
        gti = GTI(gti)
        gti = gti & self.gti
        ts, mgaps = gti.make_tedges(self.times)
        quats = self(ts)
        return self.__class__(ts, quats, gti=gti, **kwargs)
        """
        print(gti.exposure)
        if gti.exposure == 0.:
            ret = self.__class__([-np.inf, -np.inf], Rotation([[0, 0, 0, 1], [0, 0, 0, 1]]))
            ret.gti = gti
            return  ret
        else:
            return self.__class__(ts, quats, gti=gti)
        """

    def get_axis_movement_speed_gti(self, query = lambda x: x < 100., ax=OPAX):
        """
        create a gti for the selected query based on the axis movement speed

        the trik is follows: slerp stors interpolation in form of rotations and rotvec
        the interpolation works like follows: roatation*Quat(rotvec*(t - t0)/dt))
        rotvec applied BEFORE the rotation, therefore, actual angular shift (not rotation aroud axis)
        is defined by the opax*rotvec/dt, this expression gives actual angular speed arcsec/sec
        of the defined axis
        """
        tc, dt, dalphadt = self.get_optical_axis_movement_speed(ax)
        qedges = medges(query(dalphadt)) + [0, -1]


        return GTI(tc[qedges] + dt[qedges]*[-0.5, 0.5])

    def get_optical_axis_movement_speed(self, ax=OPAX):
        """
        for provided gyrodata computes angular speed
        returns:
            ts - centers of the time bins
            dt - withd of the time bins
            dlaphadt - angular speed in time bin
        """
        if self.gti.exposure == 0.:
            return None, None, None
        te, mgaps = self.gti.make_tedges(self.times)
        tc = ((te[1:] + te[:-1])/2.)[mgaps]
        dt = (te[1:] - te[:-1])[mgaps]
        vecs = self(te).apply(ax)
        dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))[mgaps]/dt*180./pi*3600.
        return tc, dt, dalphadt

    def __mul__(self, val):
        nsrc = self.__new__(self.__class__)
        super(self.__class__, nsrc).__init__(self.times, self(self.times)*val)
        nsrc.gti = self.gti
        nsrc.bt = self.bt
        nsrc.by = self.bq
        return nsrc #self.__class__(self.times, self(self.times)*val, gti=self.gti)

    def __rmul__(self, val):
        """
        ahtung!!! scipy Rotations defines mul itself,
        so this method should be used explicitly in case
        if applied rotation is consequtive to one stored in attdata
        """
        return self.__class__(self.times, val*self(self.times), gti=self.gti)

    @classmethod
    def concatenate(cls, attlist, **kwargs):
        qlist = np.concatenate([att(att.times).as_quat() for att in attlist], axis=0)
        tlist = np.concatenate([att.times for att in attlist])
        tgti = reduce(lambda a, b: a | b, [att.gti for att in attlist])
        ut, uidx = np.unique(tlist, return_index=True)
        return cls(ut, Rotation(qlist[uidx]), gti=tgti, **kwargs)

    def set_nodes(self, te):
        te, mgaps = self.gti.make_tedges(te)
        return AttDATA(te, self(te), gti=self.gti)

    def circ_gti(self, vec, app=pi/180*100/3600., ax=OPAX):
        """
        te, gaps = self.gti.make_tedges(self.times)
        vc = self((te[1:] + te[:-1])/2.).apply(ax)
        mask = np.sum(vc*vec, axis=1) > cos(app)
        mask[~gaps] = False
        lgti = GTI(te[medges(mask)])
        """
        frac, gti = slerp_circ_aperture_exposure(self, vec, app, ax)
        return gti & self.gti

    def chull_gti(self, chull, ax=OPAX):
        gti = tGTI
        attl = copy(self)
        for v in chull.orts:
            gti = gti & attl.circ_gti(-v, 90.*3600 - 0.1, ax=OPAX)
            if gti.exposure == 0:
                break
            attl = attl.apply_gti(gti, check_interpolation=False, hide_bad_interpolations=False)
        return gti

    def get_covering_chulls(self, split_criteria = lambda x: [x.apply_gti(gti) for gti in get_observations_gtis(x, False)[0]], fov=None): # expandsize=pi/180.*26.5/60.):
        if fov is None:
            fov = raw_xy_to_vec(np.array([-12, 60, 60, -12]), np.array([-12, -12, 60, 60]))

        gtis, chulls = [], []
        ssegments = [c.expand(pi/180.*26.6/60.) for c in SPHERE.childs]
        for attloc in split_criteria(self):
            #vecs = np.concatenate([attloc(attloc.times).apply(v) for v in fov], axis=0)
            #return vecs
            #mtot = np.zeros(vecs.shape[0], bool)
            for ch in ssegments: #SPHERE.childs:
                #mask = ch.check_inside_polygon(vecs[~mtot])
                gloc = attloc.chull_gti(ch)
                if gloc.exposure > 0.:
                    gtis.append(gloc)
                    q = attloc(gloc.make_tedges(attloc.times)[0])
                    chloc = ConvexHullonSphere(np.concatenate([q.apply(v) for v in fov], axis=0)) & ch
                    chulls.append(chloc)
        print("report: ", [(ch.area, g.exposure) for ch, g in zip(chulls, gtis)])
        return list(zip(chulls, gtis)) if len(chulls) > 0 else [[], []]




cache_function = {np.ndarray: lambda x: x.tobytes(),
                  Urddata: lambda x: (x.urdn, x.size, x.filters.hash()),
                  fits.FITS_rec: lambda x: x.tobytes(),
                  AttDATA: lambda x: (x.times.tobytes(), x.gti.arr.tobytes(), x.rotations.as_rotvec().tobytes())}

def _urddata_lru_cache(function):
    cache = {}
    def newfunction(*args, **kwargs):
        try:
            lhash = hash(tuple(cache_function.get(type(x), lambda x: x)(x) for x in args) + \
                        tuple(cache_function.get(type(x), lambda x: x)(x) for x in kwargs.values()))
        except:
            return function(*args, **kwargs)
        else:
            if len(cache) <= 21: #typical size of different urddata subsets used in a single data reduction
                if not lhash in cache:
                    cache[lhash] = function(*args, **kwargs)
                return cache.get(lhash)
    return newfunction


def define_required_correction(attdata):
    """
    NAME
        define_required_correction


    Some of the orientation files are provided with fk5 for current epoch.
    Those should be corrected and alternate in the standard form -> to the FK5 J2000.
    The precise information on when to apply this corrections is stored in the CALDB files.

    """
    if (attdata.gti & gyrocorrectionbti).exposure > 0.:
        a1 = attdata.apply_gti((attdata.gti & gyrocorrectionbti) + [1.4, -1.4])
        a2 = attdata.apply_gti((attdata.gti & ~gyrocorrectionbti) + [1.4, -1.4])
        print(a1.gti.exposure, a2.gti.exposure, (a1.gti & a2.gti).exposure)
        return AttDATA.concatenate([a2, make_gyro_relativistic_correction(a1)]) #a2 + make_gyro_relativistic_correction(a1)
    else:
        return attdata


def lorentz_transform(speed, vec, beta):
    speed = normalize(speed)
    vec = normalize(vec)
    calpha = np.sum(speed*vec, axis=-1)
    alpha = np.arccos(calpha)
    return Rotation.from_rotvec(normalize(np.cross(vec, speed))*(alpha - np.arccos((calpha + beta)/(1 + beta*calpha)))[:, np.newaxis])

    #gamma = np.sqrt(1 - beta**2)
    #return Rotation.from_rotvec(normalize(np.cross(vec, speed))*np.arccos((1. + (gamma - 1)*calpha**2)/np.sqrt(1. + 2*(gamma - 1)*calpha**2 + (gamma - 1)**2*calpha**2.)))


def make_gyro_relativistic_correction(attdata):
    if attdata.gti.exposure == 0:
        return attdata
    print("inverse relativistic correction required")
    """
    vec = attdata(attdata.times).apply(OPAX)
    ra, dec = vec_to_pol(vec)
    ra, dec = np.rad2deg(ra), np.rad2deg(dec)
    sc = SkyCoord(ra, dec, unit=("deg", "deg"), frame="fk5", obstime=atime.Time(51543.875, format="mjd") + atime.TimeDelta(attdata.times, format="sec"))
    vec2 = np.asarray(sc.gcrs.cartesian.xyz.T)
    vrot = np.cross(vec2, vec)
    vrot = vrot/np.sqrt(np.sum(vrot**2, axis=1))[:, np.newaxis]
    calpha = np.sum(vec*vec2, axis=1)
    calphap2 = np.sqrt((calpha + 1.)/2.)
    salphap2 = np.sqrt((1. - calpha)/2.)
    #alpha = np.arccos(np.sum(vec*vec2, axis=1))
    qcorr = np.empty((calphap2.size, 4), np.double)
    qcorr[:, :3] = vrot*salphap2[:, np.newaxis]
    qcorr[:, 3] = calphap2
    return AttDATA(attdata.times, Rotation(qcorr).inv()*attdata(attdata.times), gti=attdata.gti)
    """

    sunfk5j2000 = pol_to_vec(*np.deg2rad(np.array([281.28189297, -23.03376005]).reshape((2, -1))))[0]
    enpfk5j2000 = pol_to_vec(*np.deg2rad(np.array([270.0142976, 66.56177014]).reshape((2, -1))))[0]
    secperyear = 31557600.0
    ebeta = 9.93529142270535e-05 #earth absolute movement speed in c unit
    tshift = (Time(MJDREF, format="mjd") - Time(datetime(2000, 1, 1, 12))).sec
    #attdata times provided in seconds since J2000 + 0.624 seconds
    speed = Rotation.from_rotvec(enpfk5j2000*(tshift + attdata.times)[:,np.newaxis]/secperyear*2.*pi).apply(normalize(np.cross(enpfk5j2000, sunfk5j2000)))
    qcorr = lorentz_transform(speed, attdata(attdata.times).apply([1, 0, 0]), ebeta)
    return AttDATA(attdata.times, qcorr*attdata(attdata.times), gti=attdata.gti)

#-===========================================================================================


def read_gyro_fits(gyrohdu):
    """
    reads gyro quaternion from fits file hdu and returns AttDATA  container

    -------
    Params:
        hdu: fits.BintableHDU containing gyrofiles data

    return:
        AttDATA container, which bares attitude information
    """
    gyrodata = gyrohdu.data
    quats = np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T
    times = gyrodata["TIME"]
    idx = np.argsort(times)
    times = times[idx]
    quats = quats[idx]
    mtime = np.median(times)
    masktimes = (times > mtime - 3.*24.*3600) & (times < mtime + 3.*24.*3600.)
    #print("diffs", np.sum(np.diff(times) < 1e-8))
    masktimes[1:] = np.diff(times) > 1e-8
    #masktimes = times > T0
    mask0quats = np.sum(quats**2, axis=1) > 0.
    mask = np.logical_and(masktimes, mask0quats)
    #print("mask result", mask.size, np.sum(mask))
    times, quats = times[mask], quats[mask]
    ts, uidx = np.unique(times, return_index=True)
    #print("unique", times.size, ts.size)
    ainit = AttDATA(ts, Rotation(quats[uidx])*qgyro0*get_boresight_by_device("GYRO"))
    return ainit


def read_bokz_fits(bokzhdu):
    """
    reads bokz quaternion from fits file hdu and returns AttDATA  container

    -------
    Params:
        hdu: fits.BintableHDU containing gyrofiles data

    return:
        AttDATA container, which bares attitude information
    """
    bokzdata = bokzhdu.data
    mat = np.array([[bokzdata["MOR%d%d" % (i, j)] for i in range(3)] for j in range(3)])
    mat = np.copy(mat.swapaxes(2, 1).swapaxes(1, 0)) #.astype(np.float16)
    matn = np.zeros(mat.shape, np.double)
    matn[:, :, :] = mat
    mat = matn
    mask0quats = np.linalg.det(mat) != 0.
    masktimes = bokzdata["TIME"] > T0
    q = Rotation.from_matrix(mat)
    #mask = np.logical_and(mask0quats, masktimes)
    mask = ~np.isnan(np.sum(q.as_rotvec()**2, axis=1))
    jyear = get_hdu_times(bokzhdu).jyear[mask]
    qbokz = earth_precession_quat(jyear).inv()*q[mask]*qbokz0*\
            get_boresight_by_device("BOKZ")
    ts, uidx = np.unique(bokzdata["TIME"][mask], return_index=True)
    return AttDATA(ts, qbokz[uidx])


def read_sed_fits(sedhdu):
    q0 = Rotation([0.1830127, 0.6830127, -0.1830127, 0.6830127])
    d = np.unique(np.copy(sedhdu.data))
    d = d[np.unique(d["TIME"], return_index=True)[1]]
    q = Rotation(np.array([d["QRVE%d" % i] for i in [0, 1,2 ,3]]).T)
    return AttDATA(d["TIME"], q*q0)

def get_raw_bokz(bokzhdu):
    """
    reads bokz quaternion from fits file hdu and returns AttDATA  container

    -------
    Params:
        hdu: fits.BintableHDU containing gyrofiles data

    return:
        times and quaternions stored in bokz
    """
    bokzdata = bokzhdu.data
    mat = np.array([[bokzdata["MOR%d%d" % (i, j)] for i in range(3)] for j in range(3)])
    mat = np.copy(mat.swapaxes(2, 1).swapaxes(1, 0))
    mask0quats = np.linalg.det(mat) != 0.
    masktimes = bokzdata["TIME"] > T0
    mask = np.logical_and(mask0quats, masktimes)
    qbokz = Rotation.from_matrix(mat[mask])*qbokz0
    jyear = get_hdu_times(bokzhdu).jyear[mask]
    return bokzdata["TIME"][mask], earth_precession_quat(jyear).inv()*qbokz

#@_urddata_lru_cache
def get_events_quats(urddata, URDN, attdata):
    return attdata(urddata["TIME"])*get_boresight_by_device(URDN)


#@_urddata_lru_cache
def make_align_quat(ax1, ax2, zeroax=np.array([-1, 0, 0]), north=np.array([0, 0, 1])):
    """
    this function provides with the quat which alignes inpute vector ax1 with zeroax
    and puts second provided vector in the plane within zeroax and north
    params:
        ax1, ax2 - two mandatory input vectors
        zeroax - default direction to align ax1 with (default is [-1, 0, 0] for azimuth angle to be in the vicinity of 180 rather 0, 360
        north - second vector defining with zeroax the plane on which ax1 and ax2 should lie after rotation
    """
    cross = normalize(np.cross(north, zeroax))
    if np.abs(np.sum(normalize(ax1)*zeroax)) == 1:
        q1 = Rotation.from_rotvec(north*(3/2. - np.sum(normalize(ax1)*zeroax)/2.)*pi)
    else:
        v = normalize(np.cross(ax1, zeroax)).reshape((-1, 3))
        q1 = Rotation.from_rotvec(v*np.arccos(np.sum(normalize(ax1)*zeroax, axis=-1))[..., np.newaxis])
    v2 = q1.apply(normalize(ax2)).reshape((-1, 3))
    v3 = normalize(v2 - v2*np.sum(zeroax*v2, axis=-1)[..., np.newaxis])
    #q2 = Rotation.from_rotvec(zeroax*np.arctan2(np.sum(cross*v3, axis=-1), np.sum(north*v3, axis=-1))[..., np.newaxis])
    q2 = Rotation.from_rotvec(zeroax*np.arctan2(np.sum(cross*v3, axis=-1), np.sum(north*v3, axis=-1))[..., np.newaxis])
    return q2*q1


#@_urddata_lru_cache
def get_photons_vectors(urddata, URDN, attdata, subscale=1, randomize=False):
    """
    return cartesian vectros, defining direction to the sky, for the specific pixel, defined with urddata rawx, rawy coordinates

    -------
    Params:
        urddata: event list stored in fits file
        URDN: id of the specific detection unit, which produced the provided event list
        attdata: spacecraft attitude information in form of the AttDATA
        subscale: split each pixel on subscale x subscale subpixels, vectors are provided for each pixel

    return: cartesian vectors in form of numpy array of shape(..., 3)
    """
    if not np.all(attdata.gti.mask_external(urddata["TIME"])):
        raise ValueError("some events are our of att gti")
    urdnatt = attdata.for_urdn(URDN)
    qall = urdnatt(np.repeat(urddata["TIME"], subscale*subscale))
    photonvecs = urd_to_vec(urddata, subscale, randomize)
    if photonvecs.size == 0:
        phvec = photonvecs
    else:
        phvec = qall.apply(photonvecs)
    return phvec

def add_ra_dec(urddata, urdn, attdata):
    ra, dec = np.rad2deg(get_photons_sky_coord(urddata, urdn, attdata))
    return np.lib.recfunctions.append_fields(udata, ["RA", "DEC"], [ra, dec], usemask=False)


#@_urddata_lru_cache
def get_photons_sky_coord(urddata, URDN, attdata, subscale=1, randomize=False):
    """
    converts eventlist event pixel information in to the ra and dec spherical coordinates of fk5 system

    --------
    Params:
        urddata: event list stored in fits file
        URDN: id of the specific detection unit, which produced the provided event list
        attdata: spacecraft attitude information in form of the AttDATA
        subscale: split each pixel on subscale x subscale subpixels, for each pixel coordinates are provided

    returns:
        ra and dec coordinates in fk5 system in radians
    """
    phvec = get_photons_vectors(urddata, URDN, attdata, subscale, randomize)
    return vec_to_pol(phvec)

def nonzero_quaternions(quat):
    mask = np.sum(quat**2, axis=1) > 0
    return mask

def quat_to_pol_and_roll(qfin, opaxis=[1, 0, 0], north=[0, 0, 1]):
    """
    it is assumed that quaternion is acting on the sattelite coordinate system
    in order to orient in in icrs coordinates
    opaxis - define dirrection of the main axis (x axis [1, 0, 0] coinside with optical axis)
    we assume that north is oriented along z coordinate, we also name this coordinate
    north for detectors


    --------
    Params:
        quat - a set of quaternions in form of scipy.spatial.transform.Rotation container
        opaxis - optical axis vector in sc frame
        north - axis relative to which roll angle will be defined

    return:
        ra, dec, roll - attitude fk5 coordinates (ra, dec) and roll angle relative to north axis ! note that all angles are returned in radians
    """
    vecs = normalize(qfin.apply(opaxis))
    dec = np.arcsin(vecs[:,2])
    ra = np.arctan2(vecs[:,1], vecs[:,0])


    yzprojection = normalize(np.cross(vecs, north))
    zprojection = normalize(np.array(north) - vecs*np.sum(vecs*north, axis=1)[:, np.newaxis])

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply([0, 0, 1]), axis=1),
                           np.sum(zprojection*qfin.apply([0, 0, 1]), axis=1))
    return ra, dec, rollangle

def ra_dec_roll_to_quat(ra, dec, roll, opaxis=[1, 0, 0], north=[0, 0, 1]):
    """
    inverse to quat to poll and roll
    take as input 2 polar coordinates and roll angle and produces corresponding quaternion

    for clarity in this implementation quaternion is produced with 3 consequtive rotation
    1) rotate opax orthoganaly to galactic north z
    2) rise opax towards galactic north
    3) rotate is around obtained dirrection for roll angle

    --------
    Params:
        quat - a set of quaternions in form of scipy.spatial.transform.Rotation container
        opaxis - optical axis vector in sc frame
        north - axis relative to which roll angle will be defined

    return:
        quaternions
    """
    vecs = pol_to_vec(*np.deg2rad([ra, dec])).reshape((-1, 3))
    north = np.array(north)
    proj = normalize(vecs - north*np.sum(vecs*north, axis=1)[:, np.newaxis])
    alpha = np.arctan2(proj[:, 1], proj[:, 0])
    qlon = Rotation.from_rotvec(north[np.newaxis, :]*alpha[:, np.newaxis])
    rr = normalize(np.cross(proj, north))
    qlat = Rotation.from_rotvec(rr*np.arcsin(np.sum(vecs*north, axis=1))[:, np.newaxis])
    qrot = Rotation.from_rotvec(vecs*np.deg2rad(roll)[:, np.newaxis])
    return qrot*qlat*qlon
    #return qlat*qlon


def earth_precession_quat(jyear):
    """
    taken from astropy.coordinates.earth_orientation
    contains standrard precession ephemerides, accepted at IAU 2006,
    didn't check but should work better then IAU76 version, which provide several mas precission upto 2040

    -------
    Params:
        jyear: time in form of JD in years

    returns:
        quaternion which rolls coordinate system according to the ephemerides from J2000 epoch to the provided date
    """
    T = (jyear - 2000.0) / 100.0
    pzeta = (-0.0000003173, -0.000005971, 0.01801828, 0.2988499, 2306.083227, 2.650545)
    pz = (-0.0000002904, -0.000028596, 0.01826837, 1.0927348, 2306.077181, -2.650545)
    ptheta = (-0.0000001274, -0.000007089, -0.04182264, -0.4294934, 2004.191903, 0)
    zeta = np.polyval(pzeta, T) / 3600.0
    z = np.polyval(pz, T) / 3600.0
    theta = np.polyval(ptheta, T) / 3600.0
    return Rotation.from_euler("ZYZ", np.array([z, -theta, zeta]).T, degrees=True)


def get_axis_movement_speed(attdata):
    """
    for provided gyrodata computes angular speed

    ------
    Params:
        attdata - attitude information in form of AttDATA container

    returns:
        ts - centers of the time bins
        dt - withd of the time bins
        dlaphadt - angular speed in time bin
    """
    te, mgaps = attdata.gti.make_tedges(attdata.times)
    tc = ((te[1:] + te[:-1])/2.)[mgaps]
    dt = (te[1:] - te[:-1])[mgaps]
    vecs = attdata(te).apply(OPAX)
    dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))[mgaps]/dt*180./pi*3600.
    return tc, dt, dalphadt

def get_earth_rot_quats(times, t0=None):
    """
    for provided times and T0 epoche computes rotation of the

    initial plan for this function was to pack wery densly attitude data information
    it was assumed that spacecraft rotates around axis, which connects is with the sun,
    this axis in turn rotates in solar plane with 1yr period.

    Unfortunately, it was found that the axis, arond which SC rotates, does not coincide with the sc-snu axis,
    also this axis changes with time, and its rotation speed also changes, so the function is useless for now

    -------
    Params:
        times - times for which compute rotation of frozen fk5 system due to the earth rotation around the sun

    return:
        quaternions for each provided time point which rotates fk5 frame
    """
    if t0 is None: t0 = times[0]
    q = np.empty(np.asarray(times).shape + (4, ), np.double)
    phase = 2.*pi/SECPERYR*(times - t0)
    sphase = np.sin(phase/2.)
    q[..., 0] = SOLARSYSTEMPLANENRMALEINFK5[0]*sphase
    q[..., 1] = SOLARSYSTEMPLANENRMALEINFK5[1]*sphase
    q[..., 2] = SOLARSYSTEMPLANENRMALEINFK5[2]*sphase
    q[..., 3] = np.cos(phase/2.)
    return Rotation(q)

def get_elongation_plane_norm(attdata, gti=None): #tGTI):
    """
    for the provided attdata, searches mean rotation plane of the provided attdata imformation
    note: it is assumed that the attdata achived due to the relatively constant motion in one dirrection

    -----------
    Params:
        attdata: attitude information for whichcompute mean rotation plane
        gti: good time interval for which find the plane

    returns:
        unit cartesian vector for found mean rotation plane
    """
    aloc = attdata.apply_gti(gti) if not gti is None else attdata
    tc, dt, dalphadt = get_axis_movement_speed(aloc)
    me = medges(dalphadt < 150) + [0, -1]
    gtisurv = GTI(tc[me] + dt[me]*[-0.5, 0.5])
    te, mgaps = gtisurv.make_tedges(np.linspace(aloc.times[0], aloc.times[-1], int((aloc.times[-1] - aloc.times[0])/300) + 3))
    mask = np.ones(te.size)
    qrot = aloc(te)
    rvec = (qrot[1:]*qrot[:-1].inv()).as_quat()[..., :3]
    rvec = rvec/np.sqrt(np.sum(rvec**2, axis=1))[:, np.newaxis]
    rvecm = np.sum(rvec, axis=0)
    rvecm = rvecm/sqrt(np.sum(rvecm**2.))
    return rvecm



class SlerpForward(object):
    def __init__(self, times, rotvecs, omega):
        self.times = times
        self.rotvecs = rotvecs
        self.omega = np.ones(self.times.size, np.double)*omega

    def __call__(self, ts):
        idx = np.searchsorted(self.times, ts) - 1
        print(idx, self.rotvecs.shape, self.omega.shape)
        return Rotation.from_rotvec(self.rotvecs[idx]*(self.omega[idx]*(ts - self.times[idx]))[:, np.newaxis])


class Survey(object):
    def __init__(self, opax, pole, opaxrot, gti, qcorr=None, zcorr=None, fcorr=None):
        self.opax = opax
        self.pole = pole
        self.opaxrot = opaxrot
        self.qcorr = qcorr if not qcorr is None else Rotation([0, 0, 0, 1])
        self.zcorr = zcorr if not zcorr is None else Rotation([0, 0, 0, 1])
        self.fcorr = fcorr if not fcorr is None else Rotation([0, 0, 0, 1])
        self.gti = gti

    def __call__(self, times):
        return (self.fcorr*self.pole(times)*self.zcorr*self.opaxrot(times)*self.qcorr).apply(self.opax[np.searchsorted(self.gti.arr[:, 0], times) - 1])

    @classmethod
    def read_surv_fits(cls, fname, fcorr=None, qcorr=None, zcorr=None, omegacorr=1., romegacorr=1.):
        ffile = fits.open(fname)
        surveys = np.copy(ffile[1].data)[ffile[1].data["TYPE"] == "SURVEY"]
        start = (Time([datetime.fromisoformat(t) for t in surveys["START"].astype(str)]) - \
                Time(51543.875, format="mjd")).sec
        #start = 6.42068102e+08 + (start - 6.42068102e+08)/omegacorr
        stop = (Time([datetime.fromisoformat(t) for t in surveys["STOP"].astype(str)]) - \
                Time(51543.875, format="mjd")).sec #- 3.*3600.
        print(start, stop)
        polus = pol_to_vec(surveys["RA_P"]*pi/180, surveys["DEC_P"]*pi/180).reshape((-1, 3))
        opax = polus
        omega = (surveys["Z_SPEED"]*pi/180./(24*3600.)*omegacorr).ravel()
        print(omega)
        print(polus.shape)
        phi = np.ones(start.size*2, np.double)*2.*pi
        phi[1::2] = 2.*pi + omega*(stop - start)
        print(Rotation.from_rotvec(np.repeat(polus, 2, axis=0)*phi[:, np.newaxis]))
        print(start - start[0], stop - start[0])
        print(np.array([start, stop]).T.ravel() - start[0])
        ts = np.sort(np.array([start + 1e-6, stop]).T.ravel() )
        print(ts[1:] - ts[:-1])
        #pole = Slerp(ts, Rotation.from_rotvec(np.repeat(polus, 2, axis=0)*phi[:, np.newaxis]))
        pole = SlerpForward(start, polus, omega)

        opaxrot = pol_to_vec(surveys["RA_Z0"]*pi/180, surveys["DEC_Z0"]*pi/180).reshape((-1, 3))
        phi[1::2] = 2.*pi + (stop - start)*2.*pi/(4.*3600.)
        #opaxrot = Slerp(ts, Rotation.from_rotvec(np.repeat(opaxrot, 2, axis=0)*phi[:, np.newaxis]))
        opaxrot = SlerpForward(start, opaxrot, 2.*pi/4./3600.*romegacorr)
        gti = GTI(np.array([start, stop]).T)
        return cls(opax, pole, opaxrot, gti, qcorr, zcorr, fcorr)





class SurveyAtt(object):
    """
    """
    def __init__(self, polus, omega, gtitstart, tstop, sc_start_rotvec, sc_romega, sc_start_vec, roll):
        """
        initialize survey att container:
        for a set (or single) tstart values, store survey pole, survey angular frequency, start rotating
        vector, optical axis rotating speed
        """
        self.tstart = np.asarray(tstart)
        """ start times for survey pices"""
        idx = np.argsort(self.tstart)
        self.tstart = self.tstart[idx]
        self.polus = np.asarray(polus)[idx].reshape((-1, 3))
        """ survey poluses"""
        self.omega = np.asarray(omega)[idx].ravel()
        """ survey rotational axis rotation around pole angular speed"""
        self.tstop = np.asarray(tstop)[idx]
        self.sc_start_vec = np.asarray(sc_start_vec)[idx].reshape((-1, 3))
        """ initial orientation of the spacecraft axis"""
        self.sc_start_rotvec = np.asarray(sc_start_rotvec)[idx].reshape((-1, 3))
        """ starting rotation vector of sc optical axis"""
        self.sc_romega = np.asarray(sc_romega)[idx]
        """ spacecraft optical axis rotation speed """
        self.roll = np.asarray(roll)[idx].ravel()

        self.gti = GTI(np.array([self.tstart, self.tstop]).T)

    def get_rotax(self, times):
        times = np.asarray(times)
        mask = self.gti.mask_external(times)
        if not np.all(mask):
            print("Warning!: some of the times are out of survey attdata")
        tloc = times[mask]
        print(tloc)
        idx = np.searchsorted(self.tstart, tloc) - 1
        print(idx[0])
        pole_phase = (tloc - self.tstart[idx])*self.omega[idx]
        r = Rotation.from_rotvec(self.polus[idx]*pole_phase[:, np.newaxis])
        return mask, r.apply(self.sc_start_rotvec[idx])

    def get_opax_vec(self, time):
        time = np.asarray(time)
        mask = self.gti.mask_external(time)
        if not np.all(mask):
            print("Warning!: some of the times are out of survey attdata")
        tloc = time[mask]
        idx = self.tstart.searchsorted(tloc) - 1
        pole_phase = (tloc - self.tstart[idx])*self.omega[idx]
        rotz = Rotation.from_rotvec(self.polus[idx]*pole_phase[:, np.newaxis])
        rot_phase = (tloc - self.tstart[idx])*self.sc_romega[idx]
        rotax = Rotation.from_rotvec(self.sc_start_rotvec[idx]*rot_phase[:, np.newaxis])
        return mask, (rotz*rotax).apply(self.sc_start_vec[idx])

    def get_quaterninos(self, time):
        time = np.asarray(time)
        mask = self.gti.mask_external(time)
        if not np.all(mask):
            print("Warning!: some of the times are out of survey attdata")
        tloc = time[mask]
        tloc = time[mask]
        idx = self.tstart.searchsorted(tloc) - 1
        pidx = self.polus[idx]

        pole_phase = (tloc - self.tstart[idx])*self.omega[idx]
        rotz = Rotation.from_rotvec(self.polus[idx]*pole_phase[:, np.newaxis])

        rot_phase = (tloc - self.tstart[idx])*self.sc_romega[idx]
        rotax = Rotation.from_rotvec(self.sc_start_rotvec[idx]*rot_phase[:, np.newaxis])

        rollq = Rotation.from_rotvec(np.array([1, 0, 0])[np.newaxis, :]*self.roll[idx, np.newaxis])

        rollp = Rotation.from_rotvec(np.array([0, 0, 1])[np.newaxis, :]*np.arctan2(self.polus[idx, 1], self.polus[idx, 0])[:, np.newaxis])
        pidx = np.copy(self.polus[idx])
        pidx[:, 2] = 0.
        pidx = pidx/np.sqrt(np.sum(pidx**2, axis=1))[:, np.newaxis]
        r = np.cross(self.polus[idx], pidx)
        n = np.sqrt(np.sum(r**2, axis=1))
        rolla = Rotation.from_rotvec(r*-(np.arcsin(n)/n)[:, np.newaxis])
        return rotz*rotax*rolla*rollp*rollq

    @classmethod
    def read_survey_fits(cls, ffile):
        surveys = np.copy(ffile[1].data)[ffile[1].data["TYPE"] == "SURVEY"]
        start = (Time([datetime.fromisoformat(t) for t in surveys["START"].astype(str)]) - \
                Time(51543.875, format="mjd")).sec - 3.*3600.
        stop = (Time([datetime.fromisoformat(t) for t in surveys["STOP"].astype(str)]) - \
                Time(51543.875, format="mjd")).sec - 3.*3600.
        polus = pol_to_vec(surveys["RA_P"]*pi/180, surveys["DEC_P"]*pi/180)
        sc_start_rotvec = pol_to_vec(surveys["RA_Z0"]*pi/180, surveys["DEC_Z0"]*pi/180)
        sc_start_vec = polus # Rotation.from_rotvec(sc_start_rotvec*-30*pi/180).apply(polus) # pol_to_vec(surveys["RA"]*pi/180, surveys["DEC"]*pi/180)
        sc_romega = np.ones(surveys.size, np.double)*((90. + 0.015/24.)/3600.*pi/180.)
        omega = surveys["Z_SPEED"]*pi/180./(24*3600.)
        roll = surveys["ROLL_ANGLE"]
        return cls(polus, omega, start, stop, sc_start_rotvec, sc_romega, sc_start_vec, roll)

def slerp_circ_aperture_exposure(slerp, loc, appsize, offvec=OPAX, mask=None):
    """
    let assume we have a set of interpolations of quaternions
    this interpolation define rotations from one quaternions to a next one
    after that we want to find which part of the trajectorie of vector offvec in rotation defined by interpolation slerp, will fall inside circular aperture around vector loc
    """
    if mask is None:
        mask = np.ones(slerp.timedelta.size, np.bool)

    offvec = normalize(np.asarray(offvec))
    """
    scipy slerp works like q_i (q_i^-1 q_i+1 omega dt)

    """
    frac = np.zeros(slerp.timedelta.size, np.double)

    rmod = np.sqrt(np.sum(slerp.rotvecs**2, axis=1))
    rvec = slerp.rotvecs[mask]/rmod[mask, np.newaxis]
    rvec[rmod < 1e-10] = offvec
    a0 = slerp.rotations[mask].apply(loc, inverse=True)  # location vector in the rotating coordinate system
    cosa = np.sum(rvec*offvec, axis=1)
    cosa[rmod < 1e-10] == 1.
    cosb = np.sum(rvec*a0, axis=1)
    cose = cos(appsize*pi/180/3600.)
    """
    despite the phase of a0 vec, in order to vector trajectorie to fall inside circula aperture following
    conditions should be satisfied
    alpha + epsilon < beta
    alpha - epsilon > beta
    """
    alpha = np.arccos(cosa)
    alpha[rmod < 1e-10] = 0.
    beta = np.arccos(cosb)
    """
    if alpha + beta < epsilon than the interpolation trajectroie is in the circular aperture despite the phase
    """
    malpha = alpha < pi/2.
    maskallinsideapp = np.zeros(malpha.size, bool)
    maskallinsideapp[malpha] = alpha[malpha] + beta[malpha] < appsize*pi/180/3600
    maskallinsideapp[~malpha] = (2.*pi - alpha[~malpha]) - beta[~malpha] < appsize*pi/180/3600
    frac[mask] = maskallinsideapp.astype(np.double)
    gtiallin = GTI(np.array([slerp.times[:-1][maskallinsideapp], slerp.times[1:][maskallinsideapp]]).T)

    maskoutofapp = np.logical_and(beta + appsize*pi/180/3600 > alpha,
                                  beta - appsize*pi/180/3600 < alpha)
    msimplecase = np.logical_and(maskoutofapp, ~maskallinsideapp) #maskoutofapp | ~maskallinsideapp
    cosa, cosb, rmod, rvec, a0 = [arr[msimplecase] for arr in (cosa, cosb, rmod, rvec, a0)]
    sinbsq = 1 - cosb**2
    #print("simple cases (all in or all out", msimplecase.size, msimplecase.sum())

    """
    iam interesting about the angles, between the sphere, described by the rotaion of offvec vector around rotvec
    and circle with appsize appsize aroung a0 vector
    the solution is defined by following equation
    (a0*v) = cosa
    (rvec*v) = cose


    a - is a vector, line in the plane containe both solutions (a[1,2]*a0 = cose and a[1,2]*rvec = cosa) and line between them
    a is directed along a1 + a2, which can be decomposed on two projectionss - rvec and (a0 - rvec*cosb)/sinb
    this projections are: rvec*(a1 + a2) = cosa
    and (a1 + a2)*(a0 - rvec*cosb)/sinb = (2cose - 2cosacosb)/sinb
    composing a from product of this projections and rvec and (a0 - rvec*cos)/sinb one will get non unit vector presented bellow
    the length of the vector is actually defined by the projection angle gamma of vectors a1 and a2 on a
    """
    a = rvec*((cosa - cose*cosb)/sinbsq)[:, np.newaxis] + \
        a0*((cose - cosa*cosb)/sinbsq)[:, np.newaxis]
    cgammasq = np.sum(a**2, axis=1)

    aort = normalize(np.cross(rvec, a0)) #*np.sign(cose - cosa*cosb)
    port = normalize(np.cross(rvec, offvec)) #/sina[:, np.newaxis]
    pdir = normalize(offvec - rvec*cosa[:, np.newaxis]) #/sina[:, np.newaxis]
    """
    a1 and a2 - two vectors in the crosssection of the circles around rotation vector and loc
    """

    ap = a - rvec*cosa[:, np.newaxis]
    a1 = ap - aort*np.sqrt(1. - cgammasq)[:, np.newaxis]
    dphi = 2.*np.arccos(np.sum(normalize(a0 - cosb[:, np.newaxis]*rvec)*normalize(a1), axis=1))
    """
    phi1 and phi2 - are angles, we should to rotate offvec in order to get in to the epsilon vicinity of loc
    """
    phi1 = np.arctan2(np.sum(a1*port, axis=1), np.sum(a1*pdir, axis=1))
    phi1[phi1 < 0] = phi1[phi1 < 0.] + 2.*pi
    phis = np.maximum(phi1 + dphi, 2.*pi) - 2.*pi

    m2 = np.copy(mask)
    m2[m2] = msimplecase
    t1 = slerp.times[:-1][m2] + slerp.timedelta[m2]*np.minimum(phis, rmod)/rmod
    t2 = slerp.times[:-1][m2] + slerp.timedelta[m2]*np.minimum(phi1, rmod)/rmod
    t3 = slerp.times[:-1][m2] + slerp.timedelta[m2]*np.minimum(phi1 + dphi, rmod)/rmod
    g1 = GTI(np.array([slerp.times[:-1][m2], t1]).T)
    g2 = GTI(np.array([t2, t3]).T)
    #print("gtis", g1.exposure, g2.exposure, gtiallin.exposure)
    gti = GTI(np.array([slerp.times[:-1][m2], t1]).T) | GTI(np.array([t2, t3]).T) | gtiallin
    gti.merge_joint()

    frac[m2] = (t1 - slerp.times[:-1][m2] + t3 - t2)/slerp.timedelta[m2] #np.maximum((np.minimum(phi2, rmod) - np.maximum(phi1, 0)), 0)/rmod #*slerp.timedelta[m2]
    return frac, gti


def minimize_norm_to_survey(attdata, rpvec):
    """
    for provided attdata find vector, for which minimize SUM (vrot*opax)^2 where OPAX is mean optical axis oriented with attitude in fk5 frame


    -------
    Params:
        attdata - orientation information
        rpvec - initial gues on the normal to the rotaion plane

    return:
        unit vector, which provided minimal SUM (vrot*opax)^2
    """
    phi, theta = vec_to_pol(rpvec)
    vecs = attdata(attdata.times).apply(OPAX)
    res = minimize(lambda val: np.sum(np.sum(vecs*[cos(val[1])*cos(val[0]), cos(val[1])*sin(val[0]), sin(val[1])], axis=1)**2), [phi, theta])
    return pol_to_vec(*res.x)

def align_with_z_quat(vec):
    """
    for given vector, provides quaternion, which puts this vector  along z axis [0, 0, 1] with shortest possible trajectory

    -------
    Params:
        cartesian vector

    returns:
        quaternion in the scipy.spatial.transform.Rotation container
    """
    vec = vec/sqrt(np.sum(vec**2.))
    vrot = np.cross(vec, [0, 0, 1])
    vrot = vrot/sqrt(np.sum(vrot**2.))
    alpha = np.arccos(vec[2])
    q = np.empty(4, np.double)
    q[:3] = vrot*sin(alpha/2.)
    q[3] = cos(alpha/2.)
    return Rotation(q)

def get_attdata(fname, atshift=0., **kwargs):
    ffile = fits.open(fname)
    if "gyro" in fname:
        attdata = read_gyro_fits(ffile["ORIENTATION"])
        tshift = get_device_timeshift("gyro")
    elif "bokz" in fname:
        attdata = read_bokz_fits(ffile["ORIENTATION"])
        tshift = get_device_timeshift("bokz")
    elif "sed1" in fname:
        attdata = read_sed_fits(ffile["ORIENTATION"])
        tshift = 0.
    elif "RA" in ffile[1].data.dtype.names:
        tshift = get_device_timeshift("gyro")
        d = ffile[1].data
        attdata = AttDATA(d["TIME"], ra_dec_roll_to_quat(d["RA"], d["DEC"], d["ROLL"])*get_boresight_by_device("GYRO"))

    attdata.times = attdata.times - tshift + atshift
    attdata.gti.arr = attdata.gti.arr - tshift + atshift
    if "gyro" in fname:
        attdata = define_required_correction(attdata)
    return attdata


def attdata_for_urd(att, urdn):
    return att.for_urdn(urdn) #*get_boresight_by_device(urdn)

def get_slews_gti(attdata):
    #slew is usually performed at 240"/sec speed, we naively check the times when optical axis movement speed reaches over 100"/sec
    slews = attdata.get_axis_movement_speed_gti(lambda x: x > 100.)
    #the 100"/sec speed is reached after 30 sec of acceleration
    slews = slews.remove_short_intervals(30.)
    slews = slews + [-30, 30]
    return slews

def get_observations_gtis(attdata, join=True, intpsplitside=100.):
    '''
    for provided attitude (in the form of AttDATA container)
    computes segments separted by slew motion (which is determined as episodes when optical axis angular movement speed is greater then 100 arcsec/sec)
    additionanl parameters are

    join:
        join segments, separated by slews, if the zones covered by ART-XC FoV in this segments overlap

    intpsplitside:
        AttDATA can be composed of different pieces, which are not separated by slew but covers differeny part of the sky
        this parameter determine, which time gap in the data should be considered as a sign to split observation
        default: 100s (in GYRO data 2 and 3 seconds time gaps are prety common, but 10 s is almost unseen)

    '''
    slews = get_slews_gti(attdata)
    v = raw_xy_to_vec(np.array([0, 0, 48, 48]), np.array([0, 48, 48, 0]))
    chulls = []
    gtis = []
    sgti = ~(slews | (~attdata.gti).remove_short_intervals(intpsplitside))
    print("sgti exposure", sgti.exposure)
    for i, arr in enumerate(sgti.arr):
        attl = attdata.apply_gti(GTI(arr), check_interpolation=False)
        print("check", i, arr[1] - arr[0], attl.gti.exposure)
        if attl.gti.exposure == 0.:
            continue
        chull = ConvexHullonSphere(np.concatenate([attl.rotations.apply(v1) for v1 in v]))
        chulls.append(chull)
        gtis.append(attl.gti)

    clusters = np.arange(len(chulls))

    if join:
        clusters = np.arange(len(chulls))
        for i in range(len(chulls) - 1):
            ch1 = chulls[i]
            for j in range(i + 1,len(chulls)):
                ch2 = chulls[j]
                if np.any(ch1.check_inside_polygon(ch2.vertices)) or np.any(ch2.check_inside_polygon(ch1.vertices)):
                    clusters[(clusters == clusters[i]) | (clusters == clusters[j])] = min(clusters[i], clusters[j])

        gtis = [attdata.gti & GTI((~slews & attdata.gti).arr[clusters == cluster]) for cluster in np.unique(clusters)]
        chulls = [ConvexHullonSphere(np.concatenate([ch.vertices for ch, cl in zip(chulls, clusters) if cl == cluster], axis=0)) for cluster in np.unique(clusters)]
    return gtis, [ch.area for ch in chulls], [np.rad2deg(vec_to_pol(ch.get_center_of_mass())) for ch in chulls], chulls


class ObsClusters(object):
    """
    this class can accummulates limitless amount of the attdata (unless its cover more then half of a sky in a single covering convex hull)
    the main method of class is add (attdata)
    u can add attdata, which will be splited on separate observations separated by slew episodes (with optical axis movement speed > 100 arcsec/sec)
    for each such segments gti and covering convex hull is produced, convex hull are then joined if intersecting (gtis are joining too)
    one can get convex hulls and corresponding gtis as an expicit attributes of the class
    """
    def __init__(self, join=True, contour=None):
        self.clusters = []
        self.chulls = []
        self.gtis = []
        self._join = join
        self.counter = -1
        if contour is None:
            self._contour = raw_xy_to_vec(np.array([-6, -6, 53, 53]), np.array([-6, 51, 51, -6]))
        else:
            if not (type(contour) is np.array) or (contour.ndim > 2) or (contour.shape[-1] != 3):
                raise ValueError("contour is a set of vectors at the edges of convex hull, which incapsulates convolution core")
            self._contour = np.copy(contour)

    @property
    def totgti(self):
        return emptyGTI if len(self.gtis) == 0 else reduce(lambda a, b: a | b, self.gtis)


    def collapse(self, clist):
        print("collapse clist", clist)
        clist = sorted(clist)
        gnew = reduce(lambda a, b: a | b, [self.gtis[i] for i in clist])
        cnew = ConvexHullonSphere(np.concatenate([self.chulls[i].vertices for i in clist]))
        clidx = reduce(lambda a, b: a | b, [self.clusters[i] for i in clist])
        for idx in clist[::-1][:-1]:
            print("pop", idx, self.clusters[idx], self.gtis[idx])
            self.clusters.pop(idx)
            self.chulls.pop(idx)
            self.gtis.pop(idx)
        print("replace", cnew, gnew, clidx)
        self.chulls[clist[0]] = cnew
        self.gtis[clist[0]] = gnew
        self.clusters[clist[0]] = clidx

    def append(self, chull, gti, idx):
        self.chulls.append(chull)
        self.gtis.append(gti)
        self.clusters.append({idx,})

    def add(self, attdata, key=None):
        self.counter += 1
        attloc = attdata.apply_gti(~self.totgti)
        slews = get_slews_gti(attloc)

        if self._join:
            for i, arr in enumerate((~slews & attloc.gti).arr):
                gti = GTI(arr)
                m = gti.mask_external(attloc.times[:-1])
                chull = ConvexHullonSphere(np.concatenate([attloc.rotations[m].apply(v1) for v1 in self._contour] + [attloc(arr[0]).apply(self._contour),] + [attloc(arr[-1]).apply(self._contour),]))
                if not cloc is None:
                    self.add_chulls(chull, gti)

    def add_chulls(self, chull, gti):
        clist = []
        for k, ch in enumerate(self.chulls):
            if ch.intersect(chull):
                clist.append(k)
        clist.append(len(self.chulls))
        self.append(chull, gti, self.counter)
        if len(clist) > 1:
            self.collapse(clist)

class ChullGTI(ConvexHullonSphere):
    def __init__(self, vertices, parent=None, gti=emptyGTI):
        self.gti = gti
        super().__init__(vertices, parent)

    def update_parent_gti(self):
        self.gti = reduce(lambda a, b: a | b, [ch.gti for ch in self.childs])
        if self.parent != self:
            self.parent.update_parent_gti()

    def update_gti_for_attdata(self, attdata, expandsize=pi/180.*26.5/60.):
        self.gti = self.gti | attdata.chull_gti(self.expand(expandsize))
        if self.parent != self:
            self.parent.update_parent_gti()


class FullSphereChullGTI(ChullGTI):
    def __init__(self):
        self.childs = [ChullGTI(np.roll(CVERTICES, -i, axis=0)[:3], self) for i in range(4)]
        self.parent = self
        self.vertices = np.empty((0, 3), float)

    @property
    def area(self):
        return 4.*pi*(180/pi)**2.

    def check_inside_polygon(self, vecs):
        return np.ones(vecs.shape[0], bool)

    def __and__(self, other):
        return ChullGTI(other.vertices)


def get_linear_and_rot_components(vrot, ax=OPAX):
    """
    returns linear movement and rotation between two quaternions for specified axis (along which rotation need to be computed)
    quaternion qvals[1:].inv()*qvals[:-1] provides quaternino in original system of coordinate

    lets assume that the rotation vector is oriented along z in this system, and axis has coordinates ax = (cosp, sinp, 0)
    then rotated vector would have coordinates axr = (cosp, sinp cosa, sinp sina) so (vrot ax) = cosp
    and linear rotation between two vectors is a scale prodact of two
    coslinear = cosp**2 + sinp**2*cosa
    the rotational angle is defined by the angle between vectors, tangent to shortes trajectory, and one defined by quaternion
    cos(rot/2) = ([vrot ax] [vort ax]) / sinp      (here (vort ax) = 0, |[vort ax]| = 1)
    where 1/sinp appear due to the necessity to normalize  [vrot ax]
    vort itself can be defined as  vort = [ax axr]/sqrt(1 - coslinear**2)
    [vrot ax] = -[ax vrot] = -[ax [ax axr]]/sina = - ax coslinear/sina + axr/sina
    ([vrot ax] [vort ax]) = ([vrot ax] axr)/sina
    axr can be defined as follows axr = cosp vrot + (ax - cosp vrot)*cosa + [vrot ax] sina
    therefore
    ([vrot ax] axr) = ([vrot ax])**2 sina = sinp**2 sina
    finally
    cos(rot/2) = sinp*sina/sqrt(1 - coslinear**2) (1)

    there is specific case, which is not well handled by machine precision: coslinear -> 1
    this case can appear in two cases (moda << 1 & cosp << 1) and sinp << 1

    if cosp ->  1 (sinp << 1)  rot -> moda
    if moda -> 0 then cos(rot/2) -> 1 (first order decomposition) and 1 - moda^2 (1/3 - 1/8*sinp^2) therefore rot -> moda*sqrt(4/3 - sinp^2)
    we will approximate both cases with moda*np.sqrt(1 - cosp**2)
    """
    #qrot = (qvals[1:].inv()*qvals[:-1])
    #vrot = qrot.as_rotvec()
    moda = np.sqrt(np.sum(vrot**2, axis=1))
    vrot = vrot/moda[:, np.newaxis]
    cosp = np.sum(vrot*ax, axis=1)
    cosp[moda == 0] = 0.
    sasq = (1 - cosp**2.)
    coslinear = (cosp**2 + sasq*np.cos(moda))
    cosrot = np.empty(coslinear.size, float)
    mask = np.abs(coslinear) > 0.99 # for this condition linear components coluld not be greater then 0.01*pi, therefore a simple approximate estimation of rotation can be used to obtain rotaion angle with ~1% accuracy
    cosrot[mask] = np.abs(moda[mask])*cosp[mask] ##naive approximation to the second order decomposiition of case moda->0
    """
    print("maxcosrot", cosrot[mask].max(), coslinear[mask])
    print(cosp[~mask], coslinear[~mask], np.sin(moda[~mask]))
    print(cosp[~mask].max(), cosp[~mask].min(), coslinear[~mask].min(), coslinear[~mask].max(), moda[~mask].min(), moda[~mask].min())
    """
    #cr = np.sqrt((1. - cosp[~mask]**2.)/(1. - coslinear[~mask]**2.))*np.sin(moda[~mask])
    #print("crmin and max", cr.max(), cr.min())
    cosrot[~mask] = np.arccos(np.minimum(np.sqrt((1. - cosp[~mask]**2.)/(1. - coslinear[~mask]**2.))*np.sin(moda[~mask]), 1.))*2.
    return np.arccos(coslinear), cosrot


def split_in_two(vectors, times, att, precision=pi/180.*10./3600):
    dcalpha = np.sum(vectors*Slerp(times[[0, -1]], att(times[[0, -1]]))(times).apply([1, 0, 0]), axis=-1)
    return np.any(dcalpha < cos(precision))

def linear_segments(vectors, times, att, idx0=0, idx=None, precision=pi/180.*10./3600.):
    if idx is None:
        idx = [0, vectors.shape[0]]
    if times.size > 2 and split_in_two(vectors, times, att, precision):
        s = times.size//2
        idx.append(s + idx0)
        linear_segments(vectors[:s+1], times[:s+1], att, idx0, idx, precision)
        linear_segments(vectors[s:], times[s:], att, idx0 + s, idx, precision)
    return sorted(idx)

def pack_attdata(att, precision):
    """
    pack attitude data in a way, optical axis position can be restored with 5 arcsec accuracy
    """
    acomp = []
    for te, ts in att.gti.arr:
        gloc = GTI([te, ts])
        t, gaps = gloc.make_tedges(att.times)
        vecs = att(t).apply([1, 0, 0])
        idx = np.unique(linear_segments(vecs, t, idx=[0, t.size - 1], att=att, precision=precision))
        aloc = AttDATA(t[idx], att(t[idx]), gti=gloc, check_interpolation=False)
        aloc.gti = gloc
        acomp.append(aloc)
    return AttDATA.concatenate(acomp, check_interpolation=False) #hide_bad_interpolations=False)


def add_new_compressed_att(att, times, names, fname):
    print(fname)
    if fname.rstrip() in names:
        return att, times, names
    try:
        atl = arttools.orientation.get_attdata(fname.rstrip()) #.apply_gti(arttools.time.tGTI if att is None else ~(att.gti + [-3, 3]))
    except:
        print("can't read", fname)
    else:
        if not att is None:
            print("gti crosssecion", atl.gti & att.gti, atl.times[0] - att.times[-1])
        print(fname)
        if att is None:
            att = pack_attdata(atl, pi/180.*5./3600.)
        else:
            try:
                if (arttools.time.GTI(att.gti.arr[[0, -1],[0, 1]]) & atl.gti).exposure > 0.:
                    atl = arttools.orientation.AttDATA.concatenate([atl, att.apply_gti(arttools.time.GTI(atl.gti.arr[[0, -1],[0, 1]]), check_interpolation=False)])
                att = arttools.orientation.AttDATA.concatenate([att.apply_gti(~atl.gti, check_interpolation=False), pack_attdata(atl,pi/180.*5./3600.)], check_interpolation=False)
            except Exception:
                print("fail to concat with %s" % fname)
            else:
                print(type(times), len(times))
                times.append(atl.gti.arr[:,0])
                names = names + [fname.rstrip()]*atl.gti.arr.shape[0]
    return att, times, names

def make_mock_att_for_tempalte(template, tshift = 0, leftquat=Rotation([1, 0, 0, 0]), rightquat=Rotation([1, 0, 0, 0])):
    """
    its expexted that template contains rotation in spacecract coordinate system
    """
    vlist = template[:, 1:4]
    r = Rotation.from_rotvec(vlist*template[:, 4, np.newaxis]*pi/180.)
    rlist = [Rotation([1, 0, 0, 0]),]
    for rl in r:
        rlist.append(rlist[-1]*rl)
    r2 = Rotation(np.array([rl.as_quat() for rl in rlist]))
    te = np.concatenate([[0,], template[:, 0]])
    return AttDATA(te + tshift, leftquat*r2*rightquat, check_interpolation=False)
