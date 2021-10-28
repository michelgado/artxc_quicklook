from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin, sqrt
from ._det_spatial import urd_to_vec, F, DL
from .time import get_hdu_times, GTI, tGTI, emptyGTI
from .caldb import T0, get_boresight_by_device, get_device_timeshift
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
qgyro0 = Rotation([0., 0., 0., 1.])
OPAX = np.array([1, 0, 0])

SECPERYR = 3.15576e7
SOLARSYSTEMPLANENRMALEINFK5 = np.array([-9.83858346e-08, -3.97776911e-01,  9.17482168e-01])

def to_2pi_range(val): return val%(2.*pi)


gyrocorrectionbti = GTI([[624390347, 624399808], [6.24410643e+08, 6.30954575e+08]])
gyrocorrectionbti = emptyGTI #GTI([[624390347, 624399808], [6.24410643e+08, 6.30954575e+08]])

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
        #print("quat crossing", mask.sum())
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

    def __init__(self, *args, gti=None, hide_bad_interpolations=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.bt = np.array([])
        self.bq = np.array([])
        try:
            self.gti = GTI(gti)
        except Exception:
            self.gti = GTI(self.times[[0, -1]])
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
            bti.remove_short_intervals(3.)
            self.gti = self.gti & ~bti #self._check_interpolation_quality()
        else:
            self.gti = self.gti & ~self._check_interpolation_quality()

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
        """
        print(edges.shape)
        for s, e in edges.reshape((-1, 2)):
            radec = vec_to_pol(self.rotations[s-2:e+3].apply([1, 0, 0]))
            plt.plot(radec[0]*180/pi, radec[1]*180/pi)
            plt.scatter(radec[0][mbak[s-2:e+3]]*180/pi, radec[1][mbak[s-2:e+3]]*180/pi, marker="<", s=50, color="r")
            plt.scatter(radec[0][mfov[s-2:e+3]]*180/pi, radec[1][mfov[s-2:e+3]]*180/pi, marker=">", s=50, color="g")
            radec = vec_to_pol(self.rotations[[s, e]].apply([1, 0, 0]))
            plt.scatter(radec[0]*180/pi, radec[1]*180/pi, marker="x", s=40, color="k")
            plt.title(self.times[e] - self.times[s])
            plt.show()
        print("check interpolation", mfov.sum(), mbak.sum(), mfov.size)
        idx = np.argwhere(mfov)
        print(idx)
        for i in idx[:, 0]:
            print(mbak[i-2:i+3], mfov[i-2:i+3], mfov[i-2:i+2] & mbak[i-1:i+3])
            radec = vec_to_pol(self.rotations[i-6:i+7].apply([1, 0, 0]))
            plt.plot(radec[0]*180/pi, radec[1]*180/pi)
            plt.scatter(radec[0]*180/pi, radec[1]*180/pi, marker="+", color="k")
            plt.scatter(radec[0][mbak[i-6:i+7]]*180/pi, radec[1][mbak[i-6:i+7]]*180/pi, marker="<", s=30, color="r")
            plt.scatter(radec[0][mfov[i-6:i+7]]*180/pi, radec[1][mfov[i-6:i+7]]*180/pi, marker=">", s=30, color="g")
            plt.plot(radec[0][mgap2[i-6:i+7]]*180/pi, radec[1][mgap2[i-6:i+7]]*180/pi, "r", lw=2)
            plt.show()
        g1 = GTI(self.times[edges])
        g2 = GTI(self.times[edges] + np.minimum(dt[edges], 1e-5)*[1, -1])
        print(g1.exposure, g2.exposure)
        pause
        """
        return GTI(self.times[edges] + np.minimum(dt[edges], 1e-5)*[1, -1])

    def __add__(self, other):
        base = super().__add__(other)
        base.gti = self.gti | other.gti
        self._check_interpolation_quality()
        return base

    def _get_clean(self):
        mask = self.gti.mask_external(self.times)
        quats = self(self.times[mask])
        return self.__class__(self.times[mask], quats, self.gti)

    def apply_gti(self, gti):
        gti = GTI(gti)
        gti = gti & self.gti
        ts, mgaps = gti.make_tedges(self.times)
        quats = self(ts)
        return self.__class__(ts, quats, gti=gti)
        """
        print(gti.exposure)
        if gti.exposure == 0.:
            ret = self.__class__([-np.inf, -np.inf], Rotation([[0, 0, 0, 1], [0, 0, 0, 1]]))
            ret.gti = gti
            return  ret
        else:
            return self.__class__(ts, quats, gti=gti)
        """

    def get_axis_movement_speed_gti(self, query = lambda x: x < pi/180.*100/3600, ax=OPAX):
        """
        create a gti for the selected query based on the axis movement speed

        the trik is follows: slerp stors interpolation in form of rotations and rotvec
        the interpolation works like follows: roatation*Quat(rotvec*(t - t0)/dt))
        rotvec applied BEFORE the rotation, therefore, actual angular shift (not rotation aroud axis)
        is defined by the opax*rotvec/dt, this expression gives actual angular speed arcsec/sec
        of the defined axis
        """
        rmod = np.sqrt(np.sum(self.rotvecs**2, axis=1))
        proj = np.sqrt(1. - (np.sum(ax*self.rotvecs, axis=1)/rmod)**2.)
        vspeed = rmod*proj/self.timedelta
        return GTI(self.times[medges(query(vspeed))])

    def get_optical_axis_movement_speed(self):
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
        vecs = self(te).apply(OPAX)
        dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))[mgaps]/dt*180./pi*3600.
        return tc, dt, dalphadt

    def __mul__(self, val):
        return self.__class__(self.times, self(self.times)*val, gti=self.gti)

    def __rmul__(self, val):
        """
        ahtung!!! scipy Rotations defines mul itself,
        so this method should be used explicitly in case
        if applied rotation is consequtive to one stored in attdata
        """
        return self.__class__(self.times, val*self(self.times), gti=self.gti)

    @classmethod
    def concatenate(cls, attlist):
        qlist = np.concatenate([att(att.times).as_quat() for att in attlist], axis=0)
        tlist = np.concatenate([att.times for att in attlist])
        tgti = reduce(lambda a, b: a | b, [att.gti for att in attlist])
        ut, uidx = np.unique(tlist, return_index=True)
        return cls(ut, Rotation(qlist[uidx]), gti=tgti)

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
        gti = reduce(lambda a, b: a & b, [self.circ_gti(-v, 90.*3600. - 0.1, ax=OPAX) for v in chull.orts])
        return gti


cache_function = {np.ndarray: lambda x: x.tobytes(),
                  fits.FITS_rec: lambda x: x.tobytes(),
                  AttDATA: lambda x: (x.times.tobytes(), x.gti.arr.tobytes(), x.rotations.as_rotvec().tobytes())}

def _urddata_lru_cache(function):
    cache = {}
    def newfunction(*args, **kwargs):
        lhash = hash(tuple(cache_function.get(type(x), lambda x: x)(x) for x in args) + \
                     tuple(cache_function.get(type(x), lambda x: x)(x) for x in kwargs.values()))
        if len(cache) <= 21: #typical size of different urddata subsets used in a single data reduction
            if not lhash in cache:
                cache[lhash] = function(*args, **kwargs)
            return cache.get(lhash)
        else:
            return function(*args, **kwargs)
    return newfunction

def normalize(vecs):
    return vecs/np.sqrt(np.sum(vecs**2, axis=-1))[..., np.newaxis]

def define_required_correction(attdata):
    """
    NAME
        define_required_correction


    Some of the orientation files are provided with fk5 for current epoch.
    Those should be corrected and alternate in the standard form -> to the FK5 J2000.
    The precise information on when to apply this corrections is stored in the CALDB files.

    """
    a1 = attdata.apply_gti((attdata.gti & gyrocorrectionbti) + [0.4, -0.4])
    a2 = attdata.apply_gti((attdata.gti & ~gyrocorrectionbti) + [0.4, -0.4])
    return a2 + make_gyro_relativistic_correction(a1)

def make_gyro_relativistic_correction(attdata):
    if attdata.gti.exposure == 0:
        return attdata
    print("inverse relativistic correction required")
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
#-===========================================================================================

def vec_to_pol(phvec):
    """
    given the cartesian vectors produces phi and theta coordinates in the same frame
    """
    dec = np.arctan(phvec[...,2]/np.sqrt(phvec[...,0]**2. + phvec[...,1]**2.))
    ra = (np.arctan2(phvec[...,1], phvec[...,0])%(2.*pi))
    return ra, dec

def pol_to_vec(phi, theta):
    """
    given the spherical coordinates phi and theta produces cartesian vector
    """
    vec = np.empty((tuple() if not type(theta) is np.ndarray else theta.shape) + (3,), np.double)
    vec[..., 0] = np.cos(theta)*np.cos(phi)
    vec[..., 1] = np.cos(theta)*np.sin(phi)
    vec[..., 2] = np.sin(theta)
    return vec


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
    masktimes = times > T0
    mask0quats = np.sum(quats**2, axis=1) > 0.
    mask = np.logical_and(masktimes, mask0quats)
    times, quats = times[mask], quats[mask]
    ts, uidx = np.unique(times, return_index=True)
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
    mat = np.copy(mat.swapaxes(2, 1).swapaxes(1, 0))
    mask0quats = np.linalg.det(mat) != 0.
    masktimes = bokzdata["TIME"] > T0
    mask = np.logical_and(mask0quats, masktimes)
    jyear = get_hdu_times(bokzhdu).jyear[mask]
    qbokz = earth_precession_quat(jyear).inv()*Rotation.from_matrix(mat[mask])*qbokz0*\
            get_boresight_by_device("BOKZ")
    ts, uidx = np.unique(bokzdata["TIME"][mask], return_index=True)
    return AttDATA(ts, qbokz[uidx])

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

@_urddata_lru_cache
def get_events_quats(urddata, URDN, attdata):
    return attdata(urddata["TIME"])*get_boresight_by_device(URDN)


@_urddata_lru_cache
def make_align_quat(ax1, ax2, zeroax=np.array([-1, 0, 0]), north=np.array([0, 0, 1])):
    """
    this function provides with the quat which alignes inpute vector ax1 with zeroax
    and puts second provided vector in the plane within zeroax and north
    params:
        ax1, ax2 - two mandatory input vectors
        zeroax - default direction to align ax1 with
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


@_urddata_lru_cache
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
    qall = attdata(np.repeat(urddata["TIME"], subscale*subscale))*get_boresight_by_device(URDN)
    photonvecs = urd_to_vec(urddata, subscale, randomize)
    phvec = qall.apply(photonvecs)
    return phvec

def add_ra_dec(urddata, urdn, attdata):
    ra, dec = np.rad2deg(get_photons_sky_coord(urddata, urdn, attdata))
    return np.lib.recfunctions.append_fields(udata, ["RA", "DEC"], [ra, dec], usemask=False)


@_urddata_lru_cache
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

def quat_to_pol_and_roll(qfin, opaxis=[1, 0, 0], north=[0, 1, 0]):
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
    opticaxis = normalize(qfin.apply(opaxis))
    dec = np.arcsin(opticaxis[:,2])
    ra = np.arctan2(opticaxis[:,1], opticaxis[:,0])

    yzprojection = normalize(np.cross(opticaxis, north))

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply([0, 0, 1]), axis=1),
                           np.sum(yzprojection*qfin.apply([0, 1, 0]), axis=1))
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
    print(proj - qlon.apply([1, 0, 0]))
    rr = normalize(np.cross(proj, north))
    qlat = Rotation.from_rotvec(rr*np.arcsin(np.sum(vecs*north, axis=1))[:, np.newaxis])
    print(qlat.apply(proj) - vecs)
    qrot = Rotation.from_rotvec(-vecs*(pi + np.deg2rad(roll))[:, np.newaxis])
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

def get_wcs_roll_for_qval(wcs, qval, axlist=np.array([1, 0, 0])):
    """
    for provided wcs coordinate system, for each provided quaternion,
    defines roll angle between between local wcs Y axis and detector plane coordinate system in detector plane

    -------
    Params:
        wcs: astropy.wcs coordinate definition
        qval: set of quaternions, which rotate SC coordinate system

    return:
        for each quaternion returns roll angle
    """
    radec = np.rad2deg(vec_to_pol(qval.apply(axlist))).T
    x, y = wcs.all_world2pix(radec, 1).T
    r1, d1 = (wcs.all_pix2world(np.array([x, y - max(1./wcs.wcs.cdelt[1], 50.)]).T, 1)).T
    r2, d2 = (wcs.all_pix2world(np.array([x, y + max(1./wcs.wcs.cdelt[1], 50.)]).T, 1)).T
    vbot = pol_to_vec(r1*pi/180., d1*pi/180.)
    vtop = pol_to_vec(r2*pi/180., d2*pi/180.)
    vimgyax = vtop - vbot
    vimgyax = qval.apply(vimgyax, inverse=True)
    return np.arctan2(vimgyax[:, 1], vimgyax[:, 2])

def wcs_roll(wcs, qvals, axlist=np.array([1, 0, 0]), noffset=np.array([0., 0., 0.01])):
    ax1 = qvals.apply(axlist)
    radec = np.rad2deg(vec_to_pol(ax1)).T
    xy = wcs.all_world2pix(radec, 1)
    ax2 = pol_to_vec(*np.deg2rad(wcs.all_pix2world(xy + [0, 1], 1)).T)
    qalign = make_align_quat(ax1, ax2, zeroax=np.array([1, 0, 0]))
    print(qalign.as_rotvec())
    print(qvals.as_rotvec())
    rvec = (qalign*qvals).as_rotvec()
    return np.sqrt(np.sum(rvec**2, axis=-1))

def make_quat_for_wcs(wcs, x, y, roll):
    """
    produces rotation quaternion, which orients a cartesian systen XYZ in a
    wat X points at specified WCS system pixel and Z rotated on angle roll anticlockwise
    relative to the north pole
    params:
        WCS: wcs system defined by the astropy.wcs.WCS class
        x, y - coordinates where X cartesian vector should point after rotation
        roll - anticlockwise rotation angle between wcs north direction and Z cartesian vector
    """
    vec = pol_to_vec(*np.deg2rad(wcs.all_pix2world(np.array([x, y]).reshape((-1, 2)), 1)[0]))
    alpha = np.arccos(np.sum(vec*OPAX))
    rvec = np.cross(OPAX, vec)
    rvec = rvec/np.sqrt(np.sum(rvec**2))
    q0 = Rotation.from_rotvec(rvec*alpha)
    beta = get_wcs_roll_for_qval(wcs, q0)[0]
    return Rotation.from_rotvec(vec*(roll - beta))*q0



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
    print(q.shape)
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
    """
    scipy slerp works like q_i (q_i^-1 q_i+1 omega dt)

    """
    frac = np.zeros(slerp.timedelta.size, np.double)

    rmod = np.sqrt(np.sum(slerp.rotvecs**2, axis=1))
    rvec = slerp.rotvecs[mask]/rmod[mask, np.newaxis]
    a0 = slerp.rotations[mask].inv().apply(loc)
    cosa = np.sum(rvec*offvec, axis=1)
    cosb = np.sum(rvec*a0, axis=1)
    cose = cos(appsize*pi/180/3600)
    """
    despite the phase of a0 vec, in order to vector trajectorie to fall inside circula aperture following
    conditions should be satisfied
    alpha + epsilon < beta
    alpha - epsilon > beta
    """
    alpha = np.arccos(cosa)
    beta = np.arccos(cosb)
    """
    if alpha + beta < epsilon than the interpolation trajectroie is in the circular aperture despite the phase
    """
    maskallinsideapp = alpha + beta < appsize*pi/180/3600
    frac[mask] = maskallinsideapp.astype(np.double)

    maskoutofapp = np.logical_and(alpha + appsize*pi/180/3600 > beta,
                                  alpha - appsize*pi/180/3600 < beta)
    cosa, cosb, rmod, rvec, a0 = [arr[maskoutofapp] for arr in (cosa, cosb, rmod, rvec, a0)]
    sinbsq = 1 - cosb**2

    a = rvec*((cosa - cose*cosb)/sinbsq)[:, np.newaxis] + \
        a0*((cose - cosa*cosb)/sinbsq)[:, np.newaxis]
    """
    rot vec projections
    a = a0((cose - cosa*cosb)/sinbsq - rvec csosb(cose - cosa*cosb)/sinbsq
    """
    aort = np.cross(rvec, a0)
    port = np.cross(rvec, offvec) #/sina[:, np.newaxis]
    pdir = (offvec - rvec*cosa[:, np.newaxis]) #/sina[:, np.newaxis]
    """
    a1 and a2 - two vectors in the crosssection of the circles around rotation vector and loc
    """
    a1 = a - aort*(np.sqrt(sinbsq - cosa**2 - cose**2 + 2.*cose*cosa*cosb)/sinbsq)[:, np.newaxis]
    a2 = a + aort*(np.sqrt(sinbsq - cosa**2 - cose**2 + 2.*cose*cosa*cosb)/sinbsq)[:, np.newaxis]

    """
    phi1 and phi2 - are angles, we should to rotate offvec in order to get in to the epsilon vicinity of loc
    """
    phi1 = np.arctan2(np.sum(a1*port, axis=1), np.sum(a1*pdir, axis=1))
    phi2 = np.arctan2(np.sum(a2*port, axis=1), np.sum(a2*pdir, axis=1))

    m2 = np.copy(mask)
    m2[m2] = maskoutofapp
    t1 = slerp.times[:-1][m2] + slerp.timedelta[m2]*np.maximum(phi1, 0)/rmod
    t2 = slerp.times[:-1][m2] + slerp.timedelta[m2]*np.minimum(phi2, rmod)/rmod
    gti = GTI(np.array([t1, t2]).T)
    gti.merge_joint()

    frac[m2] = np.maximum((np.minimum(phi2, rmod) - np.maximum(phi1, 0)), 0)/rmod #*slerp.timedelta[m2]
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



def get_attdata(fname, **kwargs):
    ffile = fits.open(fname)
    if "gyro" in fname:
        attdata = read_gyro_fits(ffile["ORIENTATION"])
        tshift = get_device_timeshift("gyro")
    elif "bokz" in fname:
        attdata = read_bokz_fits(ffile["ORIENTATION"])
        tshift = get_device_timeshift("bokz")
    elif "RA" in ffile[1].data.dtype.names:
        tshift = get_device_timeshift("gyro")
        d = ffile[1].data
        attdata = AttDATA(d["TIME"], ra_dec_roll_to_quat(d["RA"], d["DEC"], d["ROLL"])*get_boresight_by_device("GYRO"))

    attdata.times = attdata.times - tshift
    attdata.gti.arr = attdata.gti.arr - tshift
    if "gyro" in fname:
        attdata = define_required_correction(attdata)
    return attdata


def attdata_for_urd(att, urdn):
    return att*get_boresight_by_device(urdn)
