from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin
from ._det_spatial import urd_to_vec
from .time import get_hdu_times, GTI
from .caldb import ARTQUATS

T0 = 617228538.1056 #first day of ART-XC work

qrot0 = Rotation([sin(15*pi/360.), 0., 0., cos(15*pi/360.)]) #ART detectors cs to the spasecraft cs
qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
qgyro0 = Rotation([0., 0., 0., 1.])
OPAX = np.array([1, 0, 0])

def to_2pi_range(val): return val%(2.*pi)

def vec_to_pol(phvec):
    dec = np.arctan(phvec[...,2]/np.sqrt(phvec[...,0]**2. + phvec[...,1]**2.))
    ra = (np.arctan2(phvec[...,1], phvec[...,0])%(2.*pi))
    return ra, dec

def pol_to_vec(phi, theta):
    vec = np.empty(theta.shape + (3,), np.double)
    vec[..., 0] = np.cos(theta)*np.cos(phi)
    vec[..., 1] = np.cos(theta)*np.sin(phi)
    vec[..., 2] = np.sin(theta)
    return vec


class SlerpWithNaiveIndexing(Slerp):
    def __getitem__(self, idx):
        return Slerp(self.times[idx], self(self.times[idx]))

    def __add__(self, other):
        times = np.concatenate([self.times, other.times])
        qself = self(self.times)
        qother = other(other.times)
        quats = np.concatenate([qself.as_quat(), qother.as_quat()])
        ts, idx = np.unique(times, return_index=True)
        return self.__class__(ts, Rotation(quats[idx]))

    def __mul__(self, val):
        return Slerp(self.times, self(self.times)*val)


class AttDATA(SlerpWithNaiveIndexing):

    def __init__(self, *args, gti=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.gti = GTI(gti)
        except Exception:
            self.gti = GTI(self.times[[0, -1]])

    def __add__(self, other):
        base = super().__add__(other)
        base.gti = self.gti | other.gti
        return base

    def _get_clean(self):
        mask = self.gti.mask_outofgti_times(self.times)
        quats = self(self.times[mask])
        return self.__class__(self.times[mask], quats, self.gti)

    def apply_gti(self, gti):
        gti = GTI(gti)
        gti = gti & self.gti
        print(gti)
        mask = gti.mask_outofgti_times(self.times)
        quats = self(self.times[mask])
        self.__init__(self.times[mask], quats, gti=gti)

    def get_optical_axis_movement_speed(self):
        """
        for provided gyrodata computes angular speed
        returns:
            ts - centers of the time bins
            dt - withd of the time bins
            dlaphadt - angular speed in time bin
        """
        te, mgaps = self.gti.make_tedges(self.times)
        tc = ((te[1:] + te[:-1])/2.)[mgaps]
        dt = (te[1:] - te[:-1])[mgaps]
        vecs = self(tc).apply(OPAX)
        dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))/dt*180./pi*3600.
        return tc, dt, dalphadt

def read_gyro_fits(gyrohdu):
    gyrodata = gyrohdu.data
    quats = np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T
    times = gyrodata["TIME"]
    masktimes = times > T0
    mask0quats = np.sum(quats**2, axis=1) > 0.
    mask = np.logical_and(masktimes, mask0quats)
    return AttDATA(times[mask], qgyro0*Rotation(quats[mask])*qrot0)

def read_bokz_fits(bokzhdu):
    bokzdata = bokzhdu.data
    mat = np.array([[bokzdata["MOR%d%d" % (i, j)] for i in range(3)] for j in range(3)])
    mat = np.copy(mat.swapaxes(2, 1).swapaxes(1, 0))
    mask0quats = np.linalg.det(mat) != 0.
    masktimes = bokzdata["TIME"] > T0
    mask = np.logical_and(mask0quats, masktimes)
    qbokz = qbokz0*Rotation.from_dcm(mat[mask])*qrot0
    jyear = get_hdu_times(hdu).jyear
    return AttDATA(bokzdata["TIME"][mask], earth_precession_quat(jyear).inv()*qbokz)

def get_photons_vectors(urddata, URDN, attdata, subscale=1):
    if not np.all(attdata.gti.mask_outofgti_times(urddata["TIME"])):
        raise ValueError("some events are our of att gti")
    qall = attdata(np.repeat(urddata["TIME"], subscale*subscale))*ARTQUATS[URDN]
    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    return phvec

def get_photons_sky_coord(urddata, URDN, attdata, subscale=1):
    phvec = get_photons_vectors(urddata, URDN, attdata, subscale)
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
    """
    opticaxis = qfin.apply(opaxis)
    dec = np.arctan(opticaxis[:,2]/np.sqrt(opticaxis[:,1]**2 + opticaxis[:,0]**2))
    ra = np.arctan2(opticaxis[:,1], opticaxis[:,0])%(2.*pi)

    yzprojection = np.cross(opticaxis, north)
    vort = np.cross(north, opaxis)

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply(vort), axis=1),
                           np.sum(yzprojection*qfin.apply(north), axis=1))
    return ra, dec, rollangle

def extract_raw_gyro(gyrodata, qadd=Rotation([0, 0, 0, 1])):
    """
    unpacks row gyro fits file in to RA, DEC and roll angle (of the telescope coordinate system)
    in J2000 coordinates.
    """
    qfin = get_gyro_quat(gyrodata)*qadd
    return quat_to_pol_and_roll(qfin)


def earth_precession_quat(jyear):
    """
    taken from astropy.coordinates.earth_orientation
    contains standrard precession ephemerides, accepted at IAU 2006,
    didn't check but should work better then IAU76 version, which provide several mas upto 2040
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
    returns:
        ts - centers of the time bins
        dt - withd of the time bins
        dlaphadt - angular speed in time bin
    """
    te, mgaps = attdata.gti.make_tedges(attdata.times)
    tc = ((te[1:] + te[:-1])/2.)[mgaps]
    dt = (te[1:] - te[:-1])[mgaps]
    vecs = attdata(tc).apply(OPAX)
    dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))/dt*180./pi*3600.
    return tc, dt, dalphadt


def get_angular_speed(vecs, time):
    dt = (time[1:] - time[:-1])
    dalphadt = np.arccos(np.sum(vecs[:-1]*vecs[1:], axis=1))/dt*180./pi*3600.
    return (time[1:] + time[:-1])/2., dt, dalphadt
