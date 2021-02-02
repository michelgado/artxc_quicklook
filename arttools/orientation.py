from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin, sqrt
from ._det_spatial import urd_to_vec, F, DL
from .time import get_hdu_times, GTI, tGTI, emptyGTI
from .caldb import T0, get_boresight_by_device, get_device_timeshift
from .mask import edges as medges
from functools import reduce
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

gyrocorrectionbti = GTI([[624390347, 624399808], [6.24410643e+08, 6.30954575e+08]])

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
    if attdata.times.size == 0:
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

qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
qgyro0 = Rotation([0., 0., 0., 1.])
OPAX = np.array([1, 0, 0])

SECPERYR = 3.15576e7
SOLARSYSTEMPLANENRMALEINFK5 = np.array([-9.83858346e-08, -3.97776911e-01,  9.17482168e-01])

def to_2pi_range(val): return val%(2.*pi)

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

class SlerpWithNaiveIndexing(Slerp):
    """
    scipy quaternions interpolation class with indexing
    """
    def __getitem__(self, idx):
        return Slerp(self.times[idx], self.rotations[idx])

    def __call__(self, tarr):
        return Rotation(np.empty((0, 4), np.double)) if np.asarray(tarr).size == 0 \
                else super().__call__(tarr)

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

    def __init__(self, *args, gti=None, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.gti = GTI(gti)
        except Exception:
            self.gti = GTI(self.times[[0, -1]])
        self._check_interpolation_quality()

    def _check_interpolation_quality(self):
        dt = np.diff(self.times, 1)
        dtmed = np.median(dt)
        gaps = dt > 1.5*dtmed
        if gaps.size > 0:
            gaps[[0, -1]] = False

        if not np.any(gaps):
            self.badrec = emptyGTI
            self.goodrec = emptyGTI
        else:
            #don't considere specific cases of the gaps at the end and beginning
            #check derivatives
            idx = np.where(gaps)[0]

            q = self.rotations[idx]*Rotation.from_rotvec(self.rotvecs[idx - 1]*\
                    self.timedelta[idx, np.newaxis]/self.timedelta[idx - 1, np.newaxis])
            dvec = np.sum(self.rotations[idx + 1].apply(OPAX)*q.apply(OPAX), axis=1)
            goodinterp = dvec > cos(pi/180.*10./3600)

            q = self.rotations[idx + 1]*Rotation.from_rotvec(self.rotvecs[idx + 1]*\
                    -self.timedelta[idx, np.newaxis]/self.timedelta[idx + 1, np.newaxis])
            dvec = np.sum(self.rotations[idx].apply(OPAX)*q.apply(OPAX), axis=1)
            goodinterp = np.logical_and(goodinterp, dvec > cos(pi/180*10/3600))

            ggaps = np.copy(gaps)
            ggaps[ggaps] = goodinterp
            bgaps = np.copy(gaps)
            bgaps[bgaps] = np.logical_not(goodinterp)
            self.goodinterp = GTI(self.times[medges(ggaps)])
            self.badinterp = GTI(self.times[medges(bgaps)])

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
        te, gaps = self.gti.make_tedges(self.times)
        vc = self((te[1:] + te[:-1])/2.).apply(ax)
        mask = np.sum(vc*vec, axis=1) > cos(app)
        mask[~gaps] = False
        lgti = GTI(te[medges(mask)])
        return lgti

    def rect_gti(self, rect):
        pass




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
    qbokz = earth_precession_quat(jyear).inv()*Rotation.from_dcm(mat[mask])*qbokz0*\
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
    qbokz = Rotation.from_dcm(mat[mask])*qbokz0
    jyear = get_hdu_times(bokzhdu).jyear[mask]
    return bokzdata["TIME"][mask], earth_precession_quat(jyear).inv()*qbokz


def get_photons_vectors(urddata, URDN, attdata, subscale=1):
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
    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    return phvec

def get_photons_sky_coord(urddata, URDN, attdata, subscale=1):
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


    --------
    Params:
        quat - a set of quaternions in form of scipy.spatial.transform.Rotation container
        opaxis - optical axis vector in sc frame
        north - axis relative to which roll angle will be defined

    return:
        ra, dec, roll - attitude fk5 coordinates (ra, dec) and roll angle relative to north axis ! note that all angles are returned in radians
    """
    opticaxis = qfin.apply(opaxis)
    dec = np.arctan(opticaxis[:,2]/np.sqrt(opticaxis[:,1]**2 + opticaxis[:,0]**2))
    ra = np.arctan2(opticaxis[:,1], opticaxis[:,0])%(2.*pi)

    yzprojection = np.cross(opticaxis, north)
    yzprojection = yzprojection/np.sqrt(np.sum(yzprojection**2., axis=1))[:, np.newaxis]

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply([0, 1, 0]), axis=1),
                           np.sum(yzprojection*qfin.apply([0, 0, 1]), axis=1))
    return ra, dec, rollangle

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

def get_wcs_roll_for_qval(wcs, qval):
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
    ra, dec = np.rad2deg(vec_to_pol(qval.apply([1, 0, 0])))
    radec = np.array([np.ravel(ra), np.ravel(dec)]).T
    x, y = wcs.all_world2pix(radec, 1).T
    r1, d1 = (wcs.all_pix2world(np.array([x, y - max(1./wcs.wcs.cdelt[1], 50.)]).T, 1)).T
    r2, d2 = (wcs.all_pix2world(np.array([x, y + max(1./wcs.wcs.cdelt[1], 50.)]).T, 1)).T
    vbot = pol_to_vec(r1*pi/180., d1*pi/180.)
    vtop = pol_to_vec(r2*pi/180., d2*pi/180.)
    vimgyax = vtop - vbot
    vimgyax = qval.apply(vimgyax, inverse=True)
    return np.arctan2(vimgyax[:, 1], vimgyax[:, 2])

def make_quat_for_wcs(wcs, x, y, roll):
    vec = pol_to_vec(*np.deg2rad(wcs.all_pix2world(np.array([x, y]).reshape((-1, 2)), 1)[0]))
    alpha = np.arccos(np.sum(vec*OPAX))
    rvec = np.cross(OPAX, vec)
    rvec = rvec/np.sqrt(np.sum(rvec**2))
    q0 = Rotation.from_rotvec(rvec*alpha)
    beta = get_wcs_roll_for_qval(wcs, q0)[0]
    print(beta*180/pi)
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


class SurveyAtt(object):
    """
    """
    def __init__(self, polus, omega, tstart, tstop, sc_start_rotvec, sc_romega, sc_start_vec, roll):
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
    print(cosa, cosb, cose)
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

    frac[m2] = np.maximum((np.minimum(phi2, rmod) - np.maximum(phi1, 0)), 0)/rmod #*slerp.timedelta[m2]
    return frac

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

def get_attdata(fname):
    ffile = fits.open(fname)
    attdata = read_gyro_fits(ffile["ORIENTATION"]) if "gyro" in fname else read_bokz_fits(ffile["ORIENTATION"])
    tshift = get_device_timeshift("gyro" if "gyro" in fname else "bokz")
    attdata.times = attdata.times - tshift
    attdata.gti.arr = attdata.gti.arr - tshift
    if "gyro" in fname:
        attdata = define_required_correction(attdata)
    return attdata
