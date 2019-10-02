from scipy.spatial.transform import Rotation, Slerp
from .time import make_ingti_times, deadtime_correction
from .orientation import get_gyro_quat, quat_to_pol_and_roll, pol_to_vec, \
    ART_det_QUAT, ART_det_mean_QUAT, vec_to_pol
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from astropy.wcs import WCS
from math import cos, sin, pi
from .telescope import OPAX
import numpy as np
from math import pi


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

def make_small_steps_quats(times, quats, gti, urdhk=None):
    quatint = Slerp(times, quats)
    tnew, maskgaps = make_ingti_times(times, gti)
    if tnew.size == 0:
        return Rotation(np.empty((0, 4), np.double)), np.array([])
    ts = ((tnew[1:] + tnew[:-1])/2.)[maskgaps]
    dt = (tnew[1:] - tnew[:-1])[maskgaps]

    if not urdhk is None:
        dt = dt*deadtime_correction(ts, urdhk)

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
        dtn = np.concatenate([dt[maskmoving], dtm])
        qval = quatint(np.concatenate([ts[maskmoving], tnew]))
    else:
        dtn = dt
    return qval, dtn

def hist_orientation_for_attdata(attdata, gti, corr_quat=ART_det_mean_QUAT):
    quats = get_gyro_quat(attdata)*corr_quat
    qval, dtn = make_small_steps_quats(attdata["TIME"], quats, gti)
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
    res = minimize(find_bbox, [0., ], method="Nelder-Mead")
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
