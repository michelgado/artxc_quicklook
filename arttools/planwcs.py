from astropy.wcs import WCS
import numpy as np
from math import cos, sin, pi, sqrt
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp

def get_vecs_convex(vecs):
    """
    for a set of vectors (which assumed to cover less the pi along any direction)
    produces set of vectors which is verteces of convex figure, incapsulating all other vectors on the sphere

    ------
    Params:
        vecs - a set of vectos in the form of ... x 3 array
    return:
        cvec - mean unit vector = SUM vecs / || SUM vecs ||
        r1, r2 - two orthogonal vectors, which will define axis dirrection on the sphere surface, at which the convex figure will be searched
        quat - quaternion, which puts cvec in the equatorial plane along the shortest trajectory
        vecs - a set of vectors, which points to the vertexes of the convex hull on the sphere surface, r, d - quasi cartesian coordinates of the vecteces

    it should be noted that convex hull is expected to be alongated along equator after quaternion rotation
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
    return min_area_wcs_for_vecs(edges, pixsize=pixsize)

def make_wcs_for_attdata(attdata, gti=tGTI, pixsize=20./3600.):
    locgti = gti & attdata.gti
    qvtot = attdata(attdata.times[locgti.mask_outofgti_times(attdata.times)])
    return make_wcs_for_quats(qvtot, pixsize)

def split_survey_mode(attdata, gti=tGTI):
    aloc = attdata.apply_gti(gti)
    rpvec = get_survey_mode_rotation_plane(aloc)
    rpvec = minimize_norm_to_survey(aloc, rpvec)
    rquat = align_with_z_quat(rpvec)
    amin = np.argmin(vec_to_pol(aloc(aloc.times).apply([1, 0, 0]))[0])
    print(amin)
    """
    zeropoint = [1, 0, 0] - rpvec*rpvec[0]
    print(zeropoint)
    #zeropoint = np.cross(rpvec, [0, 1, 0])
    zeropoint = zeropoint/sqrt(np.sum(zeropoint**2.))
    """
    zeropoint = aloc([aloc.times[amin],])[0].apply([1, 0, 0])
    print(zeropoint, vec_to_pol(zeropoint), np.arctan2(zeropoint[1], zeropoint[0]))
    zeropoint = rquat.apply(zeropoint)
    print(zeropoint)
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


