from astropy.wcs import WCS
from .time import tGTI, GTI
from .mask import edges as medges
from ._det_spatial import offset_to_vec, DL
from .orientation import get_elongation_plane_norm, align_with_z_quat, vec_to_pol, minimize_norm_to_survey
import numpy as np
from math import cos, sin, pi, sqrt, asin, acos
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt

class Corners(object):
    """
    class to define convex hull on the sphere surface
    """
    def __init__(self, vecs):
        """
        init class with a series of vectors, which defines directions,
        all vectors will be reset to have 1 length
        after that class producess following attributres:
            vertices - unit vectors in the vertices of the convex hull, containing all vectors
            idx - indexes of the vectors, located in the corners, from  input array of vectors
        """
        #plt.scatter(vecs[:, 1], vecs[:, 2])
        cvec = vecs.sum(axis=0)
        cvec = cvec/sqrt(np.sum(cvec**2.))
        idx1 = get_most_distant_idx(cvec, vecs)
        idx2 = get_most_distant_idx(vecs[idx1], vecs)
        idx3, external = get_most_orthogonal_idx(vecs[idx1], vecs[idx2], vecs)
        self.idx = [idx1, idx2, idx3]
        vec1, vec2, vec3 = vecs[idx1], vecs[idx3], vecs[idx2]
        mask = get_outof_trinagle(vec1, vec2, vec3, vecs)
        mask[self.idx] = False
        self.vecs = vecs[mask]
        self.corners = [vec1,]
        self.pos = np.arange(vecs.shape[0])[mask]
        self.check_newpoint(vec1, vec2, color="r", lvl=1.0)
        self.corners.append(vec2)
        self.check_newpoint(vec2, vec3, color="g", lvl=1.0)
        self.corners.append(vec3)
        self.check_newpoint(vec3, vec1, color="b", lvl=1.0)

        self.vertices = np.array(self.corners)
        self.idx = np.array(self.idx)
        self.orts = np.cross(self.vertices, np.roll(self.vertices, -1, axis=0))
        self.orts = self.orts/np.sqrt(np.sum(self.orts**2, axis=1))[:, np.newaxis]

    def check_newpoint(self, vec1, vec3, color, lvl):
        if self.vecs.size == 0:
            return None
        idx, external = get_most_orthogonal_idx(vec1, vec3, self.vecs)
        if external[idx]:
            vec2 = self.vecs[idx]
            iloc = self.pos[idx]
            mask = np.ones(external.size, np.bool)
            lemask = get_outof_trinagle(vec1, vec2, vec3, self.vecs[external])
            mask[external] = lemask
            mask[idx] = False
            self.pos = self.pos[mask]
            self.vecs = self.vecs[mask]
            self.check_newpoint(vec1, vec2, color, lvl-0.1)
            self.idx.append(iloc)
            self.corners.append(vec2)
            self.check_newpoint(vec2, vec3, color, lvl-0.1)

    def check_inside_polygon(self, vecs):
        return np.logical_not(np.any(np.sum(self.orts[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2) > 0, axis=1))

    def expand(self, dtheta):
        """
        expand convex moving straight lines, limiting each side of the  convex, on the angle dtheta
        negative angle should be used with caution, since the sum of the angles on vertixes inside convex shrinks with convex surface area and some corners
        can appear on the oposite side of the sphere
        """
        vproj = self.vertices + np.roll(self.vertices, -1, axis=0)
        vproj = vproj/np.sqrt(np.sum(vproj**2, axis=1))[:, np.newaxis]
        neworts = self.orts*cos(dtheta) - vproj*sin(dtheta)
        neworts = neworts/np.sqrt(np.sum(neworts**2, axis=1))[:, np.newaxis]
        newcorners = np.cross(neworts, np.roll(neworts, 1, axis=0))
        newcorners = newcorners/np.sqrt(np.sum(newcorners**2, axis=1))[:, np.newaxis]
        return Corners(newcorners)

    @property
    def area(self):
        return get_vec_triangle_area(self.vertices[np.zeros(self.vertices.shape[0] - 2).astype(np.int)],
                                     self.vertices[1:-1],
                                     self.vertices[2:]).sum()*(180./pi)**2.

def random_orthogonal_vec(vec):
    """
    for a given 3D vector in form of np.array(3)
    provides randomly oriented orthogonal vectro (not statistically random though)


    ------
    Params:
        vec

    returns:
        vec
    """
    idx = np.argsort(vec)
    rvec = np.zeros(3, np.double)
    rvec[idx[1:]] = vec[idx[:0:-1]]*[-1, 1]
    return rvec/sqrt(np.sum(rvec**2.))

def get_most_distant_idx(vec, vecs):
    return np.argmin(np.sum(vec*vecs, axis=1))

def get_most_orthogonal_idx(vec1, vec2, vecs):
    vort = np.cross(vec1, vec2)
    proj = np.sum(vort*vecs, axis=1)
    idx = np.argmax(proj)
    return idx, proj > 0

def get_outof_trinagle(vec1, vec2, vec3, vecs):
    """
    check, whether the vecs are outside of triangle with verteces vec1, vec2 and vec3
    """
    vort1 = np.cross(vec1, vec2)
    vort2 = np.cross(vec2, vec3)
    vort3 = np.cross(vec3, vec1)
    s1 = np.sum(vort1*vecs, axis=1)
    s2 = np.sum(vort2*vecs, axis=1)
    s3 = np.sum(vort3*vecs, axis=1)
    return  np.any([s1*s2 < 0, s1*s3 < 0, s2*s3 < 0], axis=0)

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
    cvec = cvec/sqrt(np.sum(cvec**2.))
    vrot = np.cross(np.array([0., 0., 1.]), cvec)
    vrot = vrot/np.sqrt(np.sum(vrot**2.))
    #vrot2 = np.cross(vrot, cvec)
    #vrot2 = vrot2/sqrt(np.sum(vrot2**2))

    alpha = pi/2. - np.arccos(cvec[2])
    quat = Rotation([vrot[0]*sin(alpha/2.), vrot[1]*sin(alpha/2.), vrot[2]*sin(alpha/2.), cos(alpha/2.)])
    r1 = np.array([0, 0, 1])
    r2 = np.cross(quat.apply(cvec), r1)
    vecn = quat.apply(vecs) - quat.apply(cvec)
    l, b = np.sum(quat.apply(vecs)*r2, axis=1), vecn[:,2]
    ch = ConvexHull(np.array([l, b]).T)
    r, d = l[ch.vertices], b[ch.vertices]
    return cvec, r1, r2, quat, vecs[ch.vertices], r, d

def get_ort_component(vec1, vec2):
    vort = vec1 - vec1*np.sum(vec1*vec2, axis=-1)
    return vort/np.sqrt(np.sum(vort**2, axis=-1))

def sphere_triangle_angle(alpha, beta, gamma):
    return  alpha - beta - gamma - pi

def get_angle_betwee_three_vectors(vec1, vec2, vec3):
    v12 = np.cross(vec1, vec2, axis=-1)
    v13 = np.cross(vec1, vec3, axis=-1)
    v23 = np.cross(vec2, vec3, axis=-1)
    v12 = 1/np.sqrt(np.sum(v12**2, axis=-1))*v12
    v13 = 1/np.sqrt(np.sum(v13**2, axis=-1))*v13
    alpha = np.arccos(np.sum(v12*v13, axis=-1))
    return alpha

def get_vec_triangle_area(vec1, vec2, vec3):
    v12 = np.cross(vec1, vec2, axis=-1)
    v13 = np.cross(vec1, vec3, axis=-1)
    v23 = np.cross(vec2, vec3, axis=-1)
    v12 = v12/np.sqrt(np.sum(v12**2, axis=-1))[:, np.newaxis]
    v13 = v13/np.sqrt(np.sum(v13**2, axis=-1))[:, np.newaxis]
    v23 = v23/np.sqrt(np.sum(v23**2, axis=-1))[:, np.newaxis]
    alpha = np.arccos(np.sum(v12*v13, axis=-1))
    beta = np.arccos(-np.sum(v12*v23, axis=-1))
    gamma = np.arccos(np.sum(v23*v13, axis=-1))
    return alpha + beta + gamma - pi

def get_convex_center(convex):
    areas = get_vec_triangle_area(convex.vertices[0], convex.vertices[1:-1], convex.vertices[2:])
    vloc = convex.vertices[0] + convex.vertices[1:-1] + convex.vertices[2:]
    vloc = vloc/np.sqrt(np.sum(vloc**2, axis=-1))[:, np.newaxis]
    vm = np.sum(vloc*areas[:, np.newaxis], axis=0)
    vm = vm/sqrt(np.sum(vm**2))
    return vm

def minarea_ver2(vecs, pixsize=20./3600.):
    """
    ra, dec = vec_to_pol(vecs)
    plt.scatter(ra, dec)
    """
    corners = Corners(vecs)
    """
    ra, dec = vec_to_pol(corners.vertices)
    plt.scatter(ra, dec, marker="x")
    plt.plot(ra, dec, "k", lw=2)
    """
    vm = get_convex_center(corners)
    rac, decc = vec_to_pol(vm)
    """
    plt.scatter(ra, dec, marker="^", s = 30)
    """
    vfidx = np.argmin(np.sum(vm*corners.vertices, axis=-1))
    alpha = get_angle_betwee_three_vectors(vm, corners.vertices[vfidx], [0, 0, 1])
    size = int(np.arccos(np.sum(vm*corners.vertices[vfidx]))*180./pi/pixsize) + 2
    print(size)

    locwcs = WCS(naxis=2)
    locwcs.wcs.crpix = [size, size]
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]
    locwcs.wcs.cdelt = [pixsize, pixsize]
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.radesys = "FK5"
    ra, dec = vec_to_pol(corners.vertices)
    x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
    sizex, sizey = int(x.max() - x.min()), int(y.max() - y.min())
    print("initial alpha guess", alpha, "and corresponding size", sizex, sizey)
    rac, decc = locwcs.all_pix2world([[(x.max() + x.min())/2., (y.max() + y.min())/2.],], 1)[0]
    locwcs.wcs.crval = [rac, decc]
    locwcs.wcs.crpix = [int(sizex//2) + 1 - int(sizex//2)%2, int(sizey//2) + 1 - int(sizey//2)%2]
    def minarea(var):
        alpha = var[0]
        cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
        locwcs.wcs.pc = cdmat
        x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
        return (x.max() - x.min())**2.*(y.max() - y.min())
    alpha = minimize(minarea, [alpha,], method="Nelder-Mead").x[0]
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
    sizex = max(int(x.max() - locwcs.wcs.crpix[0]), int(locwcs.wcs.crpix[0] - x.min())) #int((x.max() - x.min())//2)
    sizey = max(int(y.max() - locwcs.wcs.crpix[1]), int(locwcs.wcs.crpix[1] - y.min())) #int((y.max() - y.min())//2)
    print("final alpha guess", alpha, "and corresponding size", sizex, sizey)
    locwcs.wcs.crpix = [sizex + 1 - sizex%2, sizey + 1 - sizey%2]
    return locwcs


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

    --------
    Params:
        vecs: a set of vectors

    return:
        wcs (in a form of astropy.wcs.WCS class instance)
    """
    #convex = Corners(vecs)
    #vecs = convex.vertices
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
    for the set of quats returns wcs, which minimize roll angles along [0, 0, 1] axis

    -------
    params:
        quats - a set of quaternions in scipy.spatial.transform.Rotation container

    returns:
        wcs which is astropy.wcs.WCS class instance
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
    """
    compute standard realization of the wsc for the provided quaternions

    -------
    params:
        quats : a set of quaternions, for which wcs will be computed

    returns;
        wcs computed with determined algorithm
    """
    vedges = offset_to_vec(np.array([-26.*DL, 26*DL, 26.*DL, -26.*DL]),
                           np.array([-26.*DL, -26*DL, 26.*DL, 26.*DL]))
    edges = np.concatenate([quats.apply(v) for v in vedges])
    edges = edges/np.sqrt(np.sum(edges**2., axis=1))[:, np.newaxis]
    #return min_area_wcs_for_vecs(edges, pixsize=pixsize)
    return minarea_ver2(edges, pixsize=pixsize)

def make_wcs_for_attdata(attdata, gti=tGTI, pixsize=20./3600.):
    """
    compute standard realization of wcs for provided attdata

    -------
    Params:
        attdata : AttDATA container for quaternion interpolation
        gti = not mandatory, allows to  compute gti for specific time interval

    returns:
        wcs for provided attitude informaion
    """
    locgti = gti & attdata.gti
    #qvtot = attdata(attdata.times[locgti.mask_external(attdata.times)])
    qvtot = attdata(locgti.arange(1)[0])
    return make_wcs_for_quats(qvtot, pixsize)

def make_wcs_for_vec_edges(vecs, alpha=None, pixsize=10./3600.):
    cvec = Corners(vecs)
    locwcs = WCS(naxis=2)
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.cdelt = [pixsize, pixsize]
    locwcs.wcs.radesys = "FK5"






def split_survey_mode(attdata, gti=tGTI):
    aloc = attdata.apply_gti(gti)
    rpvec = get_elongation_plane_norm(aloc)
    rquat = align_with_z_quat(rpvec)
    amin = np.argmin(vec_to_pol(aloc(aloc.times).apply([1, 0, 0]))[0])
    zeropoint = aloc([aloc.times[amin],])[0].apply([1, 0, 0])
    zeropoint = rquat.apply(zeropoint)
    alpha0 = np.arctan2(zeropoint[1], zeropoint[0])

    vecs = aloc(aloc.times).apply([1, 0, 0])
    vecsz = rquat.apply(vecs)
    alpha = (np.arctan2(vecsz[:, 1], vecsz[:, 0]) - alpha0)%(2.*pi)
    edges = np.linspace(0., 2.*pi, 37)
    gtis = [medges((alpha > e1) & (alpha < e2)) + [0, -1] for e1, e2 in zip(edges[:-1], edges[1:])]
    return [GTI(aloc.times[m]) for m in gtis]

def make_wcs_for_survey_mode(attdata, gti=None, rpvec=None, pixsize=20./3600.):
    aloc = attdata.apply_gti(gti) if not gti is None else attdata

    if rpvec is None: rpvec = get_elongation_plane_norm(aloc)

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
    rasize = max(abs(xmin), abs(xmax))
    desize = max(abs(ymin), abs(ymax))
    rasize = rasize - rasize%2 + 1
    desize = desize - desize%2 + 1

    locwcs.wcs.radesys = "FK5"
    locwcs.wcs.crpix = [rasize, desize]

    return locwcs


def switch_between_corners_to_plane_axis(corners_or_axis):
    """
    corners (or axis) is a set of 4 vectors at the edges (or orthorgonal to stright lines limiting image) of a desired image
    the corners (or axis) should be ordered
    """
    vecs = np.cross(corners_or_axis, np.roll(corners_or_axis, 1, axis=0))
    return vecs/np.sqrt(np.sum(vecs**2, axis=1))[:, np.newaxis]

PRIME_NUMBERS = np.array([1,   2,   3,   5,   7,  11,  13,  17,  19,  23,  29,  31,  37,
                         41,  43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,
                         101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163,
                         167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233,
                         239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311,
                         313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389,
                         397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463,
                         467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563,
                         569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641,
                         643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727,
                         733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821,
                         823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907,
                         911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997])

def get_split_side(val, guess=None):
    guess = int(sqrt(val)) if guess is None else guess
    while guess*(val//guess) != val:
        guess = guess - 1
    return guess

def split_rectangle_on_sphere(axis, snum):
    """
    it is expected, that axis are ordered and  have unit length
    """
    alpha = acos(-np.sum(axis[0]*axis[2]))
    beta = acos(-np.sum(axis[1]*axis[3]))
    print(alpha, beta)

    smallside = get_split_side(snum, int(round(sqrt(snum)*min(alpha, beta)/sqrt(alpha*beta))) + 1)
    bigside = snum//smallside
    print(smallside, bigside)
    #dalpha = np.linspace(0, alpha, bigside  + 1 if alpha > beta else smallside + 1)
    #dbeta = np.linspace(0, beta, bigside + 1 if beta > alpha else smallside + 1)
    if alpha > beta:
        smallside, bigside = bigside, smallside

    rv1 = np.cross(axis[0], axis[2])
    rv1 = rv1/np.sqrt(np.sum(rv1**2))
    rv2 = np.cross(axis[1], axis[3])
    rv2 = rv2/np.sqrt(np.sum(rv2**2))

    r1 = Rotation.from_rotvec(-rv1[np.newaxis, :]*np.linspace(0., alpha, smallside + 1)[:, np.newaxis])
    r2 = Rotation.from_rotvec(-rv2[np.newaxis, :]*np.linspace(0., beta, bigside + 1)[:, np.newaxis])
    xax = r1.apply(axis[0])
    yax = r2.apply(axis[1])
    print(xax.shape)

    """
    xax2 = axis[2][np.newaxis, :]*np.cos(dalpha)[-2::-1, np.newaxis] + (axis[0] - cos(alpha)*axis[2])[np.newaxis]*np.sin(dalpha)[-2::-1, np.newaxis]/sin(alpha)
    yax2 = axis[3][np.newaxis, :]*np.cos(dbeta)[-2::-1, np.newaxis] + (axis[1] - cos(beta)*axis[3])[np.newaxis]*np.sin(dbeta)[-2::-1, np.newaxis]/sin(beta)
    xax1 = xax1/np.sqrt(np.sum(xax1**2, axis=1))[:, np.newaxis]
    yax1 = yax1/np.sqrt(np.sum(yax1**2, axis=1))[:, np.newaxis]
    xax2 = xax2/np.sqrt(np.sum(xax2**2, axis=1))[:, np.newaxis]
    yax2 = yax2/np.sqrt(np.sum(yax2**2, axis=1))[:, np.newaxis]
    """
    #xax = -(axis[0][np.newaxis, :]*np.cos(dalpha)[:, np.newaxis] + (-axis[2] - cos(alpha)*axis[0])[np.newaxis]*np.sin(dalpha)[:, np.newaxis]/sin(alpha))
    #yax = -(axis[1][np.newaxis, :]*np.cos(dbeta)[:, np.newaxis] + (-axis[3] - cos(beta)*axis[1])[np.newaxis]*np.sin(dbeta)[:, np.newaxis]/sin(beta))
    #xax = xax/np.sqrt(np.sum(xax**2, axis=1))[:, np.newaxis]
    #yax = yax/np.sqrt(np.sum(yax**2, axis=1))[:, np.newaxis]
    grids = [np.array([xax[i%(xax.shape[0] - 1)], yax[i//(xax.shape[0] - 1)], -xax[i%(xax.shape[0] - 1) + 1], -yax[i//(xax.shape[0] - 1) + 1]]) for i in range(snum)]

    #if alpha > beta:
    #    axisx = axis[0]*np.cos() + np.arange(bigside)[::-1]*axis[2]
    return grids


def expand_rectangle(axis, dtheta):
    alpha = acos(-np.sum(axis[0]*axis[2]))
    beta = acos(-np.sum(axis[1]*axis[3]))
    newax = np.array([axis[0]*cos(dtheta) + (axis[2] - cos(alpha)*axis[0])*sin(dtheta)/sin(alpha),
                     axis[1]*cos(dtheta) + (axis[3] - cos(beta)*axis[1])*sin(dtheta)/sin(beta),
                     axis[2]*cos(dtheta) + (axis[0] - cos(alpha)*axis[2])*sin(dtheta)/sin(alpha),
                     axis[3]*cos(dtheta) + (axis[1] - cos(beta)*axis[3])*sin(dtheta)/sin(beta),])
    return newax/np.sqrt(np.sum(newax**2, axis=1))[:, np.newaxis]

def expand_convex_hull(vertices, dtheta):
    """
    depending on the direction of boundary bypass, axis can be oriented outside or inside of the convex hull

    """
    convexhull = Corners(vertices)
    corners = np.cross(corners_or_axis, np.roll(corners_or_axis, -1, axis=0))

    get_convex_center(corners)

    alpha = acos(-np.sum(axis[0]*axis[2]))
    beta = acos(-np.sum(axis[1]*axis[3]))
    newax = np.array([axis[0]*cos(dtheta) + (axis[2] - cos(alpha)*axis[0])*sin(dtheta)/sin(alpha),
                     axis[1]*cos(dtheta) + (axis[3] - cos(beta)*axis[1])*sin(dtheta)/sin(beta),
                     axis[2]*cos(dtheta) + (axis[0] - cos(alpha)*axis[2])*sin(dtheta)/sin(alpha),
                     axis[3]*cos(dtheta) + (axis[1] - cos(beta)*axis[3])*sin(dtheta)/sin(beta),])
    return newax/np.sqrt(np.sum(newax**2, axis=1))[:, np.newaxis]
