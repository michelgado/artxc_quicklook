from astropy.wcs import WCS
from .time import tGTI, GTI
from .mask import edges as medges
from ._det_spatial import offset_to_vec, DL, raw_xy_to_vec
from .vector import pol_to_vec, vec_to_pol, normalize
from .orientation import get_elongation_plane_norm, align_with_z_quat, minimize_norm_to_survey, make_align_quat, get_slews_gti, get_photons_vectors, ObsClusters
from .sphere import ConvexHullonSphere, get_vecs_convex, get_convex_center, get_angle_betwee_three_vectors
from .containers import Urddata
from .filters import IndependentFilters
import numpy as np
from math import cos, sin, pi, sqrt, asin, acos
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation, Slerp
import matplotlib.pyplot as plt


def make_tan_wcs(rac, decc, sizex=1, sizey=1, pixsize=10./3600., alpha=0., lonpole=None, latpole=None):
    locwcs = WCS(naxis=2)
    locwcs.wcs.crpix = [sizex, sizey]
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]
    locwcs.wcs.cdelt = [pixsize, pixsize]
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.radesys = "FK5"
    if not lonpole is None:
        locwcs.wcs.lonpole = lonpole
    if not latpole is None:
        locwcs.wcs.latpole = latpole
    return locwcs

def convexhull_to_wcs(chull, alpha=None, cax=None, pixsize=10./3600., maxsize=False, landscape=False):
    cax = cax if not cax is None else chull.get_center_of_mass()
    ra, dec = vec_to_pol(cax)
    if alpha is None:
        alpha = minimize(lambda a: ConvexHullonSphere(chull.get_containing_rectangle(a, cax)).area, [0.,], method="Nelder-Mead").x[0]
    locwcs = make_tan_wcs(ra, dec, pixsize=pixsize)
    xy = locwcs.all_world2pix(np.rad2deg(vec_to_pol(chull.vertices)).T, 1).astype(np.int)
    #print(xy.shape)
    if maxsize:
        locwcs.wcs.crpix = (np.max(xy, axis=0) - np.min(xy, axis=0) + 1)//2 + [1, 1]
    else:
        locwcs.wcs.crpix = -np.min(xy, axis=0) + [1, 1]

    if landscape and locwcs.wcs.crpix[1] > locwcs.wcs.crpix[0]:
        locwcs = make_tan_wcs(ra, dec, locwcs.wcs.crpix[1], locwcs.wcs.crpix[0], pixsize, alpha + pi/2.)
    return locwcs

def minarea_ver2(vecs, pixsize=20./3600., lonpole=None, latpole=None):
    chull = ConvexHullonSphere(vecs)
    vm = chull.get_center_of_mass()
    rac, decc = vec_to_pol(vm)
    vfidx = np.argmin(np.sum(vm*chull.vertices, axis=-1))
    alpha = 0. #get_angle_betwee_three_vectors(vm, chull.vertices[vfidx], [0, 0, 1])
    size = int(np.arccos(np.sum(vm*chull.vertices[vfidx]))*180./pi/pixsize) + 2

    locwcs = WCS(naxis=2)
    locwcs.wcs.crpix = [size, size]
    locwcs.wcs.crval = [rac*180./pi, decc*180./pi]
    locwcs.wcs.cdelt = [pixsize, pixsize]
    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    locwcs.wcs.radesys = "FK5"
    if not lonpole is None:
        locwcs.wcs.lonpole = lonpole
    if not latpole is None:
        locwcs.wcs.latpole = latpole

    ra, dec = vec_to_pol(chull.vertices)
    x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
    sizex, sizey = int(x.max() - x.min()), int(y.max() - y.min())
    rac, decc = locwcs.all_pix2world([[(x.max() + x.min())/2., (y.max() + y.min())/2.],], 1)[0]
    locwcs.wcs.crval = [rac, decc]
    locwcs.wcs.crpix = [int(sizex//2) + 1 - int(sizex//2)%2, int(sizey//2) + 1 - int(sizey//2)%2]
    def minarea(var):
        alpha = var[0]
        cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
        locwcs.wcs.pc = cdmat
        x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
        return (x.max() - x.min())*(y.max() - y.min())
    alpha = minimize(minarea, [alpha,], method="Nelder-Mead").x[0]

    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T
    if (x.max() - x.min()) < (y.max() - y.min()):
        alpha = alpha + pi/2. if abs(alpha + pi/2.) < abs(alpha - pi/2.) else alpha - pi/2.

    cdmat = np.array([[cos(alpha), -sin(alpha)], [sin(alpha), cos(alpha)]])
    locwcs.wcs.pc = cdmat
    x, y = locwcs.all_world2pix(np.array([ra, dec]).T*180./pi, 1).T

    sizex = max(int(x.max() - locwcs.wcs.crpix[0]), int(locwcs.wcs.crpix[0] - x.min())) #int((x.max() - x.min())//2)
    sizey = max(int(y.max() - locwcs.wcs.crpix[1]), int(locwcs.wcs.crpix[1] - y.min())) #int((y.max() - y.min())//2)
    locwcs.wcs.crpix = [sizex + 1 - sizex%2, sizey + 1 - sizey%2]
    return locwcs



def wcs_qoffset(lwcs, dx, dy):
    dx = np.asarray(dx)
    dy = np.asarray(dy)
    vshift = pol_to_vec(*np.deg2rad(lwcs.all_pix2world(lwcs.wcs.crpix + np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), 0)).T)
    xshift = normalize(np.cross(vshift[0], vshift[1]))
    yshift = normalize(np.cross(vshift[2], vshift[3]))
    return Rotation.from_rotvec(xshift[np.newaxis, :]*dx[:, np.newaxis]*lwcs.wcs.cdelt[0]*pi/180.)*\
            Rotation.from_rotvec(yshift[np.newaxis, :]*dy[:, np.newaxis]*lwcs.wcs.cdelt[1]*pi/180.)



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
    #convex = ConvexHullonSphere(vecs)
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
    cvec = ConvexHullonSphere(vecs)
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
    rvec = (qalign*qvals).as_rotvec()
    addpart = np.zeros(rvec.shape[0], np.double)
    return np.sqrt(np.sum(rvec**2., axis=-1))*np.sign(np.sum(rvec*ax1, axis=-1))

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

def get_programms_gtis_and_wcss(attdata, join=True):
    slews = get_slews_gti(attdata)
    if join:
        v = raw_xy_to_vec(np.array([0, 0, 48, 48]), np.array([0, 48, 48, 0]))
        for i, arr in enumerate((~slews).arr):
            attl = attdata.apply_gti(arttools.time.GTI(arr))
            chull = ConvexHullonSphere(np.concatenate([attl.rotations.apply(v1) for v1 in v]))
            chulls.append(chull)

        clusters = np.arange(len(chulls))
        for i in range(len(chulls) - 1):
            ch1 = chulls[i]
            for j in range(i + 1,len(chulls)):
                ch2 = chulls[j]
                if np.any(ch1.check_inside_polygon(ch2.vertices)) or np.any(ch2.check_inside_polygon(ch1.vertices)):
                    clusters[(clusters == clusters[i]) | (clusters == clusters[j])] = min(clusters[i], clusters[j])

        gtis = [attdata.gti & GTI((~slews).arr[clusters == cluster]) for cluster in np.unique(clusters)]
    else:
        gtis = [attdata.gti & GTI(arr) for arr in (~slews).arr]


def estimate_optimal_wcs_grid(wcs):
    shape = [(0, int(wcs.wcs.crpix[1]*2 + 1)), (0, int(wcs.wcs.crpix[0]*2 + 1))]
    area = np.prod(np.diff(shape, axis=1).ravel()*wcs.wcs.cdelt)*(pi/180.)**2.
    print("area", area)
    vseg = 4.*(pi/180.)**2.  #characteristic fignetting function size
    n = area/vseg
    print("n", n)
    ratio = np.diff(shape, axis=1).ravel()*wcs.wcs.cdelt
    ratio = ratio[1]/ratio[0]
    snum = max(int(sqrt(n/ratio)), 1)
    bnum = max(int(n/snum), 1)
    print("ration snum bnum", ratio, snum, bnum)
    return shape, snum, bnum

def split_by_wcs(wcs, n=None, shape=None):
    if shape is None:
        shape = [(0, int(wcs.wcs.crpix[1]*2 + 1)), (0, int(wcs.wcs.crpix[0]*2 + 1))]
    if n is None:
        area = np.prod(np.diff(shape, axis=1).ravel()*wcs.wcs.cdelt)*(pi/180.)**2.
        print("area", area)
        vseg = 4.*(pi/180.)**2.  #characteristic vignetting function size
        n = area/vseg
        print("n", n)
    ratio = np.diff(shape, axis=1).ravel()*wcs.wcs.cdelt
    ratio = ratio[1]/ratio[0]
    snum = max(int(sqrt(n/ratio)), 1)
    bnum = max(int(n/snum), 1)
    print("ration", ratio, snum, bnum)
    x, y = np.linspace(shape[0][0], shape[0][1], snum + 1).astype(int), np.linspace(shape[1][0], shape[1][1], bnum + 1).astype(int)
    return [(x[i%snum:i%snum+2], y[i//snum:i//snum+2]) for i in range(snum*bnum)]


def cut_data_by_wcs(wcs, attdata, urdevt, shape=None, vreach = pi/180.*2/3., ereach=8./60.*pi/180.):
    if shape is None:
        shape = [(0, int(wcs.wcs.crpix[1]*2 + 1)), (0, int(wcs.wcs.crpix[0]*2 + 1))]
    v = wcs.all_pix2world(np.array(np.meshgrid(*shape)).T.reshape((-1, 2))[:,::-1], 0)
    ch = ConvexHullonSphere(pol_to_vec(*np.deg2rad(v).T))
    attloc = attdata.apply_gti(attdata.chull_gti(ch.expand(vreach)))
    ch = ch.expand(ereach)
    urdevt = {urdn: Urddata(d[ch.check_inside_polygon(get_photons_vectors(d, urdn, attdata))], urdn, d.filters & IndependentFilters({"TIME": attloc.gti})) for urdn, d in urdevt.items()}
    return attloc, urdevt

class WCSDataDistributer(object):
    def __init__(self, wcs, grid=None, shape=None):
        self.wcs = wcs
        self.grid = grid
        if grid is None:
            shape, snum, bnum = estimate_optimal_wcs_grid(wcs)
            self.grid = (np.linspace(shape[0][0], shape[0][1], snum + 1).astype(int), np.linspace(shape[1][0], shape[1][1], bnum + 1).astype(int))
        self.shapes =  [(tuple(self.grid[0][i%snum:i%snum+2]), tuple(self.grid[1][i//snum:i//snum+2])) for i in range(snum*bnum)]
        self.chulls = []
        for shape in self.shapes:
            vertices = wcs.all_pix2world([(y - 0.5, x - 0.5) for x in shape[0] for y in shape[1]], 0)
            self.chulls.append(ConvexHullonSphere(pol_to_vec(*np.deg2rad(vertices.T))))

        self.segments = {}
        self.key_cuts = {}

    def add(self, attdata, key):
        obs = ObsClusters()
        obs.add(attdata)

        for chull in obs.chulls:
            carea = 0.
            for shape, ch in zip(self.shapes, self.chulls):
                chi = ch & chull
                if not chi is None:
                    carea += chi.area
                    gti = attdata.chull_gti(ch.expand(pi/180.*30./60.))
                    self.segments[shape] = self.segments.get(shape, emptyGTI) | gti
                    self.key_cuts[key] = {*(list(self.key_cut.get(key, {})) + [shape,])}
                if abs(carea - chull.area) < 0.01:
                    break

    def get_shapes_gti(self):
        return self.segments



def make_wcs_cutouts(wcs, grid, vreach = pi/180.*2/3., ereach=8./60.*pi/180.):
    while True:
        attdata, key = yield


if __name__ == "__main__":
    pass
