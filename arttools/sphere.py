import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from math import sin, cos, pi, sqrt, acos, asin
from .vector import vec_to_pol, normalize


"""
some notes:
     center of mass of triangle on sphere
     center of mass is a point which separates equal mass with any cut of triangle by the stright line
     we can use followinf trick:
     if line runs from vertex it goes through the center of mass of any triangle, which have same opposit side as original one and vertix at this line
     the only line, wchich not moving for any triangle goes to the center of opposit side


"""

def check_big_circle_cross_circle(v1, v2, cax, rad):
    ort = np.cross(v1, v2)
    cosr = cos(rad)
    mask = np.sum(v1*cax, axis=1) > cosr
    mask = mask | (np.sum(v2*cax, axis=1) > cosr)

    vproj = normalize(cax - ort*np.sum(ort*cax, axis=1)[:, np.newaxis])
    mask = mask | ((np.sum(vproj*cax, axis=1) > cos(rad)) & (np.sum(np.cross(v1, vproj)*np.cross(vproj, v2), axis=-1) > 0))
    return mask


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

def triangle_center(vec1, vec2, vec3):
    return normalize(vec1 + vec2 + vec3)

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
    """
    return the index of most sepparated vector in set of vectors vecs relative to vector vec

    -----
    Params:
        vec - vector relative to which we search an index of most distant vector in a vector set vecs numpy array of shape 3
        vecs - a set of vectors, numpy array with shape Nx3

    returns:
        int - index of most distant to vector vec vector in a set vecs
    """
    return np.argmin(np.sum(vec*vecs, axis=1))

def get_most_orthogonal_idx(vec1, vec2, vecs):
    """
    return index of a vector with smallest normalized projection on a plane, which includes vectors vec1 and vec2
    also returns the orientation of this vector - True for left and False for right
    """
    vort = np.cross(vec1, vec2)
    proj = np.sum(vort*vecs, axis=1)
    idx = np.argmax(proj)
    return idx, proj > 0


def orient_triangle(vec1, vec2, vec3):
    if np.sum(np.cross(vec1, vec2)*vec3) < 0:
        vec1, vec2, vec3 = vec2, vec1, vec3
    return vec1, vec2, vec3

def get_outer(vec1, vec2, vecs):
    """
    considering that for all vectors np.cross(vec1, vec2)*vecs > 0 we search for one with
    maximal np.sum(norm(v - v1(v*v1))*np.cross(v1,v2))
    """
    return np.argmax(np.sum(normalize(vecs - vec1[np.newaxis, :]*np.sum(vec1*vecs, axis=1)[:, np.newaxis])*normalize(np.cross(vec1, vec2)), axis=1))

def get_all_outer(ovecs, vecs):
    idx = get_outer(ovecs[-1], ovecs[0], vecs)
    mask = np.sum(vecs*np.cross(vecs[idx], ovecs[0]), axis=1) > 0
    mask[idx]=False
    ovecs.append(vecs[idx])
    vecs = vecs[mask]
    if vecs.size > 0:
        return get_all_outer(ovecs, vecs)
    return ovecs


def get_outof_trinagle(vec1, vec2, vec3, vecs):
    """
    check, whether the vecs are outside of triangle with verteces vec1, vec2 and vec3
    returns True for vecs which are IN triangle with vec1, vec2, vec3 vertices
    """
    vec1, vec2, vec3 = orient_triangle(vec1, vec2, vec3)
    vort1 = np.cross(vec1, vec2)
    vort2 = np.cross(vec2, vec3)
    vort3 = np.cross(vec3, vec1)
    s1 = np.sum(vort1*vecs, axis=1)
    s2 = np.sum(vort2*vecs, axis=1)
    s3 = np.sum(vort3*vecs, axis=1)
    #return  np.any([s1*s2 < 0, s1*s3 < 0, s2*s3 < 0], axis=0)
    return np.all([s1 > 0, s2 > 0, s3 > 0], axis=0)

class ConvexHullonSphere(object):
    """
    class to define convex hull on the sphere surface
    """
    def __init__(self, vecs, parent=None):
        """
        init class with a series of vectors, which defines directions,
        all vectors will be reset to have 1 length
        after that class producess following attributres:
            vertices - unit vectors in the vertices of the convex hull, containing all vectors
            idx - indexes of the vectors, located in the corners, from  input array of vectors
        """
        #vecs = np.unique(vecs, axis=0)

        cvec = normalize(vecs.sum(axis=0))
        idx1 = get_most_distant_idx(cvec, vecs)
        idx2 = get_most_distant_idx(vecs[idx1], vecs)
        v1, v2 = vecs[idx1], vecs[idx2]
        mask = np.ones(vecs.shape[0], bool)
        mask[[idx1, idx2]] = False
        vecs = vecs[mask]
        ovecs = [v2, v1]
        m = np.sum(vecs*np.cross(v1, v2), axis=1) > 0.
        if np.any(m):
            ovecs = get_all_outer(ovecs, vecs[m])
        ovecs.append(ovecs.pop(0))
        #ovecs[0], ovecs[1], ovecs[-1] = ovecs[-1], ovecs[0], ovecs[0]
        m = np.sum(vecs*np.cross(v1, v2), axis=1) < 0.
        if np.any(m):
            ovecs = get_all_outer(ovecs, vecs[m])

        ovecs = np.array(ovecs)
        vnew = [ovecs[0]]
        i = 0
        while True:
            if ovecs.size == 0:
                break
            ovecs = ovecs[1:]
            ovecs = ovecs[1. - np.sum(ovecs*vnew[-1], axis=1) > 1e-15]
            if ovecs.size > 0:
                vnew.append(ovecs[0])
        ovecs = np.array(vnew)


        self.vertices = np.array(ovecs) #self.corners)
        idx = np.argmin(np.sum(self.vertices*np.roll(self.vertices, -1, axis=0), axis=1))
        self.vertices = np.roll(self.vertices, -idx - 1, axis=0)
        self.childs = [self,]
        if parent is None:
            self.parent = self
        else:
            self.parent = parent

    def all_hairs(self, res=None):
        if res is None:
            res = []
        if len(self.childs) == 1:
            res.append(self)
        else:
            for ch in self.childs:
                ch.all_hairs(res)
        return res


    def add_ort_vertices(self, da=1e-5):
        orts = self.orts
        cm = self.get_center_of_mass()
        das = np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1)
        cola = np.sum(orts*np.roll(orts, 1, axis=0), axis=1)
        if np.min(cola) > -cos(da):
            idxd = np.argmin(cola)
            nort = normalize(np.cross(self.vertices[idxd], orts[idxd]) + cm *das[idxd]*da)
            #nv = np.concatenate([[normalize(np.cross(orts[idxd], nort)), normalize(np.cross(nort, orts[idxd - 1]))], np.delete(self.vertices, idxd, axis=0)], axis=0)
            print(idxd)
            nverts = normalize(np.array([self.vertices[idxd] + da*self.vertices[idxd - 1], self.vertices[idxd] + da*self.vertices[idxd + 1 if idxd != self.vertices.shape[0] - 1 else -1]]))
            nv = np.concatenate([nverts, np.delete(self.vertices, idxd, axis=0)], axis=0)
            return self.__class__(nv)
        return self


    def join_vertices(self, frac=1e-3):
        da = np.arccos(np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1))
        if da.min() < np.max(da)*frac:
            idxd = np.argmin(da)
            nverts = np.concatenate([normalize(self.vertices[idxd] + self.vertices[idxd - 1]).reshape((1, 3)), np.delete(self.vertices, [idxd, idxd - 1], axis=0)])
            return self.__class__(nverts).join_vertices(frac)
        return self


    def simplify(self, da=1e-5):
        orts = self.orts
        das = np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1)
        cola = np.sum(orts*np.roll(orts, 1, axis=0), axis=1)
        #print("cola", cola, da, self.vertices.shape)
        if np.max(cola) > cos(da):
            idxd = np.argmax(cola)
            """
            print("removgind", idxd)
            if das[idxd] > das[idxd - 1]:
                orts = np.delete(orts, idxd - 1, axis=0)
                print("roll back")
            else:
                orts = np.delete(orts, idxd, axis=0)
            print("out roll shape", orts.shape)
            return self.__class__(normalize(np.cross(orts, np.roll(orts, 1, axis=0)))).simplify(da)
            """
            cnew = self.__class__.__new__(self.__class__)
            cnew.vertices = np.delete(self.vertices, idxd, axis=0)
            return cnew #cnew.simplify(da)

        if np.min(cola) < -cos(da):
            idx = np.argmin(cola)
            daa = (pi - np.arccos(cola[idx]))/2.*(1. - np.sum(self.vertices[idx]*self.vertices[[idx - 1, idx + 1 if idx != self.vertices.shape[0] else 0]], axis=1).max())

            print(daa, da)
            cm = self.get_center_of_mass()
            dvec = normalize(np.cross(self.vertices[idx], cm))
            cnew = self.__class__.__new__(self.__class__)
            vnew = np.empty((self.vertices.shape[0] + 1, 3), float)
            vnew[:idx, :] = self.vertices[:idx, :]
            vnew[idx] = self.vertices[idx]*cos(min(da, daa)) - sin(min(da, daa))*dvec
            vnew[idx + 1] = self.vertices[idx]*cos(min(da, daa)) + sin(min(da, daa))*dvec
            vnew[idx + 2:, :] = self.vertices[idx + 1:]
            cnew.vertices = vnew
            return cnew #.simplify(da)
        return self




    @property
    def orts(self):
        if self.vertices.shape[0] == 2:
            """
            in some cases, it apper to be userfull have a special case - a line as a convex hull
            scuh chull can be expanded in a rectangle
            """
            ort1 = normalize(np.cross(*self.vertices))
            return np.array([ort1, np.cross(self.vertices[0], ort1), -ort1, np.cross(self.vertices[1], -ort1)])

        return normalize(np.cross(self.vertices, np.roll(self.vertices, -1, axis=0)))

    def check_newpoint(self, vec1, vec3, color, lvl):
        if self.vecs.size == 0:
            return None
        idx, external = get_most_orthogonal_idx(vec1, vec3, self.vecs)
        if external[idx]:
            vec2 = self.vecs[idx]
            self.corners.append(vec2)
            lemask = get_outof_trinagle(vec1, vec2, vec3, self.vecs[external])
            mask = np.copy(~external)
            mask[external] = ~lemask
            mask[idx] = False
            self.pos = self.pos[mask]
            self.vecs = self.vecs[mask]
            self.check_newpoint(vec1, vec2, color, lvl-0.1)
            self.check_newpoint(vec2, vec3, color, lvl-0.1)

    def check_inside_polygon(self, vecs):
        return np.logical_not(np.any(np.sum(self.orts[np.newaxis, :, :]*vecs[:, np.newaxis, :], axis=2)  > 1e-15, axis=1)) # > 1e-15 instead of 0, because of new vertices lying at the previos edges


    def expand(self, dtheta):

        return self.__class__(self.expand5(dtheta))
        #return self.expand5(dtheta)

    def expand1(self, dtheta):
        """
        expand convex moving straight lines, limiting each side of the  convex, on the angle dtheta
        negative angle should be used with caution, since the sum of the angles on vertixes inside convex shrinks with convex surface area and some corners
        can appear on the oposite side of the sphere
        arguments:
            expand angle in radians
        """
        vproj = normalize(self.vertices + np.roll(self.vertices, -1, axis=0))
        neworts = self.orts*cos(dtheta) - vproj*sin(dtheta)
        newcorners = normalize(np.cross(neworts, np.roll(neworts, 1, axis=0)))
        return ConvexHullonSphere(newcorners[np.sum(newcorners*self.get_center_of_mass(), axis=1) > 0])

    def expand2(self, dtheta):
        """
        expand convex moving straight lines, limiting each side of the  convex, on the angle dtheta
        negative angle should be used with caution, since the sum of the angles on vertixes inside convex shrinks with convex surface area and some corners
        can appear on the oposite side of the sphere
        arguments:
            expand angle in radians
        """
        vproj = normalize(self.vertices + np.roll(self.vertices, -1, axis=0))
        neworts = self.orts*cos(dtheta) - vproj*sin(dtheta)

        if dtheta >0.:
            return ConvexHullonSphere(newcorners[np.sum(newcorners*self.get_center_of_mass(), axis=1) > 0])

        while True:
            mask = np.sum(np.cross(np.roll(neworts, -1, axis=0), np.roll(neworts, 1, axis=0))*neworts, axis=1) < 0
            neworts = neworts[mask]
            break
        newcorners = normalize(np.cross(neworts, np.roll(neworts, 1, axis=0)))
        return ConvexHullonSphere(newcorners[np.sum(newcorners*self.get_center_of_mass(), axis=1) > 0])

    def expand3(self, dtheta):
        """
        expand by moving orts from the center of mass,
        if negative expansion, than sides disapearing while smallest sides become revert oriented relative to rest of convex hull
        """
        if self.vertices.shape[0] == 1:
            raise ValueError("can't expand single point chull")


        #cm = normalize(self.vertices + np.roll(self.vertices, 1, axis=0)) #self.get_center_of_mass()
        cm = self.get_center_of_mass()
        orts = self.orts
        neworts = normalize(orts - normalize(cm + orts*np.sum(cm*orts, axis=1)[:, np.newaxis])*np.tan(dtheta))
        vertices = normalize(np.cross(neworts, np.roll(neworts, 1, axis=0)))
        print(vertices.shape, self.vertices.shape)
        mask = np.ones(vertices.shape[0], bool)

        #if dtheta < 0:
        if self.vertices.shape[0] > 2:
            while True:
                da = np.sum(self.vertices[mask]*np.roll(self.vertices[mask], 1, axis=0), axis=1)
                idx = np.argmax(da)
                print("idx", idx, da, np.arccos(da)*180/pi*60.)
                if np.sum(np.cross(vertices[idx], np.roll(vertices, 1, axis=0)[idx])*cm) < 0:
                    mask[idx - 1] = False
                    vertices = normalize(np.cross(neworts[mask], np.roll(neworts[mask], 1, axis=0)))
                else:
                    break
        """
        import matplotlib.pyplot as plt
        plt.scatter(*np.rad2deg(vec_to_pol(np.cross(neworts, np.roll(neworts, 1, axis=0)))))
        #return ConvexHullonSphere(normalize(np.cross(neworts, np.roll(neworts, 1, axis=0))))
        """
        return ConvexHullonSphere(vertices) #normalize(np.cross(neworts, np.roll(neworts, 1, axis=0))))


    def expand4(self, dtheta, tol=0.1):
        cm = self.get_center_of_mass()
        vnew = normalize(self.vertices - cm*np.sum(cm*self.vertices, axis=1)[:, np.newaxis])
        aold = np.arccos(np.sum(self.vertices*cm, axis=1))
        return self.__class__(vnew*np.sin(aold + dtheta)[:, np.newaxis] + cm[np.newaxis,:]*np.cos(aold + dtheta)[:, np.newaxis])

    def expand5(self, dtheta):
        orts = normalize(np.cross(self.vertices, np.roll(self.vertices, -1, axis=0))) # np.cross(self.vertices, np.roll(self.vertices, 1, axis=0)))
        #orts = normalize(np.cross(np.roll(self.vertices, -1, axis=0), self.vertices)) # np.cross(self.vertices, np.roll(self.vertices, 1, axis=0)))
        da = np.sum(self.vertices*np.roll(self.vertices, -1, axis=0), axis=1) # cos 2\alpha
        dtheta = dtheta/np.sqrt((da + 1.)/2.)
        vpro = normalize(self.vertices + np.roll(self.vertices, -1, axis=0))
        neworts = orts*np.cos(dtheta)[:, np.newaxis] - vpro*np.sin(dtheta)[:, np.newaxis]
        cm = self.get_center_of_mass()
        while True:
            """
            check is the side, produced by particular ort (orthogonal to it) is consumed by the neighbouring orts, if it does, remove it
            """
            mask = np.sum(np.roll(neworts, 1, axis=0)*np.cross(neworts, np.roll(neworts, 2, axis=0)), axis=1) < 0.
            if ~np.any(mask):
                break
            neworts = neworts[~mask]

        newverts = normalize(np.cross(neworts, np.roll(neworts, 1, axis=0)))
        return newverts
        #return self.__class__(newverts)


    @property
    def area(self):
        if self.vertices.shape[0] > 2:
            return get_vec_triangle_area(self.vertices[np.zeros(self.vertices.shape[0] - 2).astype(np.int)],
                                     self.vertices[1:-1],
                                     self.vertices[2:]).sum()*(180./pi)**2.
        else:
            return 0.

    def get_center_of_mass(self):
        clist = triangle_center(np.tile(self.vertices[0], (self.vertices.shape[0] - 2, 1)), self.vertices[1:-1], self.vertices[2:])
        mlist = get_vec_triangle_area(np.tile(self.vertices[0], (self.vertices.shape[0] - 2, 1)), self.vertices[1:-1], self.vertices[2:])
        return normalize(np.sum(clist*mlist[:, np.newaxis], axis=0))
        """
        if self.vertices.shape[0] == 2:
            return normalize(np.sum(self.vertices, axis=0))
        return get_convex_center(self)
        """

    def get_containing_rectangle(self, alpha=0., rotc=None):
        vm = rotc if not rotc is None else self.get_center_of_mass()
        qrot = Rotation.from_rotvec(vm*alpha)
        vax = qrot.apply([0, 0, 1])
        vax = normalize(vax - vm*np.sum(vax*vm))
        vrot = normalize(np.cross(vax, vm))
        vr = normalize(self.vertices - vrot*np.sum(self.vertices*vrot, axis=1)[:, np.newaxis])
        vp = normalize(self.vertices - vax*np.sum(self.vertices*vax, axis=1)[:, np.newaxis])
        v1 = vr[np.argmax(np.sum(np.cross(vm,vr)*vrot, axis=1))]
        v2 = vr[np.argmax(np.sum(np.cross(vr,vm)*vrot, axis=1))]
        v3 = vp[np.argmax(np.sum(np.cross(vm,vp)*vax, axis=1))]
        v4 = vp[np.argmax(np.sum(np.cross(vp,vm)*vax, axis=1))]
        v1 = Rotation.from_rotvec(-vrot*pi/2.).apply(v1)
        v2 = Rotation.from_rotvec(vrot*pi/2.).apply(v2)
        v3 = Rotation.from_rotvec(-vax*pi/2.).apply(v3)
        v4 = Rotation.from_rotvec(vax*pi/2.).apply(v4)
        return switch_between_corners_to_plane_axis([v1, v4, v2, v3])

    def __add__(self, other):
        if not isinstance(other, self.__class__):
            raise ValueError("one of the arguments is not  an ConvexHullonSphere instance")
        return self.__class__(np.concatenate([self.vertices, other.vertices]))

    def get_wcs(self, delta, alpha=None):
        if alpha is None:
            res = minimize(lambda alpha: self.__class__(self.get_containing_rectangle(alpha)).area, 0.)
            alpha = res.x[0]
        corners = self.get_containing_rectangle(alpha)
        sizex = int(np.arccos(np.sum(corners[0]*corners[1]))*180/pi/delta)
        sizey = int(np.arccos(np.sum(corners[0]*corners[-1]))*180/pi/delta)
        ra, dec = vec_to_pol(self.__class__(corners).get_center_of_mass())
        return make_tan_wcs(ra, dec, sizex//2 + (sizex//2)%2 - 1, sizey//2 + (sizey//2)%2 - 1, delta, alpha)

    def triangulate(self, sarea):
        cm = self.get_center_of_mass()
        triangles = []
        for v1, v2 in zip(self.vertices, np.roll(self.vertices, 1, axis=0)):
            split_uptoarea([cm, v1, v2], sarea, triangles)
        return triangles

    def split_in_pices(self, sarea):
        if self.area > 2.*sarea:
            if len(self.childs) > 1:
                for ch in self.childs:
                    ch.split_in_pices(sarea)
            else:
                if self.vertices.shape[0] == 3:
                    split_triangle_chull(self)
                else:
                    da = np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1)
                    idx = np.argmin(da)
                    cm = self.get_center_of_mass()
                    m1 = (np.sum(np.cross(self.vertices[idx], cm)*self.vertices, axis=1) > 0.)
                    m2 = (np.sum(np.cross(cm, self.vertices[idx-1])*self.vertices, axis=1) > 0)
                    print(m1, m2)
                    mask = (np.sum(np.cross(self.vertices[idx], cm)*self.vertices, axis=1) > 0.) & (np.sum(np.cross(cm, self.vertices[idx-1])*self.vertices, axis=1) > 0)
                    dc = np.sum(self.vertices*cm, axis=1)
                    dc[mask] = dc[mask] + 3
                    idxo = np.argmax(dc)
                    print(idx, idxo, mask)


                    vnew = normalize(np.cross(np.cross(self.vertices[idx-1], self.vertices[idx]), np.cross(self.vertices[idxo], cm))) #) + np.roll(v, 1, axis=0)[idx])
                    print(vnew)
                    istart = idxo - idx + 1 if idxo > idx else self.vertices.shape[0] - idx + idxo + 1
                    chnew1 = self.__class__(np.array([vnew,] + list(np.roll(self.vertices, -idx, axis=0)[:istart])))
                    chnew2 = self.__class__(np.array(list(np.roll(self.vertices, -idx, axis=0)[istart -1:]) + [vnew,]))
                    #chnew2 = self.__class__(np.array([vnew,] + list(np.roll(self.vertices, -idxo))[idx - idxo:]))
                    return chnew1, chnew2
                    return [[vnew,] + list(np.roll(v, -idx, axis=0)[:2]), list(np.roll(v, -idx, axis=0)[:2]) + [vnew,]]

                    self.childs = [ConvexHullonSphere(np.array([cm, v1, v2]), self) for v1, v2 in zip(self.vertices, np.roll(self.vertices, 1, axis=0))]
                for ch in self.childs:
                    ch.split_in_pices(sarea)




    def join_small_childs(self, sarea):
        while True:
            mergesuccess = False
            for h in self.all_hairs():
                if h not in self.all_hairs():
                    continue
                if h.area < sarea:
                    mergesuccess = h.merge(h.vertices)
            if not mergesuccess:
                break

    def remove_short_sights(self, dalpha):
        cm = self.get_center_of_mass()
        v = normalize(self.vertices - cm[np.newaxis,:]*np.sum(self.vertices*cm, axis=1)[:, np.newaxis])
        alpha = np.arccos(np.sum(v*np.roll(v, 1, axis=0), axis=1))
        if alpha.min() < dalpha:
            cnew = self.remove_shortest_sights()
            return cnew.remove_short_sights(dalpha)
        else:
            return self.__class__(self.vertices)

    def get_shortest_side_size(self):
        da = np.arccos(np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1))
        return np.min(da)

    def get_shortest_side_index(self):
        da = np.arccos(np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1))
        return np.argmin(da)


    def remove_shortest_sights(self):
        cm = self.get_center_of_mass()
        v = np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1)
        idx = np.argmax(v)
        mask = np.ones(self.vertices.shape[0], bool)
        mask[idx] = False
        return self.__class__(normalize(np.cross(self.orts[mask], np.roll(self.orts[mask], 1, axis=0))))

    def intersect(self, other):
        m1 = self.check_inside_polygon(other.vertices)
        if np.any(m1):
            return True
        m2 = other.check_inside_polygon(self.vertices)
        if np.any(m2):
            return True
        ccov = self & other
        return False if ccov is None else True

    def __and__(self, other):
        m1 = self.check_inside_polygon(other.vertices)
        m2 = other.check_inside_polygon(self.vertices)
        if np.all(m1):
            return ConvexHullonSphere(other.vertices)
        if np.all(m2):
            return ConvexHullonSphere(self.vertices)

        cm = normalize(self.get_center_of_mass()*self.area + other.get_center_of_mass()*other.area)
        vlist = [self.vertices[m2], other.vertices[m1]]

        o1 = self.orts
        o2 = other.orts

        c1 = self.expand(1e-7)
        c2 = other.expand(1e-7)


        for i in range(o1.shape[0]):
            vecs = normalize(np.cross(o1[i], o2))
            vlist.append(vecs[c1.check_inside_polygon(vecs) & c2.check_inside_polygon(vecs)])
            vlist.append(-vecs[c1.check_inside_polygon(-vecs) & c2.check_inside_polygon(-vecs)])
        vlist = np.concatenate(vlist, axis=0)
        return None if vlist.shape[0] < 3 else ConvexHullonSphere(vlist)

    def __getitem__(self, idx):
        return self.childs[idx]

    def __iter__(self):
        return iter(self.childs)


    def get_all_child_intersect(self, other, res=None):
        if res is None:
            res = []
        isect = self & other
        if not isect is None:
            if len(self.childs) == 1:
                res.append(self)
            else:
                for child in self.childs:
                    child.get_all_child_intersect(other, res)
        return res

    def check_crossing_circle(self, ax, radius):
        return np.any(check_big_circle_cross_circle(self.vertices, np.roll(self.vertices, 1, axis=0), ax, radius)) | self.check_inside_polygon(ax.reshape((-1, 3)))

    def get_all_childs_crossing_circle(self, ax, radius, res=None):
        if res is None:
            res = []
        if self.check_crossing_circle(ax, radius):
            if len(self.childs) == 1:
                res.append(self)
            else:
                for child in self.childs:
                    child.get_all_childs_crossing_circle(ax, radius, res)
        return res


    def merge(self, vertices=None):
        if vertices is None:
            vertices = self.vertices
        if self.parent != self:
            for k, ch in enumerate(self.parent.childs):
                if np.sum(1 - np.sum(vertices[:, np.newaxis,:]*ch.vertices[np.newaxis, :, :], axis=2).max(axis=1) < 1e-15) != 2:
                    continue
                chnew = ConvexHullonSphere(np.concatenate([ch.vertices, self.vertices]), self.parent)
                if chnew.vertices.shape[0] != self.vertices.shape[0] + ch.vertices.shape[0] - 2:
                    continue
                chnew.childs = self.childs + ch.childs
                idx = self.parent.childs.index(self)
                self.parent.childs[idx] = chnew
                self.parent.childs.pop(k)
                for ch in chnew.childs:
                    ch.parent = chnew
                return chnew
        return None

    def has_common_sides(self, other):
        return np.sum(1 - np.sum(other.vertices[:, np.newaxis,:]*self.vertices[np.newaxis, :, :], axis=2).max(axis=1) < 1e-15) == 2

    def stick_childs(self, sarea):
        hairs = self.all_hairs()
        print("len of all childs", len(hairs))
        for i in range(len(hairs)):
            h = hairs[i] #, h in enumerate(hairs):
            if h.area > sarea:
                continue
            ict = [k for k in range(len(h.parent.childs)) if h.parent.childs[k] != h and h.parent.childs[k] in hairs and h.has_common_sides(h.parent.childs[k])]
            if len(ict) == 0:
                chnew = h.merge(h.vertices)
                if not chnew is None:
                    self.stick_childs(sarea)
                    if h in chnew.childs:
                        chnew.childs = [chnew,]
                        self.stick_childs(sarea)
                        break
                else:
                    continue
            kloc = ict[np.argmin([h.parent.childs[k].area for k in ict])]
            k = hairs.index(h.parent.childs[kloc])
            idx = h.parent.childs.index(h)
            print("stick ", k, idx, "parent area", h.parent.area, h.area, hairs[k].area)
            chnew = self.__class__(np.concatenate([h.vertices, hairs[k].vertices]), h.parent)
            if kloc < idx:
                h.parent.childs[kloc] = chnew
                h.parent.childs.pop(idx)
            else:
                h.parent.childs[idx] = chnew
                h.parent.childs.pop(kloc)
            return self.stick_childs(sarea) #, i if k > i else i - 1)
            break

    def split_on_equal_segments(self, sarea):

        if len(self.childs) > 1:
            for ch in self.childs:
                ch.split_on_equal_segments(sarea)
        else:
            if self.area > sarea*1.5:
                if self.vertices.shape[0] == 3:
                    split_triangle_chull(self)
                else:
                    da = np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1)
                    idx = np.argmin(da)
                    cm = self.get_center_of_mass()
                    m1 = (np.sum(np.cross(self.vertices[idx], cm)*self.vertices, axis=1) > 0.)
                    m2 = (np.sum(np.cross(cm, self.vertices[idx-1])*self.vertices, axis=1) > 0)
                    mask = (np.sum(np.cross(self.vertices[idx], cm)*self.vertices, axis=1) > 0.) & (np.sum(np.cross(cm, self.vertices[idx-1])*self.vertices, axis=1) > 0)
                    dc = np.sum(self.vertices*cm, axis=1)
                    dc[mask] = dc[mask] + 2
                    idxo = np.argmax(dc)
                    vnew = normalize(np.cross(np.cross(self.vertices[idx-1], self.vertices[idx]), np.cross(self.vertices[idxo], cm))) #) + np.roll(v, 1, axis=0)[idx])
                    istart = idxo - idx + 1 if idxo > idx else self.vertices.shape[0] - idx + idxo + 1
                    print(self.__class__)
                    chnew1 = self.__class__(np.array([vnew,] + list(np.roll(self.vertices, -idx, axis=0)[:istart])), self)
                    chnew2 = self.__class__(np.array(list(np.roll(self.vertices, -idx, axis=0)[istart -1:]) + [vnew,]), self)
                    self.childs = [chnew1, chnew2]
                for ch in self.childs:
                    ch.split_on_equal_segments(sarea)

                """
                cm = self.get_center_of_mass()
                idx1 = np.argmin(np.sum(self.vertices*np.roll(self.vertices, 1, axis=0), axis=1))
                v1 = normalize(self.vertices[idx1] + np.roll(self.vertices, 1, axis=0)[idx1])
                ort = normalize(np.cross(cm, v1))
                m = np.sum(ort*self.vertices, axis=1) > 0
                idx2 = np.argwhere(m & ~np.roll(m, 1, axis=0))[0][0]
                v2 = normalize(np.cross(ort, np.roll(self.orts, 1, axis=0)[idx2]))
                size = idx2 - idx1 if idx2 > idx1 else self.vertices.shape[0] - (idx1 - idx2)
                ch1 = self.__class__(np.concatenate([[v1,], np.roll(self.vertices, -idx1, 0)[:size], [v2,]], axis=0), self)
                ch2 = self.__class__(np.concatenate([[v2,], np.roll(self.vertices, -idx1, 0)[size:], [v1,]], axis=0), self)
                self.childs = [ch1, ch2]
                for ch in self.childs:
                    ch.split_on_equal_segments(sarea)
                """


def split_triangle(v):
    #da = np.sum(np.roll(v, 1, axis=0)*np.roll(v, 2, axis=0), axis=1)
    cosb = np.sum(v*np.roll(v, 1, axis=0), axis=1)
    orts = normalize(np.cross(v, np.roll(v, -1, axis=0)))
    angles = np.arccos(np.sum(orts*-np.roll(orts, 1, axis=0), axis=1))
    idx = np.argmax(angles)
    a1 = np.sum(angles*np.roll([1, 1, -1], idx))/2.
    #print(a1, (angles[idx] + angles[idx-2] - angles[idx-1])/2.)
    x = angles[idx] - np.arctan((cos(angles[idx-1]) - sin(a1))/(sin(angles[idx-1])*cosb[idx] - cos(a1)))
    """
    cm = triangle_center(v[0], v[1], v[2])
    vnew = normalize(np.cross(np.cross(v[idx-2], v[idx-1]), np.cross(v[idx], cm))) #) + np.roll(v, 1, axis=0)[idx])
    return [[vnew,] + list(np.roll(v, -idx, axis=0)[:2]), list(np.roll(v, -idx+1, axis=0)[:2]) + [vnew,]]
    """
    #print(orts)
    #print("angles", angles, idx, np.arccos(-np.sum(orts[idx]*orts[idx-1])))
    #print(x, angles[idx])
    o1 = -normalize(orts[idx-1] - orts[idx]*np.sum(orts[idx]*orts[idx-1]))
    onew = orts[idx]*cos(x) + o1*sin(x)
    #print("check vectors", np.sum(onew**2), np.sum(o1*np.cross(orts[idx], orts[idx-1])))
    #print("check angle", np.arccos(np.sum(onew*orts[idx])), np.arccos(-np.sum(onew*orts[idx-1])))
    #return onew, normalize(np.cross(orts[idx-2], orts[idx])), normalize(np.cross(onew, orts[idx])), normalize(np.cross(orts[idx-2], onew)), normalize(np.cross(onew, orts[idx-1]))
    #============================================================================================================

    vnew = normalize(np.cross(orts[idx-2], onew))
    #return [list(np.roll(v, -idx, axis=0)[:2]) + [vnew,], [vnew,] + list(np.roll(v, -idx-1, axis=0)[:2])]
    return [list(np.roll(v, -idx, axis=0)[:2]) + [vnew,], [vnew,] + list(np.roll(v, -idx+1, axis=0)[:2])]

def split_triangle2(v):
    da = np.sum(v*np.roll(v, 1, axis=0))
    idx = np.argmin(da)

    v12 = np.cross(v[0], v[1], axis=-1)
    v13 = np.cross(v[0], v[2], axis=-1)
    v23 = np.cross(v[1], v[2], axis=-1)
    v12 = v12/np.sqrt(np.sum(v12**2, axis=-1))[:, np.newaxis]
    v13 = v13/np.sqrt(np.sum(v13**2, axis=-1))[:, np.newaxis]
    v23 = v23/np.sqrt(np.sum(v23**2, axis=-1))[:, np.newaxis]
    A = np.arccos(np.sum(v12*v13, axis=-1))
    B = np.arccos(-np.sum(v12*v23, axis=-1))
    C = np.arccos(np.sum(v23*v13, axis=-1))

    x = np.arctan((cos(B) - sin((A + B - C)/2.))/(sin(B)*np.sum(v[0]*v[1]) - cos((A + B - C)/2.)))
    v14 = v12*x/A + v

    da = np.sum(v*np.roll(v, 1, axis=0))
    idx = np.argmin(da)
    vnew = normalize(v[idx] + np.roll(v, 1, axis=0)[idx])
    return [[vnew,] + list(np.roll(v, idx, axis=0)[:2]), list(np.roll(v, idx + 2, axis=0)[:2]) + [vnew,]]



def split_triangle_chull(ch):
    v = ch.vertices
    vnew = split_triangle(v)
    chnew = [ch.__class__(np.array(v), ch) for v in vnew]
    ch.childs = chnew
    return ch.childs


def split_uptoarea(ch, area):
    if ch.area > area*1.5:
        for chnew in split_triangle_chull(chnew):
            split_uptoarea(chnew, area)


def stick_chulls(chulls, sarea):
    for i in range(len(chulls)):
        if i == len(chulls):
            break
        if chulls[i].area > sarea:
            continue
        ict = [k for k in range(len(chulls)) if k != i and np.sum(1 - np.sum(chulls[i].vertices[:, np.newaxis,:]*chulls[k].vertices[np.newaxis, :, :], axis=2).max(axis=1) < 1e-15) == 2]
        if len(ict) == 0:
            continue
        k = ict[np.argmin([chulls[k].area for k in ict])]
        if k < i:
            t1 = chulls.pop(i)
            chulls[k] = ConvexHullonSphere(np.concatenate([chulls[k].vertices, t1.vertices]))
            i -= 1
        else:
            t1 = chulls.pop(k)
            chulls[i] = ConvexHullonSphere(np.concatenate([chulls[i].vertices, t1.vertices]))
    return chulls


def split_chull_in_pieces(ch, sarea=4., join=0.1):
    if ch.area > 2.*sarea:
        sa = int(ch.area/sarea)
        chnew = ConvexHullonSphere(ch.vertices)
        while chnew.vertices.shape[0] > 3:
            tareas = get_vec_triangle_area(*np.copy(np.swapaxes(np.array(chnew.triangulate(chnew.area)), 0, 1)))
            if tareas.min() < sarea*(pi/180.)**2.:
                chnew = chnew.remove_shortest_sights()
            else:
                break
        triangles = [ConvexHullonSphere(t) & ch for t in chnew.triangulate(sarea*(pi/180.)**2.)]
        triangles = [t for t in triangles if not t is None]
        triangles = stick_chulls(triangles, sarea*join)
        return triangles
    else:
        return ConvexHullonSphere(ch.vertices)

"""
lets cover sphere with four equal triangles
for that one should find
[1, 0, 0]
[cos(alpha), sin(alpha), 0]
[cos(alpha), sin(alpha) cos(2pi/3), sin(alpha)*sin(2pi/3)]
cos(alpha) =  cos^2a + sin^2a *cos(2pi/3)
cos a = cos^2a - 1/2 (1 - cos^2a)
cos a = 3/2cos^2a - 1/2
3 cos^2a - 2cos a - 1 = 0
cos^2a - 2 1/3cos a + 1/9 = 1/3 + 1/9
(cos a - 1/3)^2 = 4/9
cos a = +- 2/3 + 1/3
"""

CVERTICES = [[1, 0, 0],
             [-1/3., 0., 2.*sqrt(2.)/3.],
             [-1/3., sqrt(2/3.), -sqrt(2.)/3.],
             [-1/3., -sqrt(2/3.), -sqrt(2.)/3.,]]


class FullSphere(ConvexHullonSphere):
    def __init__(self):
        self.childs = [ConvexHullonSphere(np.roll(CVERTICES, -i, axis=0)[:3], self) for i in range(4)]
        self.parent = [self,]
        self.vertices = np.empty((0, 3), float)

    @property
    def area(self):
        return 4.*pi*(180/pi)**2.

    def check_inside_polygon(self, vecs):
        return np.ones(vecs.shape[0], bool)

    def __and__(self, other):
        return ConvexHullonSphere(other.vertices)

SPHERE = FullSphere()


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
    """
    return an area of the smallest (area < 2pi) triangle on sphere, which has corners angles alpha, beta, gamma
    """
    return  alpha - beta - gamma - pi

def get_angle_betwee_three_vectors(vec1, vec2, vec3):
    v12 = np.cross(vec1, vec2, axis=-1)
    v13 = np.cross(vec1, vec3, axis=-1)
    v23 = np.cross(vec2, vec3, axis=-1)
    v12 = 1/np.sqrt(np.sum(v12**2, axis=-1))*v12
    v13 = 1/np.sqrt(np.sum(v13**2, axis=-1))*v13
    alpha = np.arccos(np.sum(v12*v13, axis=-1))
    return alpha


def get_convex_center(convex):
    return get_convex_center1(convex)

def get_convex_center1(convex):
    """
    finds a center of mass of a convex on a sphere

    algorithm
    sum individual triangles centers of mass according to their surface arae
    on the first step triangles are picked from one of the convex hull vertices.
    after first iteration triangles are picked from current center of mass assessment.


    -------
    Params:
        convexhullonsphere class instance
    returns:
        unit vector at the convex hull center of mass location
    """
    areas = get_vec_triangle_area(convex.vertices[0], convex.vertices[1:-1], convex.vertices[2:])
    idx = np.searchsorted(np.cumsum(areas), np.sum(areas)/2.)
    k = (np.sum(areas)/2. - np.sum(areas[:idx]))/areas[idx]
    v1 = normalize(convex.vertices[idx + 1]*(1 - k) + k*convex.vertices[idx + 2])
    areas = get_vec_triangle_area(convex.vertices[-1], convex.vertices[:-2], convex.vertices[1:-1])
    idx = np.searchsorted(np.cumsum(areas), areas.sum()/2.)
    k = (np.sum(areas)/2. - np.sum(areas[:idx]))/areas[idx]
    v2 = normalize(convex.vertices[idx]*(1 - k) + k*convex.vertices[idx + 1])

    c1 = normalize(np.cross(np.cross(convex.vertices[0], v1), np.cross(convex.vertices[-1], v2)))
    return c1

def get_convex_center2(convex):
    v = convex.vertices
    return normalize(np.sum(normalize(v + np.roll(v, 1, axis=0))*np.arccos(np.sum(v*np.roll(v, 1, axis=0), axis=1))[:, np.newaxis], axis=0))


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
    convexhull = ConvexHullonSphere(vertices)
    corners = np.cross(corners_or_axis, np.roll(corners_or_axis, -1, axis=0))

    get_convex_center(corners)

    alpha = acos(-np.sum(axis[0]*axis[2]))
    beta = acos(-np.sum(axis[1]*axis[3]))
    newax = np.array([axis[0]*cos(dtheta) + (axis[2] - cos(alpha)*axis[0])*sin(dtheta)/sin(alpha),
                     axis[1]*cos(dtheta) + (axis[3] - cos(beta)*axis[1])*sin(dtheta)/sin(beta),
                     axis[2]*cos(dtheta) + (axis[0] - cos(alpha)*axis[2])*sin(dtheta)/sin(alpha),
                     axis[3]*cos(dtheta) + (axis[1] - cos(beta)*axis[3])*sin(dtheta)/sin(beta),])
    return newax/np.sqrt(np.sum(newax**2, axis=1))[:, np.newaxis]
