import numpy as np
from math import sin, cos, pi, sqrt

def to_2pi_range(val): return val%(2.*pi)

def normalize(vecs):
    if vecs.ndim > 1:
        return vecs/np.sqrt(np.sum(vecs**2., axis=-1)[..., np.newaxis])
    else:
        return vecs/sqrt(np.sum(vecs**2))


def get_otrhogonal_vector(ax1, ax2=None):
    """
    producess orthogonal vector to ax1 in plane ax1, ax2
    if ax2 is None, then random vector not colinear to ax1 will be peacked
    """
    ax2 = np.copy(ax1)
    i, j = np.argmax(ax1), np.argmin(ax2)
    ax2[[i, j]] = ax1[[j, i]]
    ax1 = normalize(ax1)
    ax2 = normalize(ax2)
    if np.sum(ax1*ax2) == 1:
        raise ValueError("ax1 colinear to ax2")
    return normalize(ax2 - ax1*np.sum(ax1*ax2, axis=-1))


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


class Vector3d(object):
    def __init__(self, arr):
        if not arr.shape[-1] == 3:
            raise ValueError("vector should have shape ...., 3")
        self._data = arr/normalize(arr)
        self._norm = np.sqrt(np.sum(arr**2, axis=-1))
        self._theta, self._phi = vec_to_pol(self.data)

    @property
    def vec(self):
        return self._data

    @property
    def norm(self):
        return self._norm

    @property
    def phi(self):
        return self._phi

    @property
    def theta(self):
        return self._theta

    def angle(self, other):
        return np.sum(self.data*other.data, axis=-1)



