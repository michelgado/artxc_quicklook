import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from astropy.visualization import MinMaxInterval, SqrtStretch, ImageNormalize, LogStretch
from astropy.wcs import WCS
from .orientation import extract_raw_gyro, qrot0, ART_det_QUAT, \
        get_gyro_quat, filter_gyrodata, vec_to_pol
from ._det_spatial import raw_xy_to_vec, offset_to_vec, urd_to_vec
from math import pi, cos, sin
from multiprocessing import Pool, cpu_count
import copy


def get_rawxy_hist(data):
    img = np.histogram2d(data['RAW_X'], data['RAW_Y'], [np.arange(49) - 0.5,]*2)[0]
    return img

def get_sky_image(urddata, URDN, attdata, xe, ye, subscale=1):
    attdata = filter_gyrodata(attdata)
    qj2000 = Slerp(attdata["TIME"], get_gyro_quat(attdata))
    qj2000 = qj2000(np.repeat(urddata["TIME"], subscale*subscale))
    qall = qj2000*qrot0*ART_det_QUAT[URDN]

    photonvecs = urd_to_vec(urddata, subscale)
    phvec = qall.apply(photonvecs)
    dec = np.arctan(phvec[:,2]/np.sqrt(phvec[:,0]**2. + phvec[:,1]**2.))*180./pi
    ra = (np.arctan2(phvec[:,1], phvec[:,0])%(2.*pi))*180./pi
    
    return np.histogram2d(dec, ra, [ye, xe])[0]

def mktimegrid(gti):
    ts = np.arange(gti[0], gti[1] + 0.9, 1.)
    ts[-1] = gti[1]
    tc = (ts[1:] + ts[:-1])/2.
    dt = ts[1:] - ts[:-1]
    return tc, dt

def make_vec_to_sky_hist_fun(vecs, effarea, locwcs, xsize, ysize):
    def hist_vec_to_sky(quat, weight):
        vec_icrs = quat.apply(vecs)
        r, d = vec_to_pol(vec)
        x, y = locwcs.all_world2pix(np.array([r, d]).T, 1).T
        return np.histogram(x, y, [np.arange(xsize), np.arange(ysize)])[0].T
    return hist_vec_to_sky

def vignmap(vignmapfile, locwcs, xsize, ysize, qval, exptime, energy=6.):
    """
    to do: implement mpi reduce
    """

    """
    you should be very cautiont about the memory usage here, 
    we expect, that array is stored in c order - 
    effarea[i, j](in 2d representation) = effarea[i*ysize + j]
    """
    effarea = np.ravel(vignmapfile["Vign_EA"].data["EFFAREA"][
            np.searchsorted(vignmapfile["Vign_EA"].data["E"], energy)])
    #xoffset = np.repeat(vignmapfile["Coord"].data["X"])
    """
    xoffset, yoffset = np.meshgrid(vignmapfile["Coord"].data["X"], vignmapfile["Coord"].data["Y"])
    vecs = offset_to_vec(xoffset, yoffset)
    worker = make_vec_to_sky_hist_fun(vecs, locwcs, xsize, ysize)
    hist = sum(worker(q, exp*effarea(vecs[i%size1d, i//size1d, :], locwcs, xsize, ysize)*\
            effarea[i%size1d, i//size1d]*for i in range(size2d))
    return hist
    """
    

if __name__ == "__main__":
    pass
