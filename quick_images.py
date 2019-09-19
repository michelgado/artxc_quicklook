import numpy as np
import arttools
from astropy.io import fits
import sys, os
from math import pi
import re
from scipy.spatial import ConvexHull, distance
from math import sqrt, pi, acos

print("arttools", arttools)

"""
class Aquisition(object):
    def __init__(self, vecs):
        self.vecs = vecs
        self.vmean = vecs.sum(axis=0)
        self.vmean = self.vmean/sqrt(np.sum(self.vmean**2))
"""


if __name__ == "__main__":
    l0path = sys.argv[1]
    urdfnames = [l.rstrip() for l in os.listdir(l0path) if "urd.fits" in l]
    urdgroups = {}
    for name in urdfnames:
        session = name[:19]
        urdgroups[session] = urdgroups.get(session, []) + [name[19:], ]

    for session in urdgroups:
        attnames = [os.path.join(l0path, l) for l in os.listdir(l0path) \
                    if session in l and "_gyro.fits" in l]
        attdata = np.concatenate([fits.getdata(attname, "ORIENTATION") for attname in attnames])
        attdata.sort(order="TIME")
        ts, dt, dalphadt = arttools.time.get_axis_movement_speed(attdata)
        """
        import matplotlib.pyplot as plt
        plt.scatter(ts, dalphadt)
        plt.show()
        """
        m1 = dalphadt < 120.
        m1[np.isnan(dalphadt)] = True
        idx = arttools.mask.edges(m1)
        idx = idx[idx[:,1] - idx[:,0] > 1]
        gti = attdata['TIME'][idx]

        vecs = arttools.orientation.get_gyro_quat(attdata).apply([1,0,0])
        coordlist = [vecs[idx[i,0]:idx[i,1]] for i in range(idx.shape[0])]


        urdnames = [os.path.join(l0path, session + name) for name in urdgroups[session]]
        attnames = [urdname[:-12] + "_gyro.fits" for urdname in urdnames]


        print(attnames, urdnames)

        for i in range(gti.shape[0]):
            print(gti[i:i+1])
            lgti = {urd: gti[i: i+1,:] for urd in arttools.telescope.URDNS}
            arttools.plot.make_mosaic_for_urdset_by_gti(urdnames, attnames, gti=lgti)
            os.rename("tmpctmap.fits.gz", "%s_ctmap_%04d.fits.gz" % (session, i))
            os.rename("tmpemap.fits.gz", "%s_exmap_%04d.fits.gz" % (session, i))

