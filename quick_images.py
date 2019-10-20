import numpy as np
import arttools
from astropy.io import fits
import sys, os
from math import pi
import re
from scipy.spatial import ConvexHull, distance
from math import sqrt, pi, acos


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
        ts, dt, dalphadt = arttools.orientation.get_axis_movement_speed(attdata)
        m1 = dalphadt < 120.
        m1[np.isnan(dalphadt)] = True
        idx = arttools.mask.edges(m1)
        idx = idx[idx[:,1] - idx[:,0] > 1]
        gti = attdata['TIME'][idx]
        print(gti)

        urdnames = [os.path.join(l0path, session + name) for name in urdgroups[session]]
        attnames = [urdname[:-12] + "_gyro.fits" for urdname in urdnames]

        for i in range(gti.shape[0]):
            arttools.plot.make_mosaic_for_urdset_by_gti(urdnames, attnames, gti=gti[i:i+1])
            os.rename("tmpctmap.fits.gz", "%s_ctmap_%04d.fits.gz" % (session, i))
            os.rename("tmpemap.fits.gz", "%s_exmap_%04d.fits.gz" % (session, i))

