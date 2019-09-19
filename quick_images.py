import numpy as np
import arttools
from astropy.io import fits
import sys, os
from math import pi
import re

print("arttools", arttools)


if __name__ == "__main__":
    l0path = sys.argv[1]
    urdfnames = [l.rstrip() for l in os.listdir(l0path) if "urd.fits" in l]
    urdgroups = {}
    for name in urdfnames:
        urdgroups[name[:-12]] = urdgroups.get(name[:-12], []) + [name[-12:], ]

    for prefix in urdgroups:
        attname = os.path.join(l0path, prefix + "_gyro.fits")
        attdata = np.copy(fits.getdata(attname, "ORIENTATION"))
        ts, dt, dalphadt = arttools.time.get_axis_movement_speed(attdata)
        m1 = dalphadt < 50.
        m1[np.isnan(dalphadt)] = True
        idx = arttools.mask.edges(m1)
        idx = idx[idx[:,1] - idx[:,0] > 1]
        gti = attdata['TIME'][idx]
        urdnames = [os.path.join(l0path, prefix + name) for name in urdgroups[prefix]]
        attnames = [attname, ]*len(urdnames)

        for i in range(gti.shape[0]):
            print(gti[i:i+1])
            lgti = {urd: gti[i: i+1,:] for urd in arttools.telescope.URDNS}
            arttools.plot.make_mosaic_for_urdset_by_gti(urdnames, attnames, gti=lgti)
            os.rename("tmpctmap.fits.gz", "%s_ctmap_%04d.fits.gz" % (prefix, i))
            os.rename("tmpemap.fits.gz", "%s_exmap_%04d.fits.gz" % (prefix, i))

