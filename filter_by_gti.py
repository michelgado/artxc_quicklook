#!/opt/soft/psoft/bin/python3.7

import numpy as np

import sys
from arttools.time import get_gti, get_filtered_table, gti_intersection, gti_union
from astropy.io import fits

if __name__ == "__main__":
    try:
        inname, outname, gtifile = sys.argv[1:]
    except Exception:
        print("u're doing something wrong dude\nsignatire: filter_by_gti.py inputfilename outputfilename gtifilename")
    print("input file:", inname)
    print("output file:", outname)
    print("gti file", gtifile)

    fitsfile = fits.open(inname)
    gti = get_gti(fitsfile)
    newgti = gti_intersection(np.loadtxt(gtifile).T.reshape((-1, 2)), gti)
    fitsfile["EVENTS"].data = get_filtered_table(fitsfile["EVENTS"].data, newgti)
    fitsfile["EVENTS"].data.columns["TIME_I"].array = fitsfile["EVENTS"].data["TIME_I"]
    fitsfile["GTI"].data = np.array([tuple(g) for g in newgti], dtype=fitsfile["GTI"].data.dtype)
    fitsfile["HK"].data = get_filtered_table(fitsfile["HK"].data, gti_union(newgti + [-15, +15]))
    fitsfile.writeto(outname)

