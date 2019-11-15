from .caldb import ARTQUATS
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from functools import reduce
from multiprocessing import cpu_count
import numpy as np

MPNUM = cpu_count()


def make_expmap_for_wcs(wcs, attdata, urdgtis, mpnum=MPNUM, dtcorr={}):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    if dtcorr:
        overall_gti = emptyGTI
        emap = 0
    else:
        overall_gti = reduce(lambda a, b: a & b, urdgtis.values())
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti)
        vmap = make_overall_vignetting()
        print("produce overall urds expmap")
        emap = AttWCSHist.make_mp(vmap, exptime, qval, wcs, mpnum)
        print("\ndone!")

    for urd in urdgtis:
        gti = urdgtis[urd] & -overall_gti
        if urdgti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*ARTQUATS[urd], gti, \
                                                             dtcorr.get(urd, lambda x: 1))
        vmap = make_vignetting_for_urdn(urd)
        emap = AttWCSHist.make_mp(vmap, exptime, qval, wcs,  mpnum) + emap
        print(" done!")
    return emap


def make_expmap_for_healpix(attdata, urdgtis, mpnum=MPNUM, dtcorr={}):
    if dtcorr:
        overall_gti = emptyGTI
        emap = 0.
    else:
        overall_gti = reduce(lambda a, b: a & b, urdgtis.values())
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti)
        vmap = make_overall_vignetting()
        print("produce overall urds expmap")
        emap = AttHealpixhist.make_mp(2048, vmap, exptime, qval, mpnum)
        print("\ndone!")

    for urd in urdgtis:
        gti = urdgtis[urd] & -overall_gti
        if gti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*ARTQUATS[urd], gti)
        vmap = make_vignetting_for_urdn(urd)
        emap = AttHealpixHist.make_mp(2048, vmap, exptime, qval, mpnum) + emap
        print(" done!")
    return emap


