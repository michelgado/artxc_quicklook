from .orientation import ART_det_QUAT
from .atthist import hist_orientation_for_attdata, AttWCShist
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting
from .time import gti_intersection, gti_difference
from functools import reduce



def make_expmap_for_wcs(wcs, attdata, gti, mpnum=MPNUM):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    overall_gti = reduce(gti_intersection, gti.values())
    exptime, qval = hist_orientation_for_attdata(attdata, overall_gti)
    vmap = make_overall_vignetting()
    print("produce overall urds expmap")
    emap = AttWCShist.make_mp_expmap(wcs, vmap, exptime, qval, mpnum)
    print("\ndone!")
    for urd in gti:
        urdgti = gti_difference(overall_gti, gti[urd])
        if urdgti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval = hist_orientation_for_attdata(attdata, urdgti, ART_det_QUAT[urd])
        vmap = make_vignetting_for_urdn(urd)
        emap = AttWCShist.make_mp_expmap(wcs, vmap, exptime, qval, mpnum) + emap
        print(" done!")
    return emap
