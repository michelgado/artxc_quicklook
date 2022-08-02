from .mosaic2 import  HealpixSky
from .orientation import FullSphereChullGTI, ChullGTI
from .telescope import ANYTHINGTOURD


def make_gti_index(flist):
    sphere = FullSphereChullGTI()
    sphere.split_on_equal_segments(5.)
    for fname in flist:
        attdata = get_attdata(flist)
        slews = get_slews_gti(attloc)
        for i, arr in enumerate((~slews & attdata.gti).arr):
            gloc = GTI(att)
            attloc = attdata.apply_gti(gloc)
            for ch, g in attloc.get_covering_chulls():
                for sch in sphere.get_all_child_intersect(ch.expand(pi/180.*26.5/60.)):
                    sch.update_gti_for_attdata(attloc)
    return sphere

def make_attdata_gti_index(flist):
    gtot = emptyGTI
    names = []
    times = []
    for fname in flist:
        attdata = get_attdata(fname)
        slews = get_slews_gti(attloc)
        mygti = attdata.gti & ~slews & ~gtot
        gtot = gtot | mygti
        idx = np.searchsorted(times, mygti.arr[:, 0])
        for i in idx[::-1]:
            names.insert(idx, fname)
            times = np.sort(np.concatenate([times, mygti.arr[:, 0]]))
    return times, names

def make_urd_gti_index(flist):
    gtot = emptyGTI
    names = []
    times = []
    for fname in flist:
        mygti = get_gti(fits.open(fname)) & ~slews & ~gtot
        gtot = gtot | mygti
        idx = np.searchsorted(times, mygti.arr[:, 0])
        for i in idx[::-1]:
            names.insert(idx, fname)
            times = np.sort(np.concatenate([times, mygti.arr[:, 0]]))
    return times, names


if __name__ == "__main__":
    import os
    allfiles = [os.path.join("data", n) for n in os.listdir("data")]
    gtisphereindex = make_gti_index([fname for fname in allfiles if fname[-8:] == "gyro.fits"])
    attindex = make_attdata_gti_index([fname for fname in allfiles if fname[-8:] == "gyro.fits"])
    urdindex = {}
    for urdn in ["02", "04", "08", "10", "20", "40", "80"]:
        urdinex[ANYTHINGTOURD[urdn]] = make_urd_gti_index([fname for fname in allfiles if ("%s_urd.fits" % urdn) == fname[-10:]])
    import pickle
    pickle.dump([gtisphereindex, attindex, urdindex], open("/home/andrey/ART-XC/data/sphere_index.pkl", "wb"))






