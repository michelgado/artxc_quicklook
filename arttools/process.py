from .plot import make_mosaic_for_urdset_by_gti, get_attdata, make_sky_image, make_energies_flags_and_grades
from .orientation import read_bokz_fits, AttDATA
from .planwcs import make_wcs_for_attdata, split_survey_mode
from .time import tGTI, get_gti, GTI, emptyGTI, deadtime_correction
from .expmap import make_expmap_for_wcs
from .background import make_bkgmap_for_wcs
from .telescope import URDNS
import subprocess
import os
from scipy.spatial import cKDTree
from math import pi, sqrt, log, log10, sin, cos
import numpy as np
from astropy.io import fits
from functools import reduce
from scipy.interpolate import interp1d
from functools import reduce

def poltovec(ra, dec):
    shape = np.asarray(ra).shape
    vec = np.empty(shape + (3,), np.double)
    vec[..., 0] = np.cos(dec*pi/180.)*np.cos(ra*pi/180.)
    vec[..., 1] = np.cos(dec*pi/180.)*np.sin(ra*pi/180.)
    vec[..., 2] = np.sin(dec*pi/180.)
    return vec

skycells = np.copy(fits.getdata("/srg/a1/work/srg/data/eRO_4700_SKYMAPS.fits", 1))
skycelltree = cKDTree(poltovec(skycells["RA_CEN"], skycells["DE_CEN"]))
SEARCHRAD = 2.*sin(4.9/2.*pi/180.)

urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.029129860773177,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

import pickle
bkigti = pickle.load(open("/srg/a1/work/andrey/ART-XC/gc/allbki2.pickle", "rb"))
allbki = reduce(lambda a, b: a | b, bkigti.values())


def get_neighbours(fpath):
    ids = fpath.split("/")[-1]
    ra, dec = float(ids[:3]), 90. - float(ids[3:])
    print(ra, dec)
    cells = skycelltree.query_ball_point(poltovec(ra, dec), SEARCHRAD)
    print(cells)
    print(list(zip(skycells["RA_CEN"][cells], skycells["DE_CEN"][cells])))
    cells = ["%03d%03d" % (np.ceil(skycells[c]["RA_CEN"]), np.ceil(90. - skycells[c]["DE_CEN"])) for c in cells]
    return cells


def analyze_survey(fpath, pastday=None):
    abspath = fpath #os.path.abspath(".")
    allfiles = os.listdir(os.path.join(abspath, "L0"))

    bokzfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "bokz.fits" in l]
    gyrofiles = [os.path.join(abspath, "L0", l) for l in allfiles if "gyro.fits" in l]
    urdfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "urd.fits" in l]

    date = bokzfiles[-1][-29:-21]


    if not pastday is None:
        allfiles = os.listdir(os.path.join(pastday, "L0"))
        gyrofiles += [os.path.join(pastday, "L0", l) for l in allfiles if "bokz.fits" in l]
        urdfiles += [os.path.join(pastday, "L0", l) for l in allfiles if "urd.fits" in l]

    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    gtis = split_survey_mode(attdata)

    for k, sgti in enumerate(gtis):
        #if (sgti & allbki).exposure == 0:
        if os.path.exists("bmap%02d_%s.fits.gz" % (k, date)):
            #print(sgti.exposure, allbki.exposure)
            #print(sgti.arr[[0, -1], [0, 1]], allbki.arr)
            continue
        print("start", k, "exposure", sgti.exposure)
        make_mosaic_for_urdset_by_gti(urdfiles, gyrofiles, sgti + [-30, 30],
                                      "cmap%02d_%s.fits.gz" % (k, date),
                                      "bmap%02d_%s.fits.gz" % (k, date),
                                      "emap%02d_%s.fits.gz" % (k, date),
                                      urdbti=bkigti,
                                      usedtcorr=False)


def run(fpath):
    os.chdir(fpath)
    abspath = os.path.abspath(".")
    neighbours = get_neighbours(abspath)
    print(neighbours)

    allfiles = os.listdir("L0")
    bokzfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "bokz.fits" in l]
    gyrofiles = [os.path.join(abspath, "L0", l) for l in allfiles if "gyro.fits" in l]
    urdfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "urd.fits" in l]

    attgti = reduce(lambda a, b: a | b, [get_gti(fits.open(urdfile)) for urdfile in urdfiles])
    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    locwcs = make_wcs_for_attdata(attdata, attgti)

    gyrofiles = []
    urdfiles = []
    print("current locations", os.path.abspath("."))
    for neighbour in neighbours:
        abspath = os.path.abspath(os.path.join("../", neighbour, "L0"))
        print("search in cell", abspath)
        if not os.path.exists(abspath):
            print("not filled yes, skip")
            continue
        allfiles = os.listdir(abspath)
        gyrofiles += [os.path.join(abspath, fname) for fname in allfiles if "gyro" in fname]
        urdfiles += [os.path.join(abspath, fname) for fname in allfiles if "urd" in fname]

    try:
        os.mkdir("L3")
    except OSError as exc:
        pass
    os.chdir("L3")

    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    print(attdata.gti)

    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdhk = {}
    urdbkg = {}
    urdbkge = {}

    for urdfname in urdfiles:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]
        print("processing:", urdfname)
        print("overall urd exposure", get_gti(urdfile).exposure)
        locgti = get_gti(urdfile) & attdata.gti & -urdgti.get(urdn, emptyGTI)
        locgti.merge_joint()
        print("exposure in GTI:", locgti.exposure)
        if locgti.exposure == 0.:
            continue
        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        urddata = urddata[locgti.mask_outofgti_times(urddata["TIME"])]

        hkdata = np.copy(urdfile["HK"].data)
        hkdata = hkdata[(locgti + [-30, 30]).mask_outofgti_times(hkdata["TIME"])]
        urdhk[urdn] = urdhk.get(urdn, []) + [hkdata,]

        energy, grade, flag = make_energies_flags_and_grades(urddata, hkdata, urdn)
        pickimg = np.all([energy > 4., energy < 11.2, grade > -1, grade < 10, flag == 0], axis=0)
        timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, 1)
        imgdata += timg

        pickbkg = np.all([energy > 40., energy < 100., grade > -1, grade < 10, flag < 3], axis=0)
        bkgevts = urddata["TIME"][pickbkg]
        urdbkge[urdn] = urdbkge.get(urdn, []) + [bkgevts,]

    for urdn in urdgti:
        print(urdn, urdgti[urdn].exposure)

    img = fits.PrimaryHDU(header=locwcs.to_header(), data=imgdata)
    img.writeto("cmap.fits.gz", overwrite=True)

    urdhk = {urdn:np.unique(np.concatenate(hklist)) for urdn, hklist in urdhk.items()}
    urddtc = {urdn: deadtime_correction(hk) for urdn, hk in urdhk.items()}
    tgti = reduce(lambda a, b: a & b, urdgti.values())
    te = np.concatenate([np.linspace(s, e, int((e-s)//100.) + 2) for s, e in tgti.arr])
    mgaps = np.ones(te.size - 1, np.bool)
    if tgti.arr.size > 2:
        mgaps[np.cumsum([(int((e-s)//100.) + 2) for s, e in tgti.arr[:-1]]) - 1] = False
        mgaps[te[1:] - te[:-1] < 10] = False

    tevts = np.sort(np.concatenate([np.concatenate(e) for e in urdbkge.values()]))
    rate = tevts.searchsorted(te)
    rate = (rate[1:] - rate[:-1])[mgaps]/(te[1:] - te[:-1])[mgaps]
    tc = (te[1:] + te[:-1])[mgaps]/2.
    tm = np.sum(tgti.mask_outofgti_times(tevts))/tgti.exposure

    urdbkg = {urdn: interp1d(tc, rate*urdbkgsc[urdn]/7.61, bounds_error=False, fill_value=tm*urdbkgsc[urdn]/7.62) for urdn in urdbkgsc}

    emap = make_expmap_for_wcs(locwcs, attdata, urdgti, dtcorr=urddtc)
    emap = fits.PrimaryHDU(data=emap, header=locwcs.to_header())
    emap.writeto("emap.fits.gz", overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
    bmap = fits.PrimaryHDU(data=bmap, header=locwcs.to_header())
    bmap.writeto("bmap.fits.gz", overwrite=True)


if __name__ == "__main__":
    run(sys.argv[1])

