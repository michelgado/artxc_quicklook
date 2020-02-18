import numpy as np
from astropy.wcs import WCS
from ._det_spatial import get_shadowed_pix_mask_for_urddata
from .time import get_gti, GTI, tGTI, emptyGTI, deadtime_correction
from .atthist import hist_orientation_for_attdata
from .planwcs import make_wcs_for_attdata
from .caldb import get_energycal, get_shadowmask, get_energycal_by_urd, get_shadowmask_by_urd
from .energy import get_events_energy
from .telescope import URDNS
from .orientation import get_photons_sky_coord, read_gyro_fits, read_bokz_fits, AttDATA, define_required_correction
from astropy.io import fits
from math import pi, cos, sin
from multiprocessing import Pool, cpu_count, Queue, Process, Pipe
from threading import Thread
import copy
import time
import matplotlib.pyplot as plt
import os, sys
from .expmap import make_expmap_for_wcs
from .background import make_bkgmap_for_wcs
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from functools import reduce
from collections import  namedtuple

eband = namedtuple("eband", ["emin", "emax"])

urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.029129860773177,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

class NoDATA(Exception):
    pass

def make_energies_flags_and_grades(urddata, urdhk, urdn):
    flag = np.zeros(urddata.size, np.uint8)
    shadow = get_shadowmask_by_urd(urdn)
    caldbfile = get_energycal_by_urd(urdn)
    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    flag[np.logical_not(maskshadow)] = 2
    flag[np.any([urddata["RAW_X"] == 0, urddata["RAW_X"] == 47, \
                 urddata["RAW_Y"] == 0, urddata["RAW_Y"] == 47], axis=0)] = 3
    energy, xc, yc, grade = get_events_energy(urddata, urdhk, caldbfile)
    return energy, grade, flag

def make_sky_image(urddata, urdn, attdata, locwcs, photsplitside=10):
    r, d = get_photons_sky_coord(urddata, urdn, attdata, photsplitside)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    return img/photsplitside/photsplitside

def get_attdata(fname):
    ffile = fits.open(fname)
    attdata = read_gyro_fits(ffile["ORIENTATION"]) if "gyro" in fname else read_bokz_fits(ffile["ORIENTATION"])
    attdata.times = attdata.times - (0.97 if "gyro" in fname else 1.55)
    attdata.gti.arr = attdata.gti.arr - (0.97 if "gyro" in fname else 1.55)
    if "gyro" in fname:
        attdata = define_required_correction(attdata)
    return attdata


def make_mosaic_for_urdset_by_gti(urdflist, attflist, gti,
                                  outctsname, outbkgname, outexpmapname,
                                  urdbti = {}, ebands = {"soft": eband(4, 12), "hard": eband(8, 16)},
                                  usedtcorr=True, pixsize=20./3600.):
    """
    given two sets with paths to the urdfiles and corresponding attfiles,
    and gti as a dictionary, each key contains gti for particular urd
    the program produces overall count map and exposition map for this urdfiles set
    the wcs is produced automatically to cover nonzero exposition area with some margin
    """
    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(attflist)])
    #attdata usually has data points stored each 3 seconds so try here to obtaind attitude information for slightly longer time span
    attdata = attdata.apply_gti(gti + [-30, 30])
    gti = attdata.gti & gti

    locwcs = make_wcs_for_attdata(attdata, gti, 15/3600.) #produce wcs for accumulated atitude information
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = {name: np.zeros((ysize, xsize), np.double) for name in ebands}
    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdhk = {}
    urdbkg = {}
    urdbkge = {}

    for urdfname in urdflist[:]:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]

        tchk = (urdfile["HK"].data["TIME"][1:] + urdfile['HK'].data["TIME"][:-1])/2.

        print("processing:", urdfname)
        locgti = (get_gti(urdfile, "STDGTI") if "STDGTI" in urdfile else get_gti(urdfile)) & gti & -urdgti.get(urdn, emptyGTI) # & -urdbti.get(urdn, emptyGTI)
        locgti.merge_joint()
        print("exposure in GTI:", locgti.exposure)
        locgti = locgti & -urdbti.get(urdn, emptyGTI)
        print("exposure after excluding BTI", locgti.exposure)
        if locgti.exposure == 0.:
            continue
        print("Tstart, Tstop:", locgti.arr[[0, -1], [0, 1]])
        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        urddata = urddata[(locgti + [-1000, 1000]).mask_outofgti_times(urddata["TIME"])]

        hkdata = np.copy(urdfile["HK"].data)
        hkdata = hkdata[(locgti + [-30, 30]).mask_outofgti_times(hkdata["TIME"])]
        urdhk[urdn] = urdhk.get(urdn, []) + [hkdata,]

        energy, grade, flag = make_energies_flags_and_grades(urddata, hkdata, urdn)
        for bandname, band in ebands.items():
            pickimg = np.all([energy > band.emin, energy < band.emax, grade > -1, grade < 10, flag == 0, locgti.mask_outofgti_times(urddata["TIME"])], axis=0)
            print(urdfname, "used events", pickimg.sum())
            if np.any(pickimg):
                timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, 1)
                print("total photon on img", timg.sum(), "selected events", pickimg.sum())
                imgdata[bandname] += timg

        pickbkg = np.all([energy > 40., energy < 100., grade > -1, grade < 10, flag < 3], axis=0)
        bkgevts = urddata["TIME"][pickbkg]
        """
        tsr = np.arange(locgti.arr[0, 0], locgti.arr[-1, 1] + 150., 200.)
        te, mgaps = locgti.make_tedges(tsr)
        dt = (te[1:] - te[:-1])[mgaps]
        mgaps[mgaps] = dt > 50.
        tse = (te[1:] + te[:-1])[mgaps]/2.
        cs = bkgevts.searchsorted(te)
        cr = (cs[1:] - cs[:-1])[mgaps]/(te[1:] - te[:-1])[mgaps]
        urdbkg[urdn] = urdbkg.get(urdn, []) + [(tse, cr)]
        """
        urdbkge[urdn] = urdbkge.get(urdn, []) + [bkgevts,]

    for urdn in urdgti:
        print("real time exposure", urdn, urdgti[urdn].exposure)

    for bandname, img in imgdata.items():
        img = fits.PrimaryHDU(header=locwcs.to_header(), data=img)
        img.writeto(bandname + outctsname, overwrite=True)

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


    if tc.size == 0:
        urdbkg = {urdn: lambda x: np.ones(x.size)*tm*urdbkgsc[urdn]/7.62 for urdn in urdbkgsc}
    else:
        urdbkg = {urdn: interp1d(tc, rate*urdbkgsc[urdn]/7.61, bounds_error=False, fill_value=tm*urdbkgsc[urdn]/7.62) for urdn in urdbkgsc}

    if usedtcorr:
        emap = make_expmap_for_wcs(locwcs, attdata, urdgti, dtcorr=urddtc)
    else:
        emap = make_expmap_for_wcs(locwcs, attdata, urdgti)
    emap = fits.PrimaryHDU(data=emap, header=locwcs.to_header())
    emap.writeto(outexpmapname, overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
    bmap = fits.PrimaryHDU(data=bmap, header=locwcs.to_header())
    bmap.writeto(outbkgname, overwrite=True)

if __name__ == "__main__":
    pass
    #pass, r, d - quasi cartesian coordinates of the vecteces
    #it should be noted that convex hull is expected to be alongated along equator after quaternion rotation
