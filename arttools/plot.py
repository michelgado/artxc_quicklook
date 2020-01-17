import numpy as np
from astropy.wcs import WCS
from ._det_spatial import get_shadowed_pix_mask_for_urddata
from .time import get_gti, GTI, tGTI, emptyGTI, deadtime_correction
from .atthist import hist_orientation_for_attdata, make_wcs_for_attdata
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
from .lightcurve import get_overall_countrate
from .background import make_bkgmap_for_wcs
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
from functools import reduce

urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.029129860773177,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

class NoDATA(Exception):
    pass

def make_events_mask(minenergy=4., maxenergy=12., minflag=-1, ):
    def mask_events(urddata, grade, energy):
        eventsmask = np.all([grade > mingrade, grade < maxgrade,
                            urddata["RAW_X"] > minrawx, urddata["RAW_X"] < maxrawx,
                            urddata["RAW_Y"] > minrawy, urddata["RAW_Y"] < maxrawy,
                            energy > minenergy, energy < maxenergy], axis=0)
        return eventsmask
    return mask_events

standard_events_mask = make_events_mask(minenergy=4., maxenergy=12.)

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
    print("divide by", photsplitside)
    return img/photsplitside/photsplitside

def make_efg_mask(gmin=-1, gmax=0, fmin=-1, fmax=1, emin=4., emax=12.):
    def make_mask(urddata, energy, grade, flag):
        mask = np.all([grade > gmin, grade < gmax,
                            energy > emin, energy < emax,
                            flag > fmin, flag < fmax], axis=0)
        return mask
    return make_mask

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
                                  urdbti = {}):
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

    locwcs = make_wcs_for_attdata(attdata, gti, 20/3600.) #produce wcs for accumulated atitude information
    lside = np.argmin(locwcs.wcs.crpix)
    locwcs.wcs.crpix[lside] = locwcs.wcs.crpix[lside]//2
    locwcs.wcs.crpix[lside] = locwcs.wcs.crpix[lside] + 1 - locwcs.wcs.crpix[lside]%2

    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdhk = {}
    urdbkg = {}
    urdbkge = {}

    for urdfname in urdflist[:]:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]
        crate = (urdfile['HK'].data["EVENTS"][1:] - urdfile['HK'].data["EVENTS"][1:])/\
            (urdfile["HK"].data["TIME"][1:] - urdfile['HK'].data["TIME"][:-1])

        tchk = (urdfile["HK"].data["TIME"][1:] + urdfile['HK'].data["TIME"][:-1])/2.

        print("processing:", urdfname)
        #print("overall urd exposure", get_gti(urdfile, "STDGTI").exposure)
        locgti = (get_gti(urdfile, "STDGTI") if "STDGTI" in urdfile else get_gti(urdfile)) & gti & -urdgti.get(urdn, emptyGTI) # & -urdbti.get(urdn, emptyGTI)
        locgti.merge_joint()
        print("exposure in GTI:", locgti.exposure)
        locgti = locgti & -urdbti.get(urdn, emptyGTI)
        if np.any(crate[(locgti + [-30, 30]).mask_outofgti_times(tchk)] > 200.):
            print("skip due to bki")
            continue

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
        pickimg = np.all([energy > 4., energy < 11.2, grade > -1, grade < 10, flag == 0, locgti.mask_outofgti_times(urddata["TIME"])], axis=0)
        print(urdfname, "used events", pickimg.sum())
        if np.any(pickimg):
            timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, 1)
            print("total photon on img", timg.sum(), "selected events", pickimg.sum())
            imgdata += timg

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

    img = fits.PrimaryHDU(header=locwcs.to_header(), data=imgdata)
    img.writeto(outctsname, overwrite=True)

    urdhk = {urdn:np.unique(np.concatenate(hklist)) for urdn, hklist in urdhk.items()}
    urddtc = {urdn: deadtime_correction(hk) for urdn, hk in urdhk.items()}
    tgti = reduce(lambda a, b: a & b, urdgti.values())
    #print("tgti.exposure", tgti.exposure)
    #print(tgti.arr)
    te = np.concatenate([np.linspace(s, e, int((e-s)//100.) + 2) for s, e in tgti.arr])
    #print(te.size)
    mgaps = np.ones(te.size - 1, np.bool)
    if tgti.arr.size > 2:
        #print(np.cumsum([(int((e-s)//100.) + 2) for s, e in tgti.arr[:-1]]) - 1)
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
    #import pickle
    #pickle.dump([tc, rate,  urdgti, urddtc], open("checkexp.pickle", "wb"))

    """
    for urdn in urdbkg:
        tse, rate = zip(*urdbkg[urdn])
        tse = np.concatenate(tse)
        rate = np.concatenate(rate)
        idx = np.argsort(tse)
        tse = np.copy(tse[idx])
        rate = np.copy(rate[idx])
        urdbkg[urdn] = lambda x: np.ones(np.asarray(x).size, np.double) #interp1d(tse, rate, bounds_error=False, fill_value=np.median(rate))
    """

    emap = make_expmap_for_wcs(locwcs, attdata, urdgti, dtcorr=urddtc)
    emap = fits.PrimaryHDU(data=emap, header=locwcs.to_header())
    emap.writeto(outexpmapname, overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
    bmap = fits.PrimaryHDU(data=bmap, header=locwcs.to_header())
    bmap.writeto(outbkgname, overwrite=True)

if __name__ == "__main__":
    pass
