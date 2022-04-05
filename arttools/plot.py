import numpy as np
from astropy.wcs import WCS
from ._det_spatial import get_shadowed_pix_mask_for_urddata, DL, F, multiply_photons
from .time import get_gti, GTI, tGTI, emptyGTI, deadtime_correction
from .atthist import hist_orientation_for_attdata
from .planwcs import make_wcs_for_attdata
from .caldb import get_energycal, get_shadowmask, get_energycal_by_urd, get_shadowmask_by_urd, urdbkgsc, OPAXOFFSET
from .energy import get_events_energy
from .telescope import URDNS
from .orientation import get_photons_sky_coord, read_gyro_fits, read_bokz_fits, AttDATA, define_required_correction, get_attdata, get_photons_vectors
from .vector import vec_to_pol
from .lightcurve import make_overall_lc, weigt_time_intervals
from .vignetting import load_raw_wignetting_function
from .filters import IndependentFilters, RationalSet, get_shadowmask_filter, InversedRationalSet
from .filters import Intervals
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
import pickle
import glob

import arttools

eband = namedtuple("eband", ["emin", "emax"])

imgfilters = {urdn: IndependentFilters({"ENERGY": Intervals([4., 12.]),
                                        "GRADE": RationalSet(range(10)),
                                        "RAWXY": get_shadowmask_filter(urdn)}) for urdn in URDNS}

bkgfilters = IndependentFilters({"ENERGY": Intervals([40., 100.]),
                                 "GRADE": RationalSet(range(10)),
                                 "RAW_X": arttools.filters.InversedRationalSet([0, 47]),
                                 "RAW_Y": arttools.filters.InversedRationalSet([0, 47])})

class NoDATA(Exception):
    pass


def constscale(const, func):
    def newfunc(val):
        return func(val)*const
    return newfunc


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


def make_vignetting_weighted_phot_images(urddata, urdn, energy, attdata, locwcs, photsplitside=1):
    rawvignfun = load_raw_wignetting_function()
    x, y = multiply_photons(urddata, photsplitside)
    weights = rawvignfun(np.array([np.repeat(energy, photsplitside*photsplitside), x, y]).T)
    r, d = get_photons_sky_coord(urddata, urdn, attdata, photsplitside)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                            np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5],
                            weights=weights)[0].T
    return img


def make_sky_image(urddata, urdn, attdata, locwcs, photsplitside=10, weight_with_vignetting=False):
    r, d = get_photons_sky_coord(urddata, urdn, attdata, photsplitside)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                            np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    return img/photsplitside/photsplitside


def make_skyimage_for_urdset_by_gti(urdflist, attflist, outctsname, gti=tGTI, photsplitnside=1,
                                   pixsize=10/3600., usedtcorr=True, weightphotons=False,
                                   locwcs=None,
                                   **kwargs):
    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(attflist)])
    attdata = attdata.apply_gti(gti + [-30, 30])
    gti = attdata.gti & gti

    imgfilter = arttools.filters.IndependentFilters(\
                    {"ENERGY": arttools.interval.Intervals([4., 12.]),
                     "GRADE": arttools.filters.RationalSet(range(10))})


    urdgti = {URDN:emptyGTI for URDN in URDNS}
    vecs = []
    ra, dec = [], []

    for urdfname in urdflist[:]:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]
        print("processing:", urdfname)

        locgti = (get_gti(urdfile, "STDGTI") if "STDGTI" in urdfile else get_gti(urdfile)) & gti & ~urdgti.get(urdn, emptyGTI) # & -urdbti.get(urdn, emptyGTI)
        locgti.merge_joint()
        if locgti.exposure == 0.:
            continue
        imgfilter["RAWXY"] = arttools.filters.get_shadowmask_filter(urdn)
        imgfilter["TIME"] = locgti

        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        if not ("ENERGY" in urddata.dtype.names and "GRADE" in urddata.dtype.names):
            urddata = arttools.energy.add_energies_and_grades(urddata, urdfile["HK"].data,
                                                    arttools.caldb.get_energycal_by_urd(urdn))
        if not ("RA" in urddata.dtype.names and "DEC" in urddata.dtype.names):
            urddata = arttools.orientation.add_ra_dec(urddata, urdn, attdata)

        urddata = urddata[imgfilter.apply(urddata)]
        ra.append(urddata["RA"])
        dec.append(urddata["DEC"])

    ra, dec = np.concatenate(ra), np.concatenate(dec)
    tgti = reduce(lambda a, b: a |b, urdgti.values())
    locwcs = make_wcs_for_attdata(attdata, tgti, pixsize)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    img = np.zeros((ysize, xsize), np.double)
    y, x = (locwcs.all_world2pix(np.array([ra, dec]).T, 1) - 0.5).astype(np.int).T
    u, uc = np.unique(np.array([x, y]), axis=1, return_counts=True)
    img[u[0], u[1]] = uc
    fits.PrimaryHDU(data=img, header=locwcs.to_header()).writeto(outctsname)


def make_mosaic_for_urdset_by_gti(urdflist, attflist, gti,
                                  outctsname, outbkgname, outexpmapname,
                                  urdbti={}, ebands={"soft": eband(4, 12), "hard": eband(8, 16)},
                                  photsplitnside=1,
                                  pixsize=20/3600., usedtcorr=True, weightphotons=False,
                                  locwcs=None,
                                  **kwargs):
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

    if locwcs is None: locwcs = make_wcs_for_attdata(attdata, gti, pixsize) #produce wcs for accumulated atitude information
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = {name: np.zeros((ysize, xsize), np.double) for name in ebands}
    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdhk = {}
    urdbkg = {}
    urdbkge = []
    bkggti = {}

    for urdfname in urdflist[:]:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]

        tchk = (urdfile["HK"].data["TIME"][1:] + urdfile['HK'].data["TIME"][:-1])/2.

        print("processing:", urdfname)
        locgti = (get_gti(urdfile, "STDGTI") if "STDGTI" in urdfile else get_gti(urdfile)) & gti & ~urdgti.get(urdn, emptyGTI) # & -urdbti.get(urdn, emptyGTI)
        locgti.merge_joint()
        print("exposure in GTI:", locgti.exposure)
        locgti = locgti & ~urdbti.get(urdn, emptyGTI)
        print("exposure after excluding BTI", locgti.exposure)
        if locgti.exposure == 0.:
            continue
        print("Tstart, Tstop:", locgti.arr[[0, -1], [0, 1]])
        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        urddata = urddata[(locgti + [-200, 200]).mask_external(urddata["TIME"])]

        hkdata = np.copy(urdfile["HK"].data)
        hkdata = hkdata[(locgti + [-30, 30]).mask_external(hkdata["TIME"])]
        urdhk[urdn] = urdhk.get(urdn, []) + [hkdata,]

        energy, grade, flag = make_energies_flags_and_grades(urddata, hkdata, urdn)
        timemask = locgti.mask_external(urddata["TIME"])
        for bandname, band in ebands.items():
            pickimg = np.all([energy > band.emin, energy < band.emax, grade > -1, grade < 10,
                              flag == 0, locgti.mask_external(urddata["TIME"])], axis=0)
            if np.any(pickimg):
                if weightphotons:
                    timg = make_vignetting_weighted_phot_images(urddata[pickimg], urdn, energy[pickimg], attdata, locwcs, photsplitnside)
                else:
                    timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, photsplitnside)
                print("total photon on img", timg.sum(), "selected events", pickimg.sum())
                imgdata[bandname] += timg

        bkgevts = urddata["TIME"][bkgfilters.apply(urddata)]
        urdbkge.append(bkgevts)

    for bandname, img in imgdata.items():
        img = fits.PrimaryHDU(header=locwcs.to_header(), data=img)
        img.writeto(bandname + outctsname, overwrite=True)

    urdhk = {urdn:np.unique(np.concatenate(hklist)) for urdn, hklist in urdhk.items()}
    urddtc = {urdn: deadtime_correction(hk) for urdn, hk in urdhk.items()}
    urdbkg = arttools.background.get_background_lightcurve(np.sort(np.concatenate(urdbkge)), urdgti, bkgfilters, 1000., imgfilters)

    if usedtcorr:
        emap = make_expmap_for_wcs(locwcs, attdata, urdgti, imgfilters, dtcorr=urddtc, **kwargs)
    else:
        emap = make_expmap_for_wcs(locwcs, attdata, urdgti, imgfilters, **kwargs)
    emap = fits.PrimaryHDU(data=emap, header=locwcs.to_header())
    emap.writeto(outexpmapname, overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, urdbkg, imgfilters)
    bmap = fits.PrimaryHDU(data=bmap, header=locwcs.to_header())
    bmap.writeto(outbkgname, overwrite=True)

if __name__ == "__main__":
    gyrofiles, urdfiles, outimgname = sys.argv[1:2]
    gyrofiles = glob.glob(gyrofiles)
    urdfiles = glob.glob(urdfiles)
    make_skyimage_for_urdset_by_gti(gyrofiles, urdfiles, outimgname)
    #pass, r, d - quasi cartesian coordinates of the vecteces
    #it should be noted that convex hull is expected to be alongated along equator after quaternion rotation
