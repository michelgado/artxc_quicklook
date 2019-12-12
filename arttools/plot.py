import numpy as np
from astropy.wcs import WCS
from ._det_spatial import get_shadowed_pix_mask_for_urddata
from .time import get_gti, GTI, tGTI, emptyGTI
from .atthist import hist_orientation_for_attdata, make_wcs_for_attdata
from .caldb import get_energycal, get_shadowmask
from .energy import get_events_energy
from .telescope import URDNS
from .orientation import get_photons_sky_coord
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


def make_energies_flags_and_grades(urdfile):
    urddata = np.copy(urdfile["EVENTS"].data)
    flag = np.zeros(urddata.size, np.uint8)
    URDN = urdfile["EVENTS"].header["URDN"]
    caldbfile = get_energycal(urdfile)
    shadow = get_shadowmask(urdfile)
    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    flag[np.logical_not(maskshadow)] = 2
    flag[np.any([urddata["RAW_X"] == 0, urddata["RAW_X"] == 47, \
                 urddata["RAW_Y"] == 0, urddata["RAW_Y"] == 47], axis=0)] = 3
    energy, xc, yc, grade = get_events_energy(urddata, np.copy(urdfile["HK"].data), caldbfile)
    return urddata, energy, grade, flag

def make_sky_image(urddata, urdn, attdata, locwcs):
    r, d = get_photons_sky_coord(urddata, urdn, attdata, 10)
    x, y = locwcs.all_world2pix(np.array([r*180./pi, d*180./pi]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(locwcs.wcs.crpix[0]*2 + 2) + 0.5,
                                np.arange(locwcs.wcs.crpix[1]*2 + 2) + 0.5])[0].T
    return img/100.


def make_efg_mask(gmin=-1, gmax=0, fmin=-1, fmax=1, emin=4., emax=12.):
    def make_mask(urddata, energy, grade, flag):
        mask = np.all([grade > gmin, grade < gmax,
                            energy > emin, energy < emax,
                            flag > fmin, flag < fmax], axis=0)
        return mask
    return make_mask

def make_mosaic_for_urdset_by_gti(urdflist, attflist, gti):
    """
    given two sets with paths to the urdfiles and corresponding attfiles,
    and gti as a dictionary, each key contains gti for particular urd
    the program produces overall count map and exposition map for this urdfiles set
    the wcs is produced automatically to cover nonzero exposition area with some margin
    """
    attdata = sum(read_gyro_fits(fits.open(fname)["ORIENTATION"]) for fname in set(attflist))
    attdata.apply_gti(gti + [-2, 2])
    gti = attdata.gti - [-2, 2]

    locwcs = make_wcs_for_attdata(attdata, gti)
    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    urdgtis = {URDN:emptyGTI for URDN in URDNS}
    tetot = []
    mgapstot = []

    for urdfname in urdflist[:]:
        try:
            urdfile = fits.open(urdfname)
            urdn = urdfile["EVENTS"].header["URDN"]
            urdgti = get_gti(urdfile)
            urdgti = (urdgti & -urdgtis[urdn]) & attdata.gti
            urddata, energy, grade, flag = make_energies_flags_and_grades(urdfile)
            stdmask = make_efg_mask()
            gtimask = urdgti.mask_outofgti_times(urddata["TIME"])
            imgdata = urddata[stdmask(urddata, energy, grade, flag) & gtimask]
            timg = make_sky_image(urddata, urdn, attdata, locwcs)
            """
            bkgmask = make_efg_mask(emin=40., emax=100., fmax=3)
            bkgts = urddata["TIME"][bkgmask & gtimask]
            idx = bkgts.searchsorted(urdgti.arr)
            cloc = (idx[:, 1] - idx[:, 0])//1000 + 2
            te = np.concatenate([np.linspace(gti[i, 0], gti[i, 1], cloc[i]) for i in range(cloc.size)])
            """
            urdgtis[urdn] = urdgtis[urdn] & urdgti
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            lhdu.writeto("tmpctmap.fits.gz", overwrite=True)
            urdgti[urdn] = np.concatenate([urdgti.get(urdn, np.empty((0, 2), np.double)), locgti])

    emap = make_expmap_for_wcs(locwcs, attall, urdgti)
    emap = fits.ImageHDU(data=emap, header=locwcs.to_header())
    h1 = fits.PrimaryHDU(header=locwcs.to_header())
    ehdu = fits.HDUList([h1, emap])
    ehdu.writeto("tmpemap.fits.gz", overwrite=True)


def make_expmap_for_urd(urdfile, attfile, locwcs, agti=None):
    """
    given the urdfile, attfile, wcs and gti produces exposition map in the wcs coordinates.
    gti is generated from as an intesection of agti and urdfile gti
    it is assumed implicitly, that locwcs crpix is a precise center of the produced exposition map
    therefore the size of produced image is locwcs.wcs.crpix*2 + 1 (assuming crpix is odd)
    """
    gti = np.array([urdfile["GTI"].data["START"], urdfile["GTI"].data["STOP"]]).T
    gtiatt = np.array([attfile["ORIENTATION"].data["TIME"][[0, -1]]])
    if not agti is None: gtiatt = gti_intersection(gtiatt, agti)
    gti = gti_intersection(gtiatt, gti)
    exptime, qval, gti = hist_orientation_for_attdata(attfile["ORIENTATION"].data, gti)
    qval = qval*ART_det_QUAT[urdfile["EVENTS"].header["URDN"]]
    """
    to do: implement vignmap in caldb
    """
    vignfilename = "/srg/a1/work/andrey/art-xc_vignea.fits"
    xsize = int(locwcs.wcs.crpix[0]*2 - 1)
    ysize = int(locwcs.wcs.crpix[1]*2 - 1)

    pool = Pool(24)
    emap = sum(pool.imap_unordered(make_vignmap_mp,
                [(locwcs, xsize, ysize, qval[i::50], exptime[i::50], vignfilename) for i in range(50)]))
    return emap


if __name__ == "__main__":
    pass
