import arttools
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.wcs import WCS
from arttools._det_spatial import get_shadowed_pix_mask_for_urddata
from astropy.coordinates import SkyCoord
from math import pi
import time
import pdb
import sigproc


class NoDATA(Exception):
    pass


ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"

URDTOTEL = {28: "T1",
            22: "T2",
            23: "T3",
            24: "T4",
            25: "T5",
            26: "T6",
            30: "T7"}

locwcs = WCS(naxis=2)
locwcs.wcs.crpix = [511.5, 511.5]
locwcs.wcs.crval = [265.4, -34.6]
locwcs.wcs.cdelt = [20./3600., 20./3600]
locwcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

sc = SkyCoord(265.4, -34.6, unit=("deg", "deg"), frame="icrs")

def get_caldb(caldb_entry_type, telescope, CALDB_path=ARTCALDBPATH, indexfile=indexfname):
    print(caldb_entry_type, telescope)

    #Try to open file
    indexfile_path = os.path.join(CALDB_path, indexfile)
    try:
        caldbindx   = fits.open(indexfile_path)
        caldbdata   = caldbindx[1].data
        for entry in caldbdata:
            if entry['CAL_CNAME'] == caldb_entry_type and entry['INSTRUME']==telescope:
                return_path = os.path.join(CALDB_path, entry['CAL_DIR'], entry['CAL_FILE'])
                print(return_path)
                return return_path
        return None

    except:
        print ('No index file here:' + indexfile_path)
        return None

def get_image(attname, urdname, locwcs, reduced):
    if not os.path.exists(attname):
        raise NoDATA("no attitude file")

    attfile = fits.open(attname)
    attdata = np.copy(attfile[1].data)
    mask = attdata["QORT_0"]**2. + attdata["QORT_1"]**2. + attdata["QORT_2"]**2. + attdata["QORT_3"]**2. > 0.
    attdata = attdata[mask]

    ra, dec, roll = arttools.orientation.extract_raw_gyro(attdata)
    if not np.any(sc.separation(SkyCoord(ra*180./pi, dec*180./pi, unit=("deg", "deg"), frame="icrs")).value < 5.):
        raise NoData("far far away")

    urdfile = fits.open(urdname)
    if urdfile["HK"].data.size < 5:
        raise NoDATA("not enough HK data")

    urddata = np.copy(urdfile["EVENTS"].data)
    URDN = urdfile["EVENTS"].header["URDN"]
    caldbfile = fits.open(get_caldb("TCOEF", URDTOTEL[urdfile["EVENTS"].header["URDN"]]))

    masktime = (urddata["TIME"] > attdata["TIME"][0]) & (urddata["TIME"] < attdata["TIME"][-1])
    mask = np.copy(masktime)
    urddata = urddata[masktime]

    shadow = np.logical_not(fits.getdata(get_caldb("OOFPIX", URDTOTEL[urdfile[1].header["URDN"]])))
    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    urddata = urddata[maskshadow]
    mask[mask] = maskshadow


    maskenergy, ENERGY, xc, yc = arttools.energy.get_events_energy(urddata, np.copy(urdfile["HK"].data), caldbfile)
    urddata = urddata[maskenergy]
    mask[mask] = maskenergy

    maskenergy2 = (ENERGY > 5.) & (ENERGY < 16.)
    if not np.any(maskenergy2):
        raise NoDATA("empty event list, after e filter")
    urddata = urddata[maskenergy2]
    mask[mask] = maskenergy2
    print(mask.sum(), urddata.size)
    ts = np.arange(urddata["TIME"][0], urddata["TIME"][-1], 10)
    cs = np.searchsorted(urddata["TIME"], ts)
    masknew = (cs[1:] - cs[:-1]) < 1000.
    gti = np.array([ts[:-1][masknew], ts[1:][masknew] + 1e-2]).T
    gti = sigproc.merge_gti_intersections(gti)
    idx = np.searchsorted(urddata["TIME"], gti)
    masknew = np.zeros(urddata.size, np.bool)
    print(masknew.size)
    for s, e in idx:
        masknew[s:e] = True
    urddata = urddata[masknew]
    mask[mask] = masknew


    print("before filtering")
    print(urddata.size)

    r, d = arttools.orientation.get_photons_sky_coord(urddata, URDN, attdata, subscale=1)
    maskorientation = np.all([r > locwcs.wcs.crval[0] - 3., r < locwcs.wcs.crval[0] + 3,
                             d > locwcs.wcs.crval[1] - 3., d < locwcs.wcs.crval[1] + 3.], axis=0)

    if not np.any(maskorientation):
        raise NoDATA("empty event list, after o filter")
    mask[mask] = maskorientation
    urddata = urddata[maskorientation]
    if r.size > 0:
        reduced.append(urdname)


    r, d = arttools.orientation.get_photons_sky_coord(urddata, urdfile[1].header["URDN"], attdata, 1)
    print("\n\n\nfinal size after filtering!!!! ", r.size)
    x, y = locwcs.all_world2pix(np.array([r, d]).T, 1.).T
    img = np.histogram2d(x, y, [np.arange(1025),]*2)[0].T

    r, d = arttools.orientation.vec_to_pol(arttools.orientation.get_gyro_quat(attdata).apply([1,0,0]))
    masktor = np.all([r > locwcs.wcs.crval[0] - 3.5, r < locwcs.wcs.crval[0] + 3.5,
                        d > locwcs.wcs.crval[1] - 3.5, d < locwcs.wcs.crval[1] + 3.5], axis=0)
    gtior = np.array([attdata["TIME"][masktor][[0, -1]]])
    expmap = arttools.plot.make_expmap_for_urd(urdfile, attfile, locwcs, gtior)

    return img, expmap



def run_in_mp(*args):
    try:
        timg, exp = get_image(attname, urdname, locwcs, reduced)
    except NoDATA as nd:
        print(nd)
        return None, None
    else:
        return timg, exp


if __name__ == "__main__":
    ra, dec = [], []
    dnames = sorted([dname for dname in os.listdir(".") if os.path.isdir(dname) and "srg" in dname])
    print(dnames)
    reduced = []
    imgdata = np.zeros((1024, 1024), np.double)
    expdata = np.zeros((1024, 1024), np.double)
    ctr = 0

    dirname = "/srg/a1/work/hard/ART_data/GCSURV"

    urdnames = [os.path.join(dirname, name) for name in os.listdir(dirname) if "cl.fits" in name]
    attnames = [os.path.join(dirname, "srg_20190903_201629_001_gyro.fits"),]*len(urdnames)

    for attname, urdname in zip(attnames, urdnames):
        try:
            timg, exp = get_image(attname, urdname, locwcs, reduced)
        except NoDATA as nd:
            print(nd)
        else:
            imgdata += timg
            expdata += exp
            img = fits.ImageHDU(data=imgdata, header=locwcs.to_header())
            exp = fits.ImageHDU(data=expdata, header=locwcs.to_header())
            h1 = fits.PrimaryHDU(header=locwcs.to_header())
            lhdu = fits.HDUList([h1, img])
            ehdu = fits.HDUList([h1, exp])
            lhdu.writeto("gs.fits", overwrite=True)
            ehdu.writeto("egs.fits", overwrite=True)

    from pyds9 import DS9
    ds9 = DS9()
    ds9.set_pyfits(ehdu)
