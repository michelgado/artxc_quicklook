from astropy.io import fits
from astropy.table import Table
import argparse
import sys
import os
import numpy as np
from math import pi
import pandas
import copy

from arttools._det_spatial import get_shadowed_pix_mask_for_urddata
from arttools.energy import get_events_energy
from arttools.orientation import extract_raw_gyro, get_photons_sky_coord, nonzero_quaternions, get_gyro_quat_as_arr, clear_att
from arttools.caldb import get_shadowmask, get_energycal
from arttools.time import get_gti, gti_union, gti_intersection, make_hv_gti

parser = argparse.ArgumentParser(description="process L0 data to L1 format")
parser.add_argument("stem", help="part of the L0 files name, which are euqal to them")

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"
#caldbindex = pandas.DataFrame(fits.getdata(indexfname, "CIF"))

URDTOTEL = {28: "T1",
            22: "T2",
            23: "T3",
            24: "T4",
            25: "T5",
            26: "T6",
            30: "T7"}


if __name__ == "__main__":
    if len(sys.argv) != 4 or "-h" in sys.argv:
        print("description run like that 'python3 L1toL15.py stem outdir'"\
                ", where stem is srg_20190727_214739_000")
        raise ValueError("wrong arguments")
    fname = sys.argv[1]
    stem = fname.rsplit(".")[0]
    outdir = sys.argv[2]
    attfname = sys.argv[3]
    if os.path.abspath(outdir) == os.path.abspath(os.path.dirname(stem)):
        raise ValueError("The L0 files will be overwriten")

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    attfile = fits.open(attfname)
    attdata = clear_att(np.copy(attfile["ORIENTATION"].data))
    #attdata = attdata[nonzero_quaternions(get_gyro_quat_as_arr(attdata))]
    urdfname = fname
    urdfile = fits.open(urdfname)
    urddata = urdfile["EVENTS"].data
    flag = np.ones(urddata.size, np.uint8)

    gti = get_gti(urdfile)
    gti = gti_union(gti[:,:] + [0, 1.])
    gti = gti_intersection(gti, make_hv_gti(urdfile["HK"].data))
    gti = gti_intersection(gti, np.array([attdata["TIME"][[0, -1]],]))

    caldbfile = get_energycal(urdfile)
    masktime = (urddata["TIME"] > attdata["TIME"][0]) & (urddata["TIME"] < attdata["TIME"][-1])
    flag[masktime] = 0
    RA, DEC = np.empty(urddata.size, np.double), np.empty(urddata.size, np.double)
    r, d = get_photons_sky_coord(urddata[masktime],
                    urdfile["EVENTS"].header["URDN"],
                    attdata)
    RA[masktime] = r
    DEC[masktime] = d
    ENERGY, xc, yc, grades = get_events_energy(urddata, urdfile["HK"].data, caldbfile)

    shadow = get_shadowmask(urdfile)
    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    flag[np.logical_not(maskshadow)] = 2
    flag[np.any([urddata["RAW_X"] == 0, urddata["RAW_X"] == 47, urddata["RAW_Y"] == 0, urddata["RAW_Y"]  == 47], axis=0)] = 3
    h = copy.copy(urdfile["EVENTS"].header)
    h.pop("NAXIS2")

    print(urddata["TIME_I"][:10])
    cols = urdfile["EVENTS"].data.columns
    cols.add_col(fits.Column(name="ENERGY", array=ENERGY, format="1D", unit="keV"))
    cols.add_col(fits.Column(name="RA", array=np.copy(RA*180./pi), format="1D", unit="deg"))
    cols.add_col(fits.Column(name="DEC", array=np.copy(DEC*180./pi), format="1D", unit="deg"))
    cols.add_col(fits.Column(name="GRADE", array=grades, format="I"))
    cols.add_col(fits.Column(name="FLAG", array=flag, format="I"))
    cols["TIME_I"].array = urddata["TIME_I"]

    newurdtable = fits.BinTableHDU.from_columns(cols, header=h)

    newurdtable.name = "EVENTS"
    gtitable = urdfile["GTI"]
    gtitable.data = np.array([tuple(g) for g in gti], dtype=gtitable.data.dtype)
    newfile = fits.HDUList([urdfile[0], newurdtable, urdfile["HK"], gtitable])
    newfile.writeto(os.path.join(outdir, os.path.basename(urdfname)), overwrite=True)
