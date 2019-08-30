from astropy.io import fits
import argparse
import sys
import os
import numpy as np
from math import pi
import pandas

from arttools._det_spatial import get_shadowed_pix_mask_for_urddata
from arttools.energy import get_events_energy
from arttools.plot import get_photons_sky_coord

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


def get_caldb(caldb_entry_type, telescope, CALDB_path=ARTCALDBPATH, indexfile=indexfname):
    """
    Get entry from CALDB index file
    v000/hart/250719 very dirty, unprotected paths!
    v001/hart/070819 refractored
    """
  
    #Try to open file
    indexfile_path = CALDB_path + indexfile
    try:
        caldbindx   = fits.open(indexfile_path)
        caldbdata   = caldbindx[1].data
        for entry in caldbdata:
            print(entry["CAL_CNAME"], caldb_entry_type, entry["INSTRUME"], telescope)
            if entry['CAL_CNAME'] == caldb_entry_type and entry['INSTRUME']==telescope:
                return_path = CALDB_path +entry['CAL_DIR'] +entry['CAL_FILE']
                return return_path
        return None

    except:
        print ('No index file here:' + indexfile_path)
        return None

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
    attdata = attfile["ORIENTATION"].data[10:]
    urdfname = fname 
    urdfile = fits.open(urdfname)
    urddata = urdfile["EVENTS"].data

    """
    queryenergycalib = {"TEL": URDTOTEL[urdfile[1].header["URDN"], 
                        "CAL_NAME": "TCOEF"}
    caldbfilename = caldbindex.query([
    """
    print(get_caldb("TCOEF", URDTOTEL[urdfile[1].header["URDN"]]))
    caldbfile = fits.open(get_caldb("TCOEF", URDTOTEL[urdfile[1].header["URDN"]]))
    masktime = (urddata["TIME"] > attdata["TIME"][0]) & (urddata["TIME"] < attdata["TIME"][-1])
    mask = np.copy(masktime)
    urddata = urddata[masktime]

    shadow = np.logical_not(fits.getdata(get_caldb("OOFPIX", URDTOTEL[urdfile[1].header["URDN"]])))
    maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
    urddata = urddata[maskshadow]
    mask[mask] = maskshadow


    RA, DEC = get_photons_sky_coord(urddata, 
                    urdfile["EVENTS"].header["URDN"], 
                    attdata)
    maskenergy, ENERGY, xc, yc = get_events_energy(urddata,
                                    urdfile["HK"].data, caldbfile)
    mask[mask] = maskenergy
    print(mask.size)
    print(mask.sum())
    print(urddata.size)
    print(ENERGY.size)
    newurdtable = fits.BinTableHDU.from_columns(fits.ColDefs(
        [fits.Column(name=cd.name, array=cd.array[mask], format=cd.format, unit=cd.unit) \
                for cd in urddata.columns] + 
        [fits.Column(name="ENERGY", array=ENERGY, format="1D", unit="keV"), 
         fits.Column(name="RA", array=RA[maskenergy], format="1D", unit="deg"), 
         fits.Column(name="DEC", array=DEC[maskenergy], format="1D", unit="deg")]))
    newurdtable.name = "EVENTS"

    urdfile["EVENTS"] = newurdtable
    urdfile.writeto(os.path.join(outdir, os.path.basename(urdfname)), overwrite=True)


