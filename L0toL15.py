from astropy.io import fits
import argparse
import sys
import os
import numpy as np
from math import pi

from arttools.energy import get_events_energy
from arttools.plot import get_photons_sky_coord

parser = argparse.ArgumentParser(description="process L0 data to L1 format")
parser.add_argument("stem", help="part of the L0 files name, which are euqal to them")

if __name__ == "__main__":
    if len(sys.argv) != 4 or "-h" in sys.argv:
        print("description run like that 'python3 L1toL15.py stem outdir'"\
                ", where stem is srg_20190727_214739_000") 
        raise ValueError("wrong arguments")
    fname = sys.argv[1]
    stem = fname.rsplit(".")[0]
    outdir = sys.argv[2]
    if os.path.abspath(outdir) == os.path.abspath(os.path.dirname(stem)):
        raise ValueError("The L0 files will be overwriten")

    caldbfilename = sys.argv[3]
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    attfile = fits.open(stem + "_gyro.fits")
    attdata = attfile["ORIENTATION"].data[10:]
    caldbfile = fits.open(caldbfilename)

    urdfname = fname 
    urdfile = fits.open(urdfname)
    s, e = np.searchsorted(urdfile["EVENTS"].data["TIME"], 
            attdata["TIME"][[0, -1]])

    urddata = urdfile["EVENTS"].data[s:e - 1]
    RA, DEC = get_photons_sky_coord(urddata, 
            urdfile["EVENTS"].header["URDN"], 
            attdata)
    mask, ENERGY, xc, yc = get_events_energy(urddata,
            urdfile["HK"].data, caldbfile)
    newdata = np.array(urddata[mask])
    newdata = np.lib.recfunctions.append_fields(newdata, 
        ["RA", "DEC", "ENERGY"],
        [RA[mask], DEC[mask], ENERGY])

    newhdu = fits.TableHDU(newdata)
    newhdu.header.update(urdfile["EVENTS"].header)
    lhdu = fits.HDUList([urdfile["PRIMARY"], newhdu, urdfile["HK"], urdfile["GTI"]])
    lhdu.writeto(os.path.join(outdir, os.path.basename(urdfname)), overwrite=True)


