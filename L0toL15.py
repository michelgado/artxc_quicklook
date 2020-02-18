from astropy.io import fits
from astropy.table import Table
import argparse
import sys
import os
import numpy as np
from math import pi
import pandas
import copy

import arttools
from arttools._det_spatial import get_shadowed_pix_mask_for_urddata
from arttools.energy import get_events_energy
from arttools.caldb import get_shadowmask, get_energycal
from arttools.time import get_gti

import socket
import smtplib
from email.message import EmailMessage



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
    try:
        sock = socket.socket()
        sock.connect(("10.5.2.24", 8081))
        sock.send(str.encode(sys.argv[1]))
    except ConnectionRefusedError as conerr:
        msg = EmailMessage()
        msg["to"] = "san@iki.rssi.ru"
        msg["from"] = "art@cosmos.ru"
        msg["subbj"] = "makeL1b fail to start daemon"
        msg.set_content(conerr)
        smtplib.SMTP("localhost").send_message(msg)
    finally:
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

        attdata = arttools.plot.get_attdata(attfname)
        urdfname = fname
        urdfile = fits.open(urdfname)
        urddata = np.copy(urdfile["EVENTS"].data)
        flag = np.ones(urddata.size, np.uint8)

        locgti = get_gti(urdfile)
        locgti.merge_joint()


        caldbfile = get_energycal(urdfile)
        flag[(locgti & attdata.gti).mask_outofgti_times(urddata["TIME"])] = 0

        RA, DEC = np.empty(urddata.size, np.double), np.empty(urddata.size, np.double)
        attmask = attdata.gti.mask_outofgti_times(urddata["TIME"])
        RA[attmask], DEC[attmask] = arttools.orientation.get_photons_sky_coord(urddata[attmask], urdfile["EVENTS"].header["URDN"], attdata)

        ENERGY, xc, yc, grades = get_events_energy(urddata, urdfile["HK"].data, caldbfile)

        shadow = get_shadowmask(urdfile)
        maskshadow = get_shadowed_pix_mask_for_urddata(urddata, shadow)
        flag[np.logical_not(maskshadow)] = 2
        h = copy.copy(urdfile["EVENTS"].header)
        h.pop("NAXIS2")

        cols = urdfile["EVENTS"].data.columns
        cols.add_col(fits.Column(name="ENERGY", array=ENERGY, format="1D", unit="keV"))
        cols.add_col(fits.Column(name="RA", array=np.copy(RA*180./pi), format="1D", unit="deg"))
        cols.add_col(fits.Column(name="DEC", array=np.copy(DEC*180./pi), format="1D", unit="deg"))
        cols.add_col(fits.Column(name="GRADE", array=grades, format="I"))
        cols.add_col(fits.Column(name="FLAG", array=flag, format="I"))
        cols["TIME_I"].array = urddata["TIME_I"]

        newurdtable = fits.BinTableHDU.from_columns(cols, header=h)

        newurdtable.name = "EVENTS"
        gtitable = fits.BinTableHDU(Table(locgti.arr, names=("START", "STOP")), header=urdfile["GTI"].header)
        newfile = fits.HDUList([urdfile[0], newurdtable, urdfile["HK"], gtitable])
        newfile.writeto(os.path.join(outdir, os.path.basename(urdfname)), overwrite=True)

        if not os.path.exists(os.path.join(outdir, os.path.basename(attfname))):
            attfile = fits.open(attfname)
            hdus = [attfile[0], ]
            for telescope in arttools.telescope.TELESCOPES:
                ra, dec, roll = arttools.orientation.quat_to_pol_and_roll(attdata(attdata.times)*arttools.caldb.ARTQUATS[telescope])
                hdus.append(fits.BinTableHDU(Table(np.array([attdata.times, ra*180/pi, dec*180/pi, roll*180/pi]).T, names=("TIME", "RA", "DEC", "ROLL")), header={"TELESCP":telescope}, name=telescope))
            fits.HDUList(hdus).writeto(os.path.join(outdir, os.path.basename(attfname)))
