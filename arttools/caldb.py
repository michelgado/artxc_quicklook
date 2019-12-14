import os, sys
import pandas
from astropy.io import fits
from astropy.table import Table
import datetime
import numpy as np
from .telescope import URDTOTEL
from scipy.spatial.transform import Rotation

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"

TELTOURD = {v:k for k, v in URDTOTEL.items()}

idxdata = fits.getdata(os.path.join(ARTCALDBPATH, indexfname), 1)
idxtabl = Table(idxdata).to_pandas()
idxtabl["CAL_DATE"] = pandas.to_datetime(idxtabl["CAL_DATE"])
idxtabl.set_index("CAL_DATE", inplace=True)

ARTQUATS = {row[0]:Rotation(row[1:]) for row in fits.getdata(os.path.join(ARTCALDBPATH, "artxc_quats_v001.fits"), 1)}
ARTQUATS.update({TELTOURD[row[0]]:Rotation(row[1:]) for row in fits.getdata(os.path.join(ARTCALDBPATH, "artxc_quats_v001.fits"), 1) if row[0] in TELTOURD})


def get_cif(cal_cname, instrume):
    return idxtabl.query("INSTRUME=='%s' and CAL_CNAME=='%s'" %
                               (instrume, cal_cname))

def get_relevat_file(cal_cname, instrume, date=datetime.datetime(2030, 10, 10)):
    caltable = get_cif(cal_cname, instrume)
    didx = caltable.index.get_loc(date, method="ffill")
    row = caltable.iloc[didx]
    fpath = os.path.join(ARTCALDBPATH, row["CAL_DIR"].rstrip(), row["CAL_FILE"].rstrip())
    return fpath

OPAXOFFSET = {TELTOURD[tel]: [x, y] for tel, x, y in fits.getdata(get_relevat_file("OPT_AXIS", "NONE"))}

def get_vigneting_by_urd(urdn):
    """
    to do: put vignmap in the caldb
    """
    return fits.open("/srg/a1/work/ayut/art-xc_vignea_q200_191210.fits")

def get_shadowmask_by_urd(urdn):
    #temporal patch
    urdtobit = {28:2, 22:4, 23:8, 24:10, 25:20, 26:40, 30:80}
    fpath = "/home/andrey/ART-XC/sandbox/artxc_quicklook/newshadowmask/newopenpix%02d.fits" % urdtobit[urdn]
    return np.logical_not(fits.getdata(fpath, 1).astype(np.bool))

def get_shadowmask(urdfile):
    return get_shadowmask_by_urd(urdfile["EVENTS"].header["URDN"])

def get_energycal_by_urd(urdn):
    fpath = get_relevat_file('TCOEF', URDTOTEL[urdn])
    return fits.open(fpath)

def get_energycal(urdfile):
    return get_energycal_by_urd(urdfile["EVENTS"].header["URDN"])

def get_backprofile_by_urdn(urdn):
    return fits.getdata(get_relevat_file("BKG", URDTOTEL[urdn]), 0)

def get_backprofile(urdfile):
    return get_backprofile_by_urdn(urdfile["EVENTS"].header["URDN"])


def get_caldb(caldb_entry_type, telescope, CALDB_path=ARTCALDBPATH, indexfile=indexfname):
    indexfile_path = os.path.join(CALDB_path, indexfile)
    try:
        caldbindx   = fits.open(indexfile_path)
        caldbdata   = caldbindx[1].data
        for entry in caldbdata:
            if entry['CAL_CNAME'] == caldb_entry_type and entry['INSTRUME']==telescope:
                return_path = os.path.join(CALDB_path, entry['CAL_DIR'], entry['CAL_FILE'])
                #print(return_path)
                return return_path
        return None

    except:
        print ('No index file here:' + indexfile_path)
        return None
