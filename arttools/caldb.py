import os, sys
import pandas
from astropy.io import fits
from astropy.table import Table
from functools import lru_cache
import datetime
from astropy import time as atime
import numpy as np
from .telescope import URDTOTEL
from .time import GTI
from scipy.spatial.transform import Rotation
from math import sin, cos, pi


""" ART-XC mjd ref used to compute onboard seconds"""
MJDREF = 51543.875
T0 = 617228538.1056 #first day of ART-XC work

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"

TELTOURD = {v:k for k, v in URDTOTEL.items()}

idxdata = fits.getdata(os.path.join(ARTCALDBPATH, indexfname), 1)
idxtabl = Table(idxdata).to_pandas()
idxtabl["CAL_DATE"] = pandas.to_datetime(idxtabl["CAL_DATE"])
idxtabl.set_index("CAL_DATE", inplace=True)

CUTAPP = None
FLATVIGN = False
FLATBKG = False

qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
qgyro0 = Rotation([0., 0., 0., 1.])
OPAX = np.array([1, 0, 0])


#ARTQUATS = {row[0]:Rotation(row[1:]) for row in fits.getdata(os.path.join(ARTCALDBPATH, "artxc_quats_v001.fits"), 1)}
#ARTQUATS = {row[0]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_v5.fits"), 1)}
ARTQUATS = {row[0]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_V5_rotin.fits"), 1)}
#ARTQUATS.update({TELTOURD[row[0]]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_v5.fits"), 1) if row[0] in TELTOURD})
ARTQUATS.update({TELTOURD[row[0]]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_V5_rotin.fits"), 1) if row[0] in TELTOURD})
#ARTQUATS.update({TELTOURD[row[0]]:Rotation(row[1:]) for row in fits.getdata(os.path.join(ARTCALDBPATH, "artxc_quats_v001.fits"), 1) if row[0] in TELTOURD})

"""
some magical numbers, generally define mean count rate of the background of each detector relative to the mean over all seven
"""
urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.0291298607731770,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

"""
task should store required calibrations

for example, let assume I have a task
def mksomething(urddata, hkdata, attdata, gti):
    which producess someproduct  with some actions
    first product can be producesd only on the crossection of the all input data types and task

    product should store, information about used calibration data

    question concatenate and unify products of the task???

    some can be unifiend some not - i.e. we have different arf and rmf

    questions:
    * how to join exposure maps
    answer:
        if exposure maps have proper normalization, corresponding to effective area, one have simpy sum them
        result: check how the exposure map were computed

    * how to join lightcurves
        usually expect  to concatenate lightcurves in single energy band or sum lightcurves in disjoint  bands

    *how to join spectra
        spectra with same rmf but different arf - simply sum then and weight arfs with exposures
        spectra with different rmf - always use separately ????
"""


def get_caldata(urdn, ctype, gti=GTI([(atime.Time(datetime.datetime.now()) - atime.Time(MJDREF, format="mjd")).sec,]*2,)):
    """
    given the urd as a unique key for calibration data
    """
    caldata = idxtabl.query("INSTRUME=='%s' and CAL_CNAME=='%s'" % (TELTOURD[urdn], ctype)).sort_index()
    timestamps = (atime.Time(caldata.index.values) - atime.Time(MJDREF)).sec
    caldata["tstart"] = timestamps
    caldata["tstop"] = np.roll(timestamps, -1)
    caldata.iloc[-1].tstop = np.inf
    idxloc = np.maximum(np.unique(timestamps.searchsorted(gti.arr)) - 1, 0)
    caldata = caldata.iloc[idxloc].groupby(["CAL_DIR", "CAL_FILE"])
    return {g: GTI(caldata.iloc[idx][["tstart", "tstop"]].values) for g, idx in caldata.groups.items()}

def get_cif(cal_cname, instrume):
    return idxtabl.query("INSTRUME=='%s' and CAL_CNAME=='%s'" % (instrume, cal_cname))

def get_relevat_file(cal_cname, instrume, date=datetime.datetime(2030, 10, 10)):
    caltable = get_cif(cal_cname, instrume)
    didx = caltable.index.get_loc(date, method="ffill")
    row = caltable.iloc[didx]
    fpath = os.path.join(ARTCALDBPATH, row["CAL_DIR"].rstrip(), row["CAL_FILE"].rstrip())
    return fpath

OPAXOFFSET = {TELTOURD[tel]: [x, y] for tel, x, y in fits.getdata(get_relevat_file("OPT_AXIS", "NONE"))}

@lru_cache(maxsize=5)
def get_vigneting_by_urd(urdn):
    """
    to do: put vignmap in the caldb
    """
    return fits.open("/srg/a1/work/ayut/art-xc_vignea_q200_191210.fits")

@lru_cache()
def get_shadowmask_by_urd(urdn):
    global CUTAPP
    #temporal patch
    print(CUTAPP)
    urdtobit = {28:2, 22:4, 23:8, 24:10, 25:20, 26:40, 30:80}
    fpath = "/home/andrey/ART-XC/sandbox/artxc_quicklook/newshadowmask/newopenpix%02d.fits" % urdtobit[urdn]
    mask = np.logical_not(fits.getdata(fpath, 1).astype(np.bool))
    if not CUTAPP is None:
        x, y = np.mgrid[0:48:1, 0:48:1] + 0.5
        maskapp = (OPAXOFFSET[urdn][0] - x)**2. + (OPAXOFFSET[urdn][1] - y)**2. < CUTAPP**2.
        mask = np.logical_and(mask, maskapp)
    return mask

def get_shadowmask(urdfile):
    return get_shadowmask_by_urd(urdfile["EVENTS"].header["URDN"])

@lru_cache()
def get_energycal_by_urd(urdn):
    fpath = get_relevat_file('TCOEF', URDTOTEL[urdn])
    return fits.open(fpath)

def get_energycal(urdfile):
    return get_energycal_by_urd(urdfile["EVENTS"].header["URDN"])

@lru_cache()
def get_backprofile_by_urdn(urdn):
    global FLATBKG
    bkg = fits.getdata(get_relevat_file("BKG", URDTOTEL[urdn]), 0)
    if FLATBKG:
        bkg = np.ones(bkg.shape)
    return bkg

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
