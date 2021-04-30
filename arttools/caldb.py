import os, sys
import pandas
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from functools import lru_cache
import datetime
from astropy import time as atime
import numpy as np
from .telescope import URDTOTEL
from .time import GTI, get_gti
from scipy.spatial.transform import Rotation
from math import sin, cos, pi
#from .energy import get_events_energy
import pickle


""" ART-XC mjd ref used to compute onboard seconds"""
MJDREF = 51543.875
T0 = 617228538.1056 #first day of ART-XC work

relativistic_corrections_gti = GTI([
                  [6.24401146e+08, 6.24410608e+08],
                  [6.24410643e+08, 6.26464675e+08],
                  [6.26472730e+08, 6.27085963e+08],
                  [6.27087894e+08, 6.28720330e+08],
                  [6.28720417e+08, 6.30954255e+08]])

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"

TELTOURD = {v:k for k, v in URDTOTEL.items()}

idxtabl = Table(fits.getdata(os.path.join(ARTCALDBPATH, indexfname), 1))
idxtabl["CAL_DATE"] = idxtabl["CAL_DATE"][:, 0]
idxtabl = idxtabl.to_pandas()
idxtabl["CAL_DATE_ISO"] = (Time(idxtabl.REF_TIME.values, format="mjd") + \
                            TimeDelta(idxtabl.CAL_DATE.values, format="sec")).to_datetime()
idxtabl.set_index("CAL_DATE_ISO", inplace=True)

CUTAPP = None
FLATVIGN = False
FLATBKG = False
el = None
bkggti = None

qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])
qgyro0 = Rotation([0., 0., 0., 1.])
OPAX = np.array([1, 0, 0])

#ARTQUATS = {row[0]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_V5_rotin.fits"), 1)}
#ARTQUATS.update({TELTOURD[row[0]]:Rotation(row[1:]) for row in fits.getdata(os.path.join("/srg/a1/work/andrey/ART-XC/Quats_V5", "ART_QUATS_V5_rotin.fits"), 1) if row[0] in TELTOURD})

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

"""
def apply_energycal(hdulist, gti=None):
    URDN = hdulist["EVENTS"].header["URDN"]
    if gti is None: gti = get_gti(hdulist)
    ecals = get_caldata(URDN, "ENERGY", gti)
    data = Table(hdulist["EVENTS"])
    energy = np.empty(hdulist["EVENTS"].data.size, np.double)
    grade = np.empty(hdulist["EVENTS"].data.size, np.double)
    mask = np.ones(energy.size, np.bool)
    calinfo = {"TSTART": [], "TSTOP": [], "CALNAME": [], "CALVER": [], "INSTRUME": []}

    for e, gtiloc in ecals:
        gtimask = gtiloc.mask_external(data["TIME"])
        ecal = fits.open(e)
        eloc, xc, yc, gloc = get_events_energy(data[gtimask], hdulist["HK"].data, ecal)
        energy[gtimask] = eloc
        grage[gtimask] = gloc
        mask[gtimask] = False
        for tstart, tstop in gtiloc.arr:
            calinfo["TSTART"].append(tstart)
            calinfo["TSTOP"].append(tstop)
            calinfo["INSTRUME"].append(URDTOTEL[URDN])
            calinfo["CALNAME"].append(e)
            calinfo["CALVER"].append(ecal[0].header["VERSION"])
    if np.any(mask):
        raise ValueError("Some data is not covered with calibrations")
    hdulist.append(fits.BintableHDU(data=Table(calinfo), name="CALIB")
    #energy, xc, yc, grade = get_events_energy(hdulist["EVENTS"], hdulist["HK"]
"""



def get_cif(cal_cname, instrume):
    return idxtabl.query("INSTRUME=='%s' and CAL_CNAME=='%s'" % (instrume, cal_cname))

def get_relevat_file(cal_cname, instrume, date=datetime.datetime(2030, 10, 10)):
    caltable = get_cif(cal_cname, instrume)
    didx = caltable.index.get_loc(date, method="ffill")
    row = caltable.iloc[didx]
    fpath = os.path.join(ARTCALDBPATH, row["CAL_DIR"].rstrip(), row["CAL_FILE"].rstrip())
    return fpath

OPAXOFFSET = {TELTOURD[tel]: [x, y] for tel, x, y in fits.getdata(get_relevat_file("OPT_AXIS", "NONE"))}
OPAXOFFSET = {28:[21.28, 22.86],
              22:[18.15, 21.02],
              23:[20.65, 20.27],
              24:[20.57, 21.97],
              25:[22.21, 20.97],
              26:[22.41, 23.03],
              30:[20.6, 22.71]}


def get_optical_axis_offset_by_device(dev):
    return OPAXOFFSET[dev]


@lru_cache(maxsize=7)
def get_boresight_by_device(dev):
    return Rotation(fits.getdata(get_relevat_file("BORESIGHT", URDTOTEL.get(dev, dev)), 1)[0])


@lru_cache(maxsize=7)
def get_vigneting_by_urd(urdn):
    """
    to do: put vignmap in the caldb
    """
    return fits.open(os.path.join(ARTCALDBPATH, "art-xc_vignea_q200_191210.fits"))

@lru_cache()
def get_shadowmask_by_urd(urdn):
    global CUTAPP
    #temporal patch
    print(CUTAPP)
    urdtobit = {28:2, 22:4, 23:8, 24:10, 25:20, 26:40, 30:80}
    fpath = os.path.join(ARTCALDBPATH, "artxc_detmask_%s_20200414_v001.fits" % URDTOTEL[urdn])
    #mask = np.logical_not(fits.getdata(fpath, 1).astype(np.bool))
    mask = np.copy(fits.getdata(fpath, 1)).astype(np.bool)
    if not CUTAPP is None:
        x, y = np.mgrid[0:48:1, 0:48:1] + 0.5
        maskapp = (OPAXOFFSET[urdn][0] - x)**2. + (OPAXOFFSET[urdn][1] - y)**2. < CUTAPP**2.
        mask = np.logical_and(mask, maskapp)
    return mask

def get_shadowmask(urdfile):
    return get_shadowmask_by_urd(urdfile["EVENTS"].header["URDN"])

@lru_cache(maxsize=7)
def get_energycal_by_urd(urdn):
    fpath = get_relevat_file('TCOEF', URDTOTEL[urdn])
    return fits.open(fpath)

def get_energycal(urdfile):
    return get_energycal_by_urd(urdfile["EVENTS"].header["URDN"])

@lru_cache(maxsize=7)
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
    except:
        print ('No index file here:' + indexfile_path)

#temporary solution for time shifts in different device relative to spacecraft time
def get_device_timeshift(dev):
    return 0.97 if dev == "gyro" else 1.55

def get_background_for_urdn(urdn):
    global el, bkggti
    x = np.arange(48)
    y = np.arange(48)
    e = np.arange(4., 150., 1.)
    g = np.arange(0, 16)
    grid = {"RAW_X": x, "RAW_Y": y, "ENERGY": e, "GRADE": g}

    if el is None:
        el, bkggti = pickle.load(open(os.path.join(ARTCALDBPATH, "bkghist2.pickle"), "rb"))
    return grid, el.get(urdn)/bkggti.get(urdn).exposure

def get_overall_background():
    global el, bkggti
    x = np.arange(0, 48)
    y = np.arange(0, 48)
    e = np.arange(4., 150., 1.)
    g = np.arange(0, 16)
    grid = {"RAW_X": x, "RAW_Y": y, "ENERGY": e, "GRADE": g}

    if el is None:
        el, bkggti = pickle.load(open(os.path.join(ARTCALDBPATH, "bkghist2.pickle"), "rb"))
    return grid, sum(el.values())

@lru_cache(maxsize=1)
def get_crabspec():
    spec, ee, ge = pickle.load(open(os.path.join(ARTCALDBPATH, "crabspec3.pkl"), "rb"))
    grid = {"ENERGY": ee, "GRADE": ge}
    return grid, spec

def get_crabspec_for_filters(filters):
    grid, spec = get_crabspec()
    emask = filters["ENERGY"].apply(grid["ENERGY"])
    emasks = np.logical_and(emask[1:], emask[:-1])
    gmask = filters["GRADE"].apply(grid["GRADE"])
    return {"ENERGY": grid["ENERGY"][emask], "GRADE": grid['GRADE'][gmask]}, spec[emasks, :][:, gmask]


@lru_cache(maxsize=14)
def make_background_brightnes_profile(urdn, filterfunc):
    """
    saved background is a 4d cube which is accumulated from the detector in the
    several observations of empty fields.
    the axis of the cube corresponds to the RAW_X, RAW_Y, ENERGY and grade axis
    """
    x = np.arange(1, 47)
    y = np.arange(1, 47)
    e = np.arange(4.5, 150., 1.)
    g = np.arange(0, 16)

    eidx = np.arange(e.size)[filterfunc(energy=e)]
    gidx = np.arange(g.size)[filterfunc(grade=g)]
    return el[0][urdn][1:-1, 1:-1, :, gidx].sum(axis=3)[:, :, eidx].sum(axis=2)/el[1][urdn].exposure

def get_filtered_backgrounds_ratio(urdn, f1, f2):
    return np.sum(make_background_brightnes_profile(urdn, f1))/np.sum(make_background_brightnes_profile(urdn, f2))

def default_useful_events_filter(energy=None, grade=None):
    return ((energy > 4.) & (energy < 12.)), ((grade > -1) & (grade < 10.))

def default_background_events_filter(energy=None, grade=None):
    return ((energy > 40.) & (energy < 100.)), ((grade > -1) & (grade < 10.))

@lru_cache(maxsize=7)
def get_inverse_psf():
    return fits.open(os.path.join(ARTCALDBPATH, "inverse_psf.fits"))

@lru_cache(maxsize=1)
def get_inverse_psf_data():
    ipsf = pickle.load(open(os.path.join(ARTCALDBPATH, "invert_psf_v9_53pix.pickle"), "rb"))
    return ipsf

def get_inversed_psf_data_packed():
    ipsf = fits.open(os.path.join(ARTCALDBPATH, "iPSF.fits"))
    return ipsf

@lru_cache(maxsize=1)
def get_inverse_psf_datacube_packed():
    ipsf = np.copy(get_inversed_psf_data_packed()["iPSF"].data)
    return ipsf

def get_ayut_inversed_psf_data_packed():
    ipsf = fits.open(os.path.join(ARTCALDBPATH, "iPSF_ayut.fits"))
    return ipsf

@lru_cache(maxsize=1)
def get_ayut_inverse_psf_datacube_packed():
    ipsf = np.copy(get_ayut_inversed_psf_data_packed()[1].data)
    return ipsf


def get_arf():
    return fits.open(os.path.join(ARTCALDBPATH, "artxc_arf_v000.fits"))
