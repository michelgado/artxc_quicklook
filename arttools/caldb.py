import os, sys
import pandas
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from functools import lru_cache
import datetime
from astropy import time as atime
import numpy as np
from .telescope import URDTOTEL, ANYTHINGTOTELESCOPE, ANYTHINGTOURD
from scipy.spatial.transform import Rotation
from math import sin, cos, pi
#from .energy import get_events_energy
import pickle


""" ART-XC mjd ref used to compute onboard seconds"""
MJDREF = 51543.875  # board time MJDREF
T0 = 617228538.1056 #first day of ART-XC work

relativistic_corrections_gti = np.array([
                  [6.24401146e+08, 6.24410608e+08],
                  [6.24410643e+08, 6.26464675e+08],
                  [6.26472730e+08, 6.27085963e+08],
                  [6.27087894e+08, 6.28720330e+08],
                  [6.28720417e+08, 6.30954255e+08]])

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_calib/caldb.indx"

idxtabl = Table(fits.getdata(os.path.join(ARTCALDBPATH, indexfname), 1))
idxtabl = idxtabl.to_pandas()

idxtabl["CAL_VSB"] = [(Time(a + "T" + b) - Time(MJDREF, format="mjd")).sec for a, b in idxtabl[["CAL_VSD", "CAL_VST"]].values]
idxtabl.sort_values(["CAL_DATE", "CAL_VSB"], ascending=[False, True], inplace=True)
idxtabl["CAL_CNAM"] = idxtabl.CAL_CNAM.str.rstrip()
idxtabl["INSTRUME"] = idxtabl.INSTRUME.str.rstrip()
idxtabl = idxtabl.groupby("INSTRUME").apply(lambda x: x[x.CAL_VSB.values == x.CAL_VSB.cummax().values])


CUTAPP = None
FLATVIGN = False
FLATBKG = False
el = None
bkggti = None
imask = None
usenewquat = False

qbokz0 = Rotation([0., -0.707106781186548,  0., 0.707106781186548])*Rotation([ 1.07307728e-05,  2.94924883e-07, -3.05587484e-05, -9.99999999e-01])
qgyro0 = Rotation([0., 0., 0., 1.])
_INST_QUAT = {
    'GYRO': [ 0.               ,  0.               ,  0.               , 1.               ],
    'BOKZ': [ 2.91961004e-05   ,  0.707106989      , -1.40204960e-05   , -0.707106572     ],
    'SED1': [ 0.183012701892219,  0.683012701892219, -0.183012701892219, 0.683012701892219],
    'SED2': [-0.183012701892219,  0.683012701892219,  0.183012701892219, 0.683012701892219]}


OPAX = np.array([1, 0, 0])


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


def get_bokz_timepatches(gti=None):
    patches = np.loadtxt("/srg/a1/work/andrey/ART-XC/Crab/bokz_time_patches.txt").reshape((-1, 2))
    return patches if gti is None else patches[gti.mask_external(patches[:, 0])]

@lru_cache(maxsize=1)
def get_bokz_bti():
    bti = np.loadtxt("/srg/a1/work/andrey/ART-XC/Crab/bokz_btis_patches.txt")
    return bti


@lru_cache(maxsize=1)
def get_bokz_fjump_bti():
    bti = np.loadtxt("/srg/a1/work/andrey/ART-XC/Crab/bad_future_jumps.txt")
    return bti


@lru_cache(maxsize=7)
def get_deadtime_for_dev(dev, gti=None):
    mfile, gti = get_caldata("DEADTIME", ANYTHINGTOURD[dev], gti)[0]
    return fits.getdata(mfile, 1)["DEADTIME"][0]*1e-3


def get_obt_timecorr_calib():
    #return os.path.join(ARTCALDBPATH, "art_clock_model_corr_v07092022.txt")
    return fits.getdata(os.path.join(ARTCALDBPATH, "timecorrection.fits"), 1)


@lru_cache(maxsize=1)
def get_illumination_mask():
    """TODO: put link in the index file"""
    pfile = fits.open(os.path.join(ARTCALDBPATH, "imask8.fits.gz"))
    return pfile[1].data["STOP"], pfile[2].data["EDGES"], pfile[3].data["EDGES"], pfile[4].data.astype(bool)

@lru_cache(maxsize=7)
def get_stray_light_mask_by_urd(urdn):
    return pickle.load(open(os.path.join(ARTCALDBPATH, "stray_light_patches.pkl"), "rb"))[urdn]




def get_azimask_for_urdn(urdn):
    mfile = fits.open(os.path.join(ARTCALDBPATH, "azimuthal_imask7.fits.gz"))
    sbounds = mfile[1].data["OPAXOFFL"]*pi/180./3600.
    ebounds = mfile["EVTOFGRID"].data["SRCEVTOF"]
    ctot = [np.array([hdu.data["SRCOFFL"], hdu.data["AZSTART"], hdu.data["AZSTOP"]]).T for hdu in mfile[3:]]
    return sbounds, ebounds, ctot


def get_caldata(ctype, dev, gti=None):
    """
    given the urd as a unique key for calibration data
    """
    caldata = idxtabl.query("INSTRUME=='%s' and CAL_CNAM=='%s'" % (ANYTHINGTOTELESCOPE.get(dev, dev), ctype)).sort_index()
    if gti is None:
        te, gaps = np.array([caldata.iloc[0].CAL_VSB, np.inf]), np.ones(1, bool)
    else:
        te, gaps = git.make_tedges(caldata.CAL_VSB.vaues)
    ti = np.array([te[:-1], te[1:]]).T[gaps]
    idx = np.searchsorted(caldata.CAL_VSB.values, ti.mean(axis=1)) - 1
    u, ui = np.unique(idx, return_inverse=True)
    return [(os.path.join(ARTCALDBPATH, caldata.iloc[i].CAL_DIR.rstrip(), caldata.iloc[i].CAL_FILE.rstrip()), ti[ui == i]) for i in u]


"""
def get_cif(cal_cname, instrume):
    return idxtabl.query("INSTRUME=='%s' and CAL_CNAM=='%s'" % (instrume, cal_cname))

def get_relevat_file(cal_cname, instrume, date=datetime.datetime(2030, 10, 10)):
    caltable = get_cif(cal_cname, instrume)
    didx = caltable.index.get_loc(date, method="ffill")
    row = caltable.iloc[didx]
    fpath = os.path.join(ARTCALDBPATH, row["CAL_DIR"].rstrip(), row["CAL_FILE"].rstrip())
    return fpath
"""

#OPAXOFFSET = {TELTOURD[tel]: [x, y] for tel, x, y in fits.getdata(get_relevat_file("OPT_AXIS", "NONE"))}
OPAXOFFSET = {28:[21.28, 22.86],
              22:[18.15, 21.02],
              23:[20.65, 20.27],
              24:[20.57, 21.97],
              25:[22.21, 20.97],
              26:[22.41, 23.03],
              30:[20.6, 22.71]}


def get_optical_axis_offset_by_device(dev):
    return OPAXOFFSET[dev]


def set_quat_state(usenew):
    global usenewquat
    usenewquat = usenew

boresight = {dev: Rotation(fits.getdata(get_caldata("BORESIGH", ANYTHINGTOTELESCOPE.get(dev, dev))[0][0], 1)[0]) for dev in [28, 22, 23, 24, 25, 26, 30, "GYRO", "BOKZ"]}
def set_boresight(dev, qset):
    global boresight
    boresight[dev] = qset

#@lru_cache(maxsize=7)
def get_boresight_by_device(dev):
    #return boresight[dev]
    global usenewquat
    #print("usenewquat", usenewquat)
    """
    if int(dev) == 28:
        alpha, beta, angle = 5.95739460e+00, -1.40807804e+00,  1.85982924e-04
        return Rotation.from_rotvec([cos(alpha)*cos(beta)*angle, cos(alpha)*sin(beta)*angle, sin(alpha)*angle])
        #return pickle.load(open("/srg/a1/work/andrey/ART-XC/Crab/qc28.pkl", "rb"))
    if usenewquat and dev in [28, 22, 23, 24, 25, 26, 30]:
        #alpha, beta, angle = pickle.load(open("/srg/a1/work/andrey/ART-XC/Crab/%d_qcorr_lsrc.pkl" % int(dev), "rb")).x
        alpha, beta, angle = pickle.load(open("/srg/a1/work/andrey/ART-XC/Crab/%d_qcorr_bokz_slow.pkl" % int(dev), "rb")).x
        return Rotation.from_rotvec([cos(alpha)*cos(beta)*angle, cos(alpha)*sin(beta)*angle, sin(alpha)*angle])
        return pickle.load(open("/srg/a1/work/andrey/ART-XC/Crab/%d_qcorr_bokz_new.pkl" % dev, "rb"))
    if str(dev).lower() == "bokz":
        #return Rotation([ 0.12630457, -0.00162314, -0.00102878,  0.99198965]) # temporal patch, based on the comparison of GYRO and BOKZ from 2020 02 01 (correction presented at 2023 04 04)
        return Rotation([ 0.12632717, -0.00164866, -0.00102702,  0.99198673])# temporal patch, based on the comparison of GYRO and BOKZ from 2020 02 01 (correction presented at 2023 04 04)
    """
    """
    if dev == 28 and usenewquat:
        #return Rotation([-9.00361776e-05, -5.77130725e-05,  1.19482282e-05,  9.99999994e-01])
        return Rotation([7.43509950e-04, -4.22695966e-05, -4.01483537e-06,  9.99999723e-01])
    if dev == 22 and usenewquat:
        #return Rotation([-2.47195250e-04, -2.62154574e-05, -8.48821491e-06,  9.99999969e-01])
        return Rotation([-4.08141862e-04, -4.21216133e-05, -2.19725598e-06,  9.99999916e-01])
    if dev == 23 and usenewquat:
        return Rotation([1.28239400e-03, -4.41720522e-05, -2.55864811e-06,  9.99999177e-01])
    """

    if str(dev).lower() == "bokz":
        return Rotation([ 0.12632717, -0.00164866, -0.00102702,  0.99198673])# temporal patch, based on the comparison of GYRO and BOKZ from 2020 02 01 (correction presented at 2023 04 04)
    return Rotation(fits.getdata(get_caldata("BORESIGH", ANYTHINGTOTELESCOPE.get(dev, dev))[0][0], 1)[0])


@lru_cache(maxsize=7)
def get_vigneting_by_urd(urdn):
    """
    to do: put vignmap in the caldb
    """
    return fits.open(os.path.join(ARTCALDBPATH, "art-xc_vignea_q200_191210.fits"))

def get_highres_vign_model():
    return fits.open(os.path.join(ARTCALDBPATH, "art-xc_vignea_q200_191210.fits"))


@lru_cache()
def get_shadowmask_by_urd(urdn):
    global CUTAPP
    #temporal patch
    """
    urdtobit = {28:2, 22:4, 23:8, 24:10, 25:20, 26:40, 30:80}
    fpath = os.path.join(ARTCALDBPATH, "artxc_detmask_%s_20200414_v001.fits" % URDTOTEL[urdn])
    #mask = np.logical_not(fits.getdata(fpath, 1).astype(bool))
    """
    #print("urdn", urdn, ANYTHINGTOTELESCOPE[urdn])
    fpath = get_caldata("DETMASK", ANYTHINGTOTELESCOPE.get(urdn, urdn))[0][0]
    #print("fpath!!!", fpath)
    mask = np.copy(fits.getdata(fpath, 1)).astype(bool)
    #mask = np.ones((48, 48), bool)
    if not CUTAPP is None:
        x, y = np.mgrid[0:48:1, 0:48:1] + 0.5
        maskapp = (OPAXOFFSET[urdn][0] - x)**2. + (OPAXOFFSET[urdn][1] - y)**2. < CUTAPP**2.
        mask = np.logical_and(mask, maskapp)
    return mask

def get_shadowmask(urdfile):
    return get_shadowmask_by_urd(urdfile["EVENTS"].header["URDN"])

@lru_cache(maxsize=7)
def get_energycal_by_urd(urdn):
    fpath = get_caldata('THRESHOL', ANYTHINGTOTELESCOPE.get(urdn, urdn))[0][0]
    return fits.open(fpath)

def get_energycal(urdfile):
    return get_energycal_by_urd(urdfile["EVENTS"].header["URDN"])

def get_escale_by_urd(urdn):
    fpath = get_caldata('ESCALE', ANYTHINGTOTELESCOPE.get(urdn, urdn))[0][0]
    return fits.getdata(fpath)

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
    dt = 0.
    if str(dev).lower() == "gyro":
        dt = 0.97
    if str(dev).lower() == "bokz":
        dt = 1.672
    return dt


fshifts = [l.rstrip().rsplit() for l in open('/srg/a1/work/andrey/ART-XC/att_compressed/bokz_shifts.list')]
fshifts = {os.path.basename(name): int(k) for name, k in fshifts}
def get_specific_fileshift(fname):
    return fshifts.get(os.path.basename(fname), 0.)



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


def get_ayut_inversed_psf_data_packed():
    """
    ipsf1 = fits.open(os.path.join(ARTCALDBPATH, "iPSF_ayut.fits"))
    mm = np.zeros((121, 121), float)
    mm[56:65, 56:65] = 1.
    ipsf = fits.HDUList([ipsf1[0], ipsf1[1], fits.ImageHDU(np.tile(mm, (ipsf1[2].data.shape[0], ipsf1[2].data.shape[1], 1, 1)))])
    """
    ipsf = fits.open(os.path.join(ARTCALDBPATH, "marchall_ipsf.fits.gz"))
    #ipsf = fits.open(os.path.join(ARTCALDBPATH, "iPSF_ayut.fits"))
    #ipsf = fits.open(os.path.join(ARTCALDBPATH, "iPSF_hybrid.fits.gz"))
    return ipsf

@lru_cache(maxsize=1)
def get_ayut_inverse_psf_datacube_packed():
    ipsf = np.copy(get_ayut_inversed_psf_data_packed()[2].data)
    #ipsf = fits.getdata(os.path.join(ARTCALDBPATH, "iPSF_hybrid.fits.gz"), 2)
    #ipsf = pickle.load(open("/srg/a1/work/srg/ARTCALDB/caldb_files/iPSF_marshall.pkl", "rb"))
    return ipsf

def get_arf():
    return fits.open(os.path.join(ARTCALDBPATH, "artxc_arf_v000.fits"))

import yaml
def get_telescope_crabrates(dev=None):
    crates = yaml.load(open(os.path.join(ARTCALDBPATH, "crab_urd_rates_3surveys.txt")), Loader=yaml.FullLoader)
    if dev is None:
        return crates
    else:
        return crates[TELTOURD.get(dev, dev)]
