import os, sys
import pandas
from astropy.io import fits
from astropy.table import Table
import datetime
import numpy as np
from .telescope import URDTOTEL

ARTCALDBPATH = os.environ["ARTCALDB"]
indexfname = "artxc_index.fits"

idxdata = fits.getdata(os.path.join(ARTCALDBPATH, indexfname), 1)
idxtabl = Table(idxdata).to_pandas()
idxtabl["CAL_DATE"] = pandas.to_datetime(idxtabl["CAL_DATE"])
idxtabl.set_index("CAL_DATE", inplace=True)

def get_cif(cal_cname, instrume):
    return idxtabl.query("INSTRUME=='%s' and CAL_CNAME=='%s'" %
                               (instrume, cal_cname))

def get_relevat_file(cal_cname, instrume, date=datetime.datetime(2030, 10, 10)):
    caltable = get_cif(cal_cname, instrume)
    didx = caltable.index.get_loc(date, method="ffill")
    row = caltable.iloc[didx]
    fpath = os.path.join(ARTCALDBPATH, row["CAL_DIR"], row["CAL_FILE"])
    return fpath

def get_shadowmask(urdfile):
    fpath = get_relevat_file('OOFPIX', URDTOTEL[urdfile["EVENTS"].header["URDN"]])
    return np.logical_not(fits.getdata(fpath, 1).astype(np.bool))

def get_energycal(urdfile):
    fpath = get_relevat_file('TCOEF', URDTOTEL[urdfile["EVENTS"].header["URDN"]])
    return fits.open(fpath)

def get_caldb(caldb_entry_type, telescope, CALDB_path=ARTCALDBPATH, indexfile=indexfname):
    #print(caldb_entry_type, telescope)

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