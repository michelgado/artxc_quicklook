#!/usr/bin/env python3
# -*- coding: utf8 -*-
'''
artdatasel

General description:
    
INPUT:
        
OUTPUT:
        

TODO:


Stand-alone version v001

15/08/19 v001 hart
    first stand-alone version
    all hail the hypnotoad

15/08/19 v001a hart
    added UI
    all hail the hypnotoad
'''

import os, shutil, argparse
import datetime
import sys
from astropy.io import fits
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter


from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy.time import Time
from astropy.wcs import WCS

import subprocess

def merge_gti_intersections(gti):
    """
    we want to erase all casess when time intervals in gti are intersected

    to do that we use a simple algorithm:

    we want to find all start and end times for which all
    start and end times of other gtis are greater or lesser....
    such start and end times contain intersected gti intervals and will be start and end times of new gti

    1. sort gtis by the start time in ascending order
    1.1 find sorted index of raveled  sorted gtis where start times ar evend and end times are odd
    1.2 check start times which have sorted index equal to their positional indexes
    1.3 extract these times - they are start times for new gtis

    2. repeat for end times

    3. compbine obtained start and end times in new gti
    """
    gtiss = np.argsort(gti[:,0])
    gtise = np.argsort(gti[:,1])

    gtirs = np.ravel(gti[gtiss, :])
    gtire = np.ravel(gti[gtise, :])
    gtiidxs = np.argsort(gtirs)
    gtiidxe = np.argsort(gtire)

    newgti = np.array([
        gti[gtiss[np.arange(0, gtiidxs.size, 2) == gtiidxs[0::2]], 0],
        gti[gtise[np.arange(1, gtiidxe.size, 2) == gtiidxe[1::2]], 1]
                ]).T

    return newgti

def filter_nonitersect(gti, gtifilt):
    if gtifilt.size == 0 or gti.size == 0:
        return np.empty((0, 2), np.double)
    gti = gti[(gti[:, 0] < gtifilt[-1, 1]) & (gti[:, 1] > gtifilt[0, 0])]
    gtis = np.searchsorted(gtifilt[:,1], gti[:,0])
    mask = gti[:,1] > gtifilt[gtis, 0]
    return gti[mask]

def intervals_in_gti(gti, tlist):
    return np.searchsorted(gti[:, 0], tlist) - 1 == np.searchsorted(gti[:, 1], tlist)

def check_gti_shape(gti):
    if gti.ndim != 2 or gti.shape[1] != 2:
        raise ValueError("gti is expected to be a numpy array of shape (n, 2), good luck next time")

def overall_gti(gti1, gti2):
    check_gti_shape(gti1)
    check_gti_shape(gti2)

    gti1 = merge_gti_intersections(gti1)
    gti2 = merge_gti_intersections(gti2)

    gti1 = filter_nonitersect(gti1, gti2)
    gti2 = filter_nonitersect(gti2, gti1)

    tend = np.concatenate([gti1[intervals_in_gti(gti2, gti1[:,1]), 1], gti2[intervals_in_gti(gti1, gti2[:,1]), 1]])
    tend = np.unique(tend)
    ts = np.sort(np.concatenate([gti1[:,0], gti2[:,0]]))
    gtinew = np.array([ts[np.searchsorted(ts, tend) - 1], tend]).T
    return gtinew

def run(command, verbose=False):
    if verbose:
        print('command run: {}'.format(command))  # add timestamp
    try:
        retcode = subprocess.call(command, shell=True)
        if retcode < 0:
            print("command run: terminated by signal", -retcode, file=sys.stderr)
       # else:
       #     print("command run: returned", retcode, file=sys.stderr)
    except OSError as e:
        print("command run: execution failed:", e, file=sys.stderr)




def hk_gti(hk, VOLTAGE_LIMIT):
    t_hk, v_hk = hk['TIME'], hk['HV']
    hv_mask    = v_hk<=VOLTAGE_LIMIT
    t_hk       = t_hk[hv_mask]
    v_gti      = np.array([t_hk-5., t_hk+5.]).T
    return v_gti



def select_data_from_L0_STRICT_POINTING(stem, subvers, obsstart, obsstop, src_ra, src_dec, obsid, modules='1111111'):

    #Attitude constants
    ANG_SPEED_TOLERANCE = 1. #arcsec/s    
    #During pointing angular speed of GYROS should not exceed ANG_SPEED_TOLERANCE 
    POINTING_TOLERANCE  = 60.*30. #arcsec
    #During the pointing GYROS axis should have an offset
    #from source position that is smaller than POINTING_TOLERANCE
    #Hk constants
    VOLTAGE_LIMIT = -95. #V
    #Voltage at detector should be less than VOLTAGE_LIMIT
    
    GTI_TOLERANCE = 0.02 #s
    #GTIs with gap less than 0.02 s will be merged
    
    wdir      = '/srg/a1/work/oper/data/2019/'
    L0        = wdir + stem+'/L0/'
    L1        = wdir + stem+'/L1/'
    stem_tail = '_urd.fits'
      
    
    
    obsisostart, obsisostop = Time(obsstart.replace('.','-'), format='iso'),\
                              Time(obsstop.replace('.','-'), format='iso')
    print ('Looking for data for obsid:',obsid)
    print ('starting at:',obsisostart)
    print ('  ending at:',obsisostop)
    MJDREF  =            51543.875
    predicted_obsonboardstart = (obsisostart.mjd - MJDREF)*86400
    predicted_obsonboardstop  = (obsisostop.mjd - MJDREF)*86400
    
    gyro_path = L1 + stem+'_'+subvers+'_gyro_att.fits'
    attfile = fits.open(gyro_path)
    time   = np.array(attfile[1].data['TIME'])
    obs_time_mask = np.bitwise_and(time>=predicted_obsonboardstart,\
                                   time<=predicted_obsonboardstop)

    time   = time[obs_time_mask]
    ra     =  np.array(attfile[1].data['RA'])[obs_time_mask]
    dec    =  np.array(attfile[1].data['DEC'])[obs_time_mask]
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    src_coords = SkyCoord(ra=src_ra*u.degree, dec=src_dec*u.degree, frame='icrs')


    timeints  = time[1:] - time[:-1]
    meantimes = (time[1:] + time[:-1])*0.5

    deg2arcsec  =  3600.
    angular_speeds = []
    good_t  = []
    for (c1,c2,dt,mt) in zip(coords[:-1],coords[1::], timeints[::],meantimes[::]):
        if dt>0.:
            offset = c1.separation(c2).degree*deg2arcsec
            angular_speeds.append(offset/dt)
            good_t.append(mt)
    good_t = np.array(good_t)
    angular_speeds = np.array(angular_speeds)
    ang_speed_mask = angular_speeds<=ANG_SPEED_TOLERANCE
    unstable       = angular_speeds>ANG_SPEED_TOLERANCE
    if len(angular_speeds[unstable]):
        print ('Pointing is unstable during', np.round(np.sum(timeints[unstable])), 's ')
    settled_gti        = np.array([time[:-1][ang_speed_mask], time[1:][ang_speed_mask]]).T    

    offsets     = src_coords.separation(coords).degree
    offset_mask_1 = offsets[:-1]<=POINTING_TOLERANCE
    offset_mask_2 = offsets[1:]<=POINTING_TOLERANCE
    offset_mask   = np.bitwise_and(offset_mask_1,offset_mask_2)
    pointing_gti =np.array([time[:-1][offset_mask], time[1:][offset_mask]]).T
    att_gti = overall_gti(settled_gti, pointing_gti)


    module_names     = ['02','04','08','10','20','40','80']
    telescope_names  = ['_t1','_t2','_t3','_t4','_t5','_t6','_t7']    
    selected_modules = []
    for (idx, module_name) in zip(modules, module_names):
        if idx=='1':
            selected_modules.append(module_name)
    for module,telname in zip(selected_modules,telescope_names):
        evtlist_path = L0 + stem +'_'+subvers+'.'+ module + stem_tail
        print ("Opened URD"+module+" eventlist:",evtlist_path)
        evtlist = fits.open(evtlist_path)
        #   read GTI 
        gti_start   = np.array(evtlist['GTI'].data['START'])
        gti_stop    = np.array(evtlist['GTI'].data['STOP'])
        gti_total   = np.sum(gti_stop-gti_start)
        gti = np.array([gti_start,gti_stop]).T
        mjdref = float(evtlist['EVENTS'].header['MJDREF'])
        mjdstart, mjdstop = mjdref + gti_start[0]/86400 , mjdref + gti_stop[-1]/86400  
        isosta, isosto = Time(mjdstart, format='mjd').iso, Time(mjdstop, format='mjd').iso
        print ('events from:',isosta)
        print ('      up to:',isosto)
        print('Total observation length',int(np.round(gti_total,0)),' s')
        overall_start, overall_stop = obsisostart, obsisostop
        if isosta<=obsisostart and isosto>=obsisostop:
            print ('!all observation is inside the data!')
        elif isosta>obsisostart:
            print ('!beginning of observation is truncated!')
            overall_start = isosta
        elif isosto<obsisostop:
            print ('!end of observation is truncated!')
            overall_stop = isosto
            
        obsonboardstart = (overall_start.mjd - mjdref)*86400
        obsonboardstop  = (overall_stop.mjd - mjdref)*86400
        print ('selected ',np.round(obsonboardstop-obsonboardstart),' s')
        obs_gti   = np.array([[obsonboardstart],[obsonboardstop]]).T
        gti = overall_gti(gti, obs_gti)
        v_gti =  hk_gti(evtlist['HK'].data, VOLTAGE_LIMIT)
        gti = overall_gti(gti, v_gti)
        total_gti = merge_gti_intersections(overall_gti(gti, att_gti)).T

        c1 = fits.Column(name='START', array=total_gti[0], format='1D', unit='sec')
        c2 = fits.Column(name='STOP', array=total_gti[1], format='1D', unit='sec')
        gtitable = fits.BinTableHDU().from_columns([c1, c2])
        gtitable.name = 'GTI'
        gtitable.header = evtlist['GTI'].header
        evtlist[1].header['RA_OBJ'] = str(src_ra)
        evtlist[1].header['DEC_OBJ'] = str(src_dec)
        evtlist[1].header['ObsID'] = str(obsid)
        outfile = fits.HDUList([evtlist[0],evtlist[1],evtlist[2], gtitable])
        tmpname  = 'art'+obsid+'_urd'+module+'_tmp.fits'
        clfile  = 'art'+obsid+telname+'_cl.fits'
        REMOVE_TMP_FILE = 'rm {tmpfile}'
        try:
            run(REMOVE_TMP_FILE.format(tmpfile=clfile))
            run(REMOVE_TMP_FILE.format(tmpfile=tmpname))
        except:
            pass
        outfile.writeto(tmpname, clobber=True)
        RUN_GTIFILTER="export HEADASPROMPT=/dev/null;fcopy infile='{infile}[gtifilter()]' outfile='!{outfile}'"
        run(RUN_GTIFILTER.format(infile=tmpname,outfile=tmpname))
        RUN_GTIMERGE="export HEADASPROMPT=/dev/null;ftadjustgti infile='{infile}[GTI]' outfile={outfile} maxgap={maxgap}"
        run(RUN_GTIMERGE.format(maxgap=str(GTI_TOLERANCE), infile=tmpname, outfile=clfile))
        RUN_HKFILTER="export HEADASPROMPT=/dev/null;fcopy infile='{infile}[HK][TIME.ge.{start}&&TIME.le.{stop}]' outfile='!{outfile}'"
        run(RUN_HKFILTER.format(infile=clfile,start=obsonboardstart-100., stop=obsonboardstop+100.,outfile=clfile))
        run(REMOVE_TMP_FILE.format(tmpfile=tmpname))



parser = argparse.ArgumentParser()
parser.add_argument("--stem", help="ART-XC stem")
parser.add_argument("--version", help="data version", default='000')
parser.add_argument("--obsid", help="ObsID")
parser.add_argument("--obsstart", help="Observation start time, as in PZ")
parser.add_argument("--obsstop", help="Observation end time, as in PZ")
parser.add_argument("--ra", help="Source RA")
parser.add_argument("--dec", help="Source DEC")
parser.add_argument("--modules", help="Which modules to use 1-use,0-do not use, default=1111111", default='1111111')
args = parser.parse_args()
    



select_data_from_L0_STRICT_POINTING(args.stem, args.subversion, args.obsstart, args.obsstop, float(args.ra), float(args.dec), args.obsid, args.modules)



