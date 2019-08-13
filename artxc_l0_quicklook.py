#!/usr/bin/env python3
# -*- coding: utf8 -*-
'''
artxc_l0_quicklook

General description:
    This task produce several elementary products in order to 
    characterize the condition of ART-XC detectors and their health.

INPUT:
    L0 event files
    GYRO attitude file
    
OUTPUT:
    pdf file with report
        

TODO:
    0) integrate inside artxc pipeline structure


Stand-alone version v003

01/08/19 v001
    all hail the hypnotoad
09/08/19 v001a hart
    slightly modified output
    all hail the hypnotoad
10/08/2019 v002 hart
    added indication of low-voltage periods
    all hail the hypnotoad
12/08/2019 v003 hart
    refractored code
    now correctly working with deadtime - 
	as was explained by VVL all events that trigger detector produce deadtime, 
	even events with PHA==1023. 
	Therefore from now on we will use total number of events from telemetry in order to estimate deadtime
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

from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy.time import Time
from astropy.wcs import WCS



def  get_gti_livetime(tstart, tstop, gti_start, gti_stop):
    #calculate total amount of live time in interval from tstart to tstop
    # livetime = sum(gti)
    livetime = 0.
    gti_before = gti_start >= tstop
    gti_after  = gti_stop<tstart
    gti_mask   = np.bitwise_or(gti_before, gti_after)
    gti_mask   = np.bitwise_not(gti_mask) 
    gti_start, gti_stop =  gti_start[gti_mask], gti_stop[gti_mask]
    
    for gti_start_time, gti_stop_time in zip(gti_start, gti_stop):
        if gti_start_time<=tstart and gti_stop_time<=tstop:
            dlivetime = (gti_stop_time - tstart)
            livetime+=dlivetime
        if tstart<=gti_start_time and gti_stop_time<=tstop:
            dlivetime = (gti_stop_time - gti_start_time)
            livetime+=dlivetime
        if tstart<=gti_start_time and tstop<=gti_stop_time:
            dlivetime = (tstop - gti_start_time)
            livetime+=dlivetime
        if gti_start_time<=tstart and tstop<=gti_stop_time:
            dlivetime = (tstop - tstart)
            livetime+=dlivetime
    return livetime

def  get_deadtime_corr(tstart, tstop, livetime, all_events, mean_eff):
    # DEADTIME for ART-XC is fixed at 0.77 ms 
    if not np.isfinite(mean_eff) or mean_eff<=0:
        return 0
    DEADTIME_const = 0.77/1000. #seconds
    mask           = np.bitwise_and(all_events>=tstart, all_events<tstop)
    N_good_evts    = len(all_events[mask])

    livetime_corr       = livetime - N_good_evts*DEADTIME_const/mean_eff
    if livetime_corr == 0:
        return 0
    print ('DEADTIME correction is ',np.round(((livetime/livetime_corr)-1.)*100, 1),'%')    
    return livetime_corr
            
    

parser = argparse.ArgumentParser()
parser.add_argument("--stem", help="ART-XC stem")
parser.add_argument("--version", help="data version", default='000')
parser.add_argument("--interactive", help="show skymap?", default='no',choices=['no', 'yes'])
args = parser.parse_args()
    

stem    = args.stem
subvers = args.version

wdir      = '/srg/a1/work/oper/data/2019/'
L0        = wdir + stem+'/L0/'
L1        = wdir + stem+'/L1/'
stem_tail = '_urd.fits'
gyro_file = stem + '_'+subvers+'_gyro_att.fits'
module_names = ['02','04','08','10','20','40','80']
colorz       = ['b','b','b','b','b','b','b']
pdfname = stem + '.pdf'

wdir = '/home/hart/Current_projects/ART-XC/data/'
L0 = wdir
L1 = ''
module_names = ['02']


with PdfPages(pdfname) as pdf: 
    
    for module,col in zip(module_names,colorz):
        print ("Module "+module+" eventlist:")
        evtlist_path = L0 + stem +'_'+subvers+'.'+ module + stem_tail
        print (evtlist_path)
        evtlist = fits.open(evtlist_path)
    #   read GTI 
        gti_start   = np.array(evtlist['GTI'].data['START'])
        gti_stop    = np.array(evtlist['GTI'].data['STOP'])
        gti_total   = np.sum(gti_stop-gti_start)
        livetime    = str(np.round(gti_total,1)) 
        print('Total livetime ',np.round(gti_total,2), ' s, not DEADTIME corrected')
        
    #   read events and select 
        evtimes     = np.array(evtlist[1].data['TIME'])
        evphatop    = np.array(evtlist[1].data['PHA_TOP'])
        evphabot    = np.array(evtlist[1].data['PHA_BOT'])    
        evrawx      = np.array(evtlist[1].data['RAW_X'])
        evrawy      = np.array(evtlist[1].data['RAW_Y'])
        chanlow, chanhigh = 60, 250
        topmask  = np.bitwise_and(evphatop>=chanlow, evphatop<=chanhigh)
        botmask  = np.bitwise_and(evphabot>=chanlow, evphabot<=chanhigh)
        goodmask = np.bitwise_and(topmask, botmask)
        
        raw_evtimes = np.copy(evtimes) 
            # we have to keep number of all raw events
            # in order to correctly calculate DEADTIME   
        evtimes  = evtimes[goodmask]
        evphatop = evphatop[goodmask]
        evrawx   = evrawx[goodmask]    
        evrawy   = evrawy[goodmask]
    
    
        hk = evtlist[2].data
        t_hk, v_hk = hk['TIME'], hk['HV']
        evtcounter, proc_evtcounter = np.array(hk['EVENTS'], dtype=np.float),np.array(hk['EVENTS_PROCESSED'], dtype=np.float)

        delta_evt, delta_procevt = evtcounter[1:]-evtcounter[:-1], proc_evtcounter[1:]-proc_evtcounter[:-1]
        delta_mask = np.bitwise_and(delta_evt>0, delta_procevt>0)
        delta_rat  = delta_procevt/delta_evt
        delta_rat  = delta_rat[delta_mask]
        delta_time = t_hk[1:][delta_mask]
#        plt.figure()
#        plt.plot(t_hk[1:][delta_mask], delta_rat, 'b.')

        hv_mask    = v_hk>-95.
        t_hk       = t_hk[hv_mask]


#        for lowv_t in t_hk:
#            plt.axvspan(lowv_t-5., lowv_t+5., color='r', alpha=0.3)
        plt.show()


        
        

    
    
    
        #Plot raw detector map
        rawbins = np.linspace(0,48,49)
        fig = plt.figure(figsize=(9,9))
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005
        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]
        ax_scatter = plt.axes(rect_scatter)
        ax_scatter.hist2d(evrawx, evrawy, bins=[rawbins,rawbins],cmap='magma',norm=mc.LogNorm())
        ax_scatter.set_xlabel('RAWX')
        ax_scatter.set_ylabel('RAWY')
        ax_histx = plt.axes(rect_histx)
        ax_histx.tick_params(direction='in', labelbottom=False)
        histx, tmpbinz, tmppatches = ax_histx.hist(evrawx, bins=rawbins, histtype='step', color='b', lw=2., log=True)
        ax_histx.set_xlim((0, 48))
        ax_histx.set_ylim((0.8*np.min(histx), 1.2*np.max(histx)))
        ax_histx.set_ylabel("Events", size=12)
        fig.suptitle('Detector plane image, module '+module+', channels:'+str(int(chanlow))+','+str(int(chanhigh))+', total livetime='+livetime+'s', size=14)
        ax_histy = plt.axes(rect_histy)
        ax_histy.tick_params(direction='in', labelleft=False)
        histy, tmpbinz, tmppatches = ax_histy.hist(evrawy, bins=rawbins, histtype='step', color='b', lw=2., log=True , orientation='horizontal')
        ax_histy.set_ylim((0, 48))
        ax_histy.set_xlim((0.8*np.min(histy), 1.2*np.max(histy)))
        ax_histy.set_xlabel("Events", size=12)
        pdf.savefig() 
        plt.close()

        #Plot lightcurve 
#        print('Using only ',chanlow,'-',chanhigh,' channels')
#        print('selected: ',len(evtimes), ' events')
        mean_rate = len(evtimes)/gti_total    
#        print('Mean background countrate: ', np.round(mean_rate,2), 'cts/s')
    
        from matplotlib.ticker import ScalarFormatter
        timebin = 100.
        starttime, endtime = gti_start[0],gti_stop[-1]
        c_time = starttime
        times, times_err, rates, rates_err = [],[],[],[]
        active_pix = 48**2
        while c_time+timebin < endtime:
            n_photons = len(evtimes[np.bitwise_and(evtimes>=c_time, evtimes<(c_time + timebin))])
            livetime = get_gti_livetime(c_time, c_time+timebin, gti_start, gti_stop)
            delta_mean = np.mean(delta_rat[np.bitwise_and(delta_time>=c_time, delta_time<(c_time + timebin))]) 
            livetime = get_deadtime_corr(c_time, c_time+timebin, livetime, raw_evtimes,delta_mean)
            if livetime == 0 or livetime/timebin<0.1:
                c_time+=timebin
                continue
            print ('From ',c_time,' to ',c_time+timebin,' selected ',n_photons,' in livetime of ',livetime)
            print ('Rate is ',np.round(n_photons/livetime,2 ),' cts')
            times.append(c_time+0.5*timebin)
            times_err.append(0.5*timebin)
            rates.append(n_photons/(livetime))
            rates_err.append(np.sqrt(n_photons)/(livetime))
            c_time+=timebin

        n_photons = len(evtimes[np.bitwise_and(evtimes>=c_time, evtimes<(endtime))])
        delta_mean = np.mean(delta_rat[np.bitwise_and(delta_time>=c_time, delta_time<endtime)]) 
        livetime = get_gti_livetime(c_time, endtime, gti_start, gti_stop)
        livetime = get_deadtime_corr(c_time, endtime, livetime, raw_evtimes, delta_mean)
        if livetime>0:
            print ('From ',c_time,' to ',endtime,' selected ',n_photons,' in livetime of ',livetime)
            print ('Rate is ',np.round(n_photons/livetime,2 ),' cts')
            times.append((endtime+c_time)/2)
            times_err.append((endtime-c_time)/2)
            rates.append(n_photons/livetime)
            rates_err.append(np.sqrt(n_photons)/livetime)

        mjdref = float(evtlist['EVENTS'].header['MJDREF'])
        mjdstart, mjdstop = mjdref + gti_start[0]/86400 , mjdref + gti_stop[-1]/86400  
        isosta, isosto = Time(mjdstart, format='mjd').iso, Time(mjdstop, format='mjd').iso

        plt.figure(figsize=(9,7))
        plt.title('All-detector lightcurve, module '+module+', channels:'+str(int(chanlow))+','+str(int(chanhigh))+'\n live/dead time corrected', size=14)
        plt.errorbar(times, rates, yerr=rates_err, xerr=times_err, color=col)
        plt.axvline(gti_start[0], color='darkred', ls='dashed', lw=0.8)
        plt.text(gti_start[0]-2000, (np.max(rates)-np.min(rates))*0.5, str(isosta), rotation='vertical', color='darkred',verticalalignment='center')
        plt.axvline(gti_stop[-1], color='darkred', ls='dashed', lw=0.8)
        plt.text(gti_stop[-1]+100, (np.max(rates)-np.min(rates))*0.5, str(isosto), rotation='vertical', color='darkred',verticalalignment='center')
        plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        for lowv_t in t_hk:
            plt.axvspan(lowv_t-5., lowv_t+5., color='r', alpha=0.3)
        plt.ylabel('Counts/s')
        plt.xlabel('Time, s')
        plt.xlim(gti_start[0]-3000, gti_stop[-1]+2000)
        pdf.savefig() 
        plt.close()

        energybins = np.linspace(chanlow,chanhigh,chanhigh-chanlow+1)
        emeans, ewidths = (energybins[:-1]+energybins[1:])*0.5, (-energybins[:-1]+energybins[1:])*0.5  
        bkg_counts, fov_counts = [],[]
     
        fov_hist, tmpbinz = np.histogram(evphabot, bins=energybins) 
        
        plt.figure(figsize=(9, 9))
        plt.title('PHA_BOT spectra of module '+module)
        fov, em, ew = fov_hist, emeans, ewidths
        plt.errorbar(em, fov/ew, yerr=np.sqrt(fov)/ew, xerr=ew,
                     label =module, color='k', ls='',fmt='')
        plt.step(em+ew, fov/ew, color='k',lw=1.2)
        plt.legend()
        plt.xlim(chanlow,chanhigh)
        plt.xlabel('PHA_BOT')
        plt.ylabel('Counts/channel')
        pdf.savefig() 
        plt.close()
        evtlist.close()

    print ("Making GYRO maps...")
    gyro_path = L1 + gyro_file
    attfile = fits.open(gyro_path)
    time   = np.array(attfile[1].data['TIME'])
    ra     =  np.array(attfile[1].data['RA'])
    dec    =  np.array(attfile[1].data['DEC'])
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')


    time_offset = time[0]
    time = time - time_offset
    timeints  = time[1:] - time[:-1]
    meantimes = (time[1:] + time[:-1])*0.5
#    
    offsets = []
    angular_speeds = []
    good_t  = []
    deg2arcsec  =  3600.
    for (c1,c2,dt,mt) in zip(coords[:-1:60],coords[1::60], timeints[::60],meantimes[::60]):
        if dt>0.:
            offset = c1.separation(c2).degree*deg2arcsec
            offsets.append(offset)
            angular_speeds.append(offset/dt)
            good_t.append(mt)
    offsets        = np.array(offsets)
    angular_speeds = np.array(angular_speeds)
    good_t         = np.array(good_t)
    plt.figure(figsize=(10, 6))
    plt.title("GYRO: using 1 point per minute!")
    plt.xlabel('Time, s from '+str(time_offset))
    plt.ylabel('Angular speed, arcsec/s')
    plt.plot(good_t[::], angular_speeds[::], 'r.-')
    plt.yscale('log')
    plt.tight_layout()
    pdf.savefig() 
    plt.close()


    header = {
        'NAXIS': 2,
        'NAXIS1': 7600,
        'NAXIS2': 3800,
        'CRPIX1': 3800.,
        'CRPIX2': 1900.,
        'CRVAL1': 0.,
        'CRVAL2': 0.,
        'CDELT1': -0.05,
        'CDELT2': 0.05,
        'CTYPE1': 'GLON-MOL',
        'CTYPE2': 'GLAT-MOL',
        'RADESYS': 'FK5'}
    wcs  = WCS(header)
    plt.figure(figsize=(12,7))
    ax = plt.subplot(projection=wcs)
    lon = ax.coords['glon']
    lat = ax.coords['glat']
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')
    lon.set_ticks(spacing=30. * u.degree)
    lat.set_ticks(spacing=30. * u.degree)
    ax.grid(lw=1.)
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=time[0]*0.8, vmax=time[-1]*1.2)
##
    ax.coords[0].set_axislabel('Galactic Longitude')
    ax.coords[1].set_axislabel('Galactic Latitude')

    
    cprev = None   
    pointings = []
    for c1,t,angsp in zip(coords[1::60],time[1::60], angular_speeds):
        if angsp < 5.:
            if not cprev:
                cprev = c1
                time_label =  Time(mjdref + (t+time_offset)/86400, format='mjd').iso
                label = str(time_label)+' RA,DEC:'+str(np.round(c1.icrs.ra.degree,3))\
                        +','+str(np.round(c1.icrs.dec.degree,3))
                pointings.append(label)
                ax.plot_coord(c1.transform_to('galactic'), 'o', color=cmap(norm(t)),label=label)
            elif c1.separation(cprev).degree*deg2arcsec > 20.:
                cprev = c1
                time_label =  Time(mjdref + (t+time_offset)/86400, format='mjd').iso[:-4]
                label = str(time_label)+' RA,DEC:'+str(np.round(c1.icrs.ra.degree,3))\
                        +','+str(np.round(c1.icrs.dec.degree,3))
                pointings.append(label)
                ax.plot_coord(c1.transform_to('galactic'), 'o', color=cmap(norm(t)),label=label) 
            else:
                continue
#
    def add_src(sra, sdec, sname, axiz):
        srcx = SkyCoord(ra=sra*u.degree, dec=sdec*u.degree, frame='icrs')
        axiz.text(srcx.galactic.l.degree, srcx.galactic.b.degree,
                  sname, transform=axiz.get_transform('world'),ha='center')

        axiz.plot_coord(srcx, 'rx') 
        c = Circle((sra, sdec), 0.1, edgecolor='darkred', facecolor='none',
              transform=ax.get_transform('fk5'))
        axiz.add_patch(c)
        c = Circle((sra, sdec), 0.2, edgecolor='red', facecolor='none',
                   transform=axiz.get_transform('fk5'))
        axiz.add_patch(c)
        c = Circle((sra, sdec), 0.3, edgecolor='orange', facecolor='none',
                   transform=axiz.get_transform('fk5'))
        axiz.add_patch(c)
    add_src(299.5903159138620, 35.2016062534181, 'Cyg X-1', ax)
    add_src(201.36506287933, -43.01911266736, 'Cen A', ax)
    add_src(170.3128834862825, -60.6237854181203, 'Cen X-3', ax)
    add_src(34.4784,	-4.9812, 'UDS', ax)
    add_src(244.9794536441080, -15.6402833113192, 'Sco X-1', ax)
    
    if args.interactive == 'yes':     
        plt.show()
    pdf.savefig() 
    plt.close()

    print ("...finishing")

    plt.figure(figsize=(12,12))
    nlines = len(pointings)
    for label, ii in zip(pointings,np.arange(nlines)):
        plt.text(0, ii, label, size=15., ha='center')
    plt.ylim(-1, nlines+1)
    plt.xlim(-5, 5)
    plt.title('Suspected pointings starts')
    plt.tight_layout()
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    pdf.savefig() 
    plt.close()

    attfile.close()

    d = pdf.infodict()
    d['Title'] = 'Quicklook ART-XC report,'
    d['Author'] = 'hart'
    d['Subject'] = 'ART-XC quicklook data'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()
