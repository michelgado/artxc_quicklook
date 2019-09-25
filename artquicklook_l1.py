#!/usr/bin/env python3
# -*- coding: utf8 -*-
'''

                                  _.---"'"""""'`--.._
                             _,.-'                   `-._
                         _,."                            -.
                     .-""   ___...---------.._             `.
                     `---'""                  `-.            `.
                                                 `.            \
                                                   `.           \
                                                     \           \
                                                      .           \
   QUICKLOOK?                                         |            .
                                                      |            |
                                _________             |            |
                          _,.-'"         `"'-.._      :            |
                      _,-'                      `-._.'             |
                   _.'                              `.             '
        _.-.    _,+......__                           `.          .
      .'    `-"'           `"-.,-""--._                 \        /
     /    ,'                  |    __  \                 \      /
    `   ..                       +"  )  \                 \    /
     `.'  \          ,-"`-..    |       |                  \  /
      / " |        .'       \   '.    _.'                   .'
     |,.."--"""--..|    "    |    `""`.                     |
   ,"               `-._     |        |                     |
 .'                     `-._+         |                     |
/                           `.                        /     |
|    `     '                  |                      /      |
`-.....--.__                  |              |      /       |
   `./ "| / `-.........--.-   '              |    ,'        '
     /| ||        `.'  ,'   .'               |_,-+         /
    / ' '.`.        _,'   ,'     `.          |   '   _,.. /
   /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
  /... _.`:.________,.'              `._,.-..|        "'
 `.__.'                                 `._  /
                                           "' mh

picture from: https://www.fiikus.net/asciiart/pokemon/079.txt



artquicklook_l1

General description:
    This task produce several elementary products in order to 
    characterize the condition of ART-XC detectors, their health,
    and also the quality of data
    

    1) it generates lightcurves from non-illuminated parts of ART-XC detectors 
        in order to provide estimate of particle background.

    2) plot histograms of detector count, so operator
        can easily detect 'hot' or 'dead' strips
        and detector spectrums

    3) plot dependence of angular speed vs time


INPUT:
    cleaned event files
    attitude file
    CALDB files
    
OUTPUT:
    pdf file with report
    
AIMS:
    1) 


ALGORITHM:
    for each module:
        select photons with:
            E in 4-11 keV range
            RAWX,RAWY outside illuminated area
            NTOP==NBOT==1
        for dt in observation:
            calculate total livetime
            correct for deadtime
            calculate background rate
    plot lightcurve

    for each module:
        select photons with:
            E in 5-30 keV range
            NTOP==NBOT==1
    plot detector histogram
    




17/09/19 v001
    initial version derived from old artxc_l0_quicklook
    all hail the hypnotoad
'''

import os, shutil, argparse
from astropy.io import fits
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.colors
from arttools.caldb import get_shadowmask
from sys import exit
import os.path
import arttools.quicktools as artql
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--stem", help="ART-XC stem")
parser.add_argument("--version", help="data version", default='000')
args = parser.parse_args()
    

stem    = args.stem
if stem==None or not stem:
    print ('>>ERROR>> Please, provide stem')
    exit(1)
else:
    print ('>>>>>>>>> stem:'+stem)
subvers = args.version
if subvers==None or not subvers:
    print ('>>>>>>>>>  using subversion 000')
else:
    print ('>>>>>>>>> subversion:'+subvers)

wdir      = '/srg/a1/work/oper/data/2019/'
wdir      = '/srg/a1/pub/DATA/processed/2019/'
L1        = os.path.join(wdir,stem,'L1')
L1b       = os.path.join(wdir,stem,'L1b')
stem_tail = '_urd.fits'
gyro_file = stem + '_'+subvers+'_gyro_att.fits'
module_names   = ['02','04','08','10','20','40','80']
tel_names = ['T1','T2','T3','T4','T5','T6','T7']
module_color   = ['k','r','g','b','m','c','lime']
pdfname = stem + '.pdf'
#pdffile =  PdfPages(pdfname)

with PdfPages(pdfname) as pdffile: 
    for module,teln,modc in zip(module_names[:1],tel_names,module_color):
        print ('>>>>>>>>> Working with module '+ module)
        evtfile = stem +'_'+subvers+'.'+ module + '_urd.fits'
        evtpath = os.path.join(L1b,evtfile)
        try:
            evtfits = fits.open(evtpath)
            evtfits.close()
        except:
            print ('>>ERROR>> Cannot open '+evtpath)
        evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti = artql.get_cl_events(evtpath,module,teln)
        fov_hist,bkg_hist,emeans, ewidths = artql.get_spectrum(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, evtpath,module,teln, pdffile)
        cleanmask = np.bitwise_and(np.bitwise_and(evgrade>=0,evgrade<=8),np.bitwise_and(evenergies>=4,evenergies<=11.))
        artql.get_rawmap(evrawx[cleanmask], evrawy[cleanmask], pdffile, teln)    
        artql.get_lcurve(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, evtpath,module,teln, pdffile)
    
    gyropath = os.path.join(L1,gyro_file)
    artql.get_radec(gyropath, gti, pdffile)
    d = pdffile.infodict()
    d['Title'] = 'Quicklook ART-XC report, v.1'
    d['Author'] = 'hart'
    d['Subject'] = 'ART-XC quicklook data'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()



#    plt.figure(figsize=(9, 9))
#    plt.title('Out-of-FOV countrates, 5-60 keV, single events only')
#    for module,col in zip(module_names,colorz):
#        print ("Module "+module+" eventlist:")
#        evtlist_path = stem + module + '_el.fits'
#        t, te, r, re = get_bkg_lcurve(evtlist_path, module) 
#        plt.errorbar(t, r, yerr=re, xerr=te, label =module,color=col)
#    plt.legend()
#    plt.xlabel('TIME, s')
#    plt.ylabel('Counts/s per pixel')
#    pdf.savefig() 
#    plt.close()
###    
#    for module in module_names:
#        print ("Module "+module+" eventlist:")
#        evtlist_path = stem + module + '_el.fits'
#        get_detimage(evtlist_path, module, pdf)
#        get_detspectrum(evtlist_path, module, pdf)
#        
#    
#    d = pdf.infodict()
#    d['Title'] = 'Quicklook ART-XC report'
#    d['Author'] = 'oper'
#    d['Subject'] = 'ART-XC quicklook data'
#    d['CreationDate'] = datetime.datetime.today()
#    d['ModDate'] = datetime.datetime.today()










#
#
#
#    

#def get_detimage(filepath, modname, pdf_to_write):
#    """ Read eventfile, get events and corresponding GTIs"""
##    try:
#    evtlist = fits.open(filepath)
##   read GTI 
#    gti_start   = np.array(evtlist[2].data['START'])
#    gti_stop    = np.array(evtlist[2].data['STOP'])
#    gti_total   = np.sum(gti_stop-gti_start)
#    livetime    = str(np.round(gti_total,1)) 
#    print('Total livetime ',np.round(gti_total,2), ' s, not DEADTIME corrected')
#    
##   read events and select 
#    evtimes     = np.array(evtlist[1].data['TIME'])
#    evenergies  = np.array(evtlist[1].data['ENERGY'])
#    evntop      = np.array(evtlist[1].data['NTOP'])
#    evnbot      = np.array(evtlist[1].data['NBOT'])
#    evrawx      = np.array(evtlist[1].data['RAW_X'])
#    evrawy      = np.array(evtlist[1].data['RAW_Y'])
##    
#    Elow, Ehigh = 5., 30.
#    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
#    pattmask    = np.bitwise_and(evntop==1, evnbot==1)
#    totalmask  = np.bitwise_and(energymask, pattmask)
#    cl_events   = evtimes[totalmask]
#    total_evts = len(evtimes[totalmask])        
#    print('Using only ',Elow,'-',Ehigh,' keV, NTOP==NBOT==1 photons')
#    print('selected: ',len(cl_events), ' events from ',len(evtimes))
#    mean_rate = total_evts/gti_total    
#    print('Mean countrate: ', np.round(mean_rate,2), 'cts/s')
#    
#    rawbins = np.linspace(0,48,49)
#    fig = plt.figure(figsize=(9,9))
#    left, width = 0.1, 0.65
#    bottom, height = 0.1, 0.65
#    spacing = 0.005
#    rect_scatter = [left, bottom, width, height]
#    rect_histx = [left, bottom + height + spacing, width, 0.2]
#    rect_histy = [left + width + spacing, bottom, 0.2, height]
#    ax_scatter = plt.axes(rect_scatter)
#    ax_scatter.hist2d(evrawx[totalmask], evrawy[totalmask], bins=[rawbins,rawbins],cmap='magma',norm=mc.LogNorm())
#    #overplot excluded out-of-FOV pixels
#    detmask     =  get_CALDB_outfov(modname)  
#    for rawx in np.arange(48):
#        for rawy in np.arange(48):
#            if detmask[rawx,rawy]==1:
#                ax_scatter.plot(rawx+0.5, rawy+0.5, 'gx')
#
#    ax_scatter.set_xlabel('RAWX')
#    ax_scatter.set_ylabel('RAWY')
#    ax_histx = plt.axes(rect_histx)
#    ax_histx.tick_params(direction='in', labelbottom=False)
#    histx, tmpbinz, tmppatches = ax_histx.hist(evrawx[totalmask], bins=rawbins, histtype='step', color='b', lw=2., log=True)
#    ax_histx.set_xlim((0, 48))
#    ax_histx.set_ylim((0.8*np.min(histx), 1.2*np.max(histx)))
#    ax_histx.set_ylabel("Events", size=12)
#    fig.suptitle('Detector plane image, '+modname+', 5-30 keV, single events only, total livetime='+livetime+'s', size=14)
#
#
#    ax_histy = plt.axes(rect_histy)
#    ax_histy.tick_params(direction='in', labelleft=False)
#    histy, tmpbinz, tmppatches = ax_histy.hist(evrawy[totalmask], bins=rawbins, histtype='step', color='b', lw=2., log=True , orientation='horizontal')
#    ax_histy.set_ylim((0, 48))
#    ax_histy.set_xlim((0.8*np.min(histy), 1.2*np.max(histy)))
#    ax_histy.set_xlabel("Events", size=12)
#
#    pdf_to_write.savefig(fig)
#    plt.close()
#    
#    evtlist.close()
#    
#def get_detspectrum(filepath, modname, pdf_to_write):
#    """ Read eventfile, get events and corresponding GTIs"""
##    try:
#    evtlist = fits.open(filepath)
##   read GTI 
#    gti_start   = np.array(evtlist[2].data['START'])
#    gti_stop    = np.array(evtlist[2].data['STOP'])
#    gti_total   = np.sum(gti_stop-gti_start)
#    livetime    = str(np.round(gti_total,1)) 
#    print('Total livetime ',np.round(gti_total,2), ' s, not DEADTIME corrected')
#    
##   read events and select 
#    evtimes     = np.array(evtlist[1].data['TIME'])
#    evenergies  = np.array(evtlist[1].data['ENERGY'])
#    evntop      = np.array(evtlist[1].data['NTOP'])
#    evnbot      = np.array(evtlist[1].data['NBOT'])
#    evrawx      = np.array(evtlist[1].data['RAW_X'])
#    evrawy      = np.array(evtlist[1].data['RAW_Y'])
##    
#    Elow, Ehigh = 1., 100.
#    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
#    pattmask    = np.bitwise_and(evntop==1, evnbot==1)
#    totalmask   = np.bitwise_and(energymask, pattmask)    
#    cl_pi       = evenergies[totalmask]
#    total_evts = len(cl_pi)        
#    print('Using only ',Elow,'-',Ehigh,' keV, NTOP==NBOT==1 photons')
#    print('selected: ',len(cl_pi), ' events from ',len(evtimes))
#    mean_rate = total_evts/gti_total    
#    print('Mean countrate: ', np.round(mean_rate,2), 'cts/s')
#    
#    energybins = np.logspace(0,2,64)
#    emeans, ewidths = (energybins[:-1]+energybins[1:])*0.5, (-energybins[:-1]+energybins[1:])*0.5  
#    bkg_counts, fov_counts = [],[]
# 
#    detmask     =  get_CALDB_outfov(modname)  
#    bkg_pix  = np.sum(detmask)
#    fov_pix  = 48**2 - bkg_pix
#    
#    for rawx, rawy, energy in zip(evrawx[totalmask],evrawy[totalmask],cl_pi):
#        if detmask[rawx, rawy]==0:
#            fov_counts.append(energy)
#        else:
#            bkg_counts.append(energy)
#    fov_hist, tmpbinz = np.histogram(fov_counts, bins=energybins) 
#    bkg_hist, tmpbinz = np.histogram(bkg_counts, bins=energybins)
#    bkg_hist = bkg_hist*(fov_pix/bkg_pix)
#    evtlist.close()
#    
#    
#    plt.figure(figsize=(9, 9))
#    plt.title('Single event spectra of '+modname)
#    fov, bkg, em, ew = fov_hist, bkg_hist,emeans, ewidths
#    plt.errorbar(em, fov/ew, yerr=np.sqrt(fov)/ew, xerr=ew,
#                 label =modname, color='k', ls='',fmt='')
#    plt.step(em+ew, fov/ew, color='k',lw=1.2)
#    plt.errorbar(em, bkg/ew, yerr=np.sqrt(bkg)/ew, xerr=ew,
#                 color='r', ls='',fmt='',alpha=0.5,label=modname+' bkg')
#    plt.step(em+ew, bkg/ew, color='r',lw=1.2, ls=':',alpha=0.5)
#    plt.loglog()    
#    plt.legend()
#    plt.xlabel('Energy, keV')
#    plt.ylabel('Counts/keV')
#    pdf_to_write.savefig() 
#    plt.close()
#    
#    return None
#
#
#def get_angspeed(filepath, pdf):
#    """ Read attitude file and plot angular speed vs time"""
#    from astropy import units as u
#    from astropy.coordinates import SkyCoord, Galactic
#    
##    try:
#    attfile = fits.open(filepath)
##   read GTI 
#    time   = np.array(attfile[1].data['TIME'])
#    ra     =  np.array(attfile[1].data['RA'])
#    dec    =  np.array(attfile[1].data['DEC'])
#    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
#    
#    
#    time_offset = time[0]
#    time = time - time_offset
#    timeints  = time[1:] - time[:-1]
#    meantimes = (time[1:] + time[:-1])*0.5
#    
#    offsets = []
#    angular_speeds = []
#    good_t  = []
#    deg2arcsec  =  3600.
#    for (c1,c2,dt,mt) in zip(coords[:-1],coords[1:], timeints,meantimes):
#        offset = c1.separation(c2).degree*deg2arcsec
#        offsets.append(offset)
#        angular_speeds.append(offset/dt)
#        good_t.append(mt)
#    offsets = np.array(offsets)
#    angular_speeds = np.array(angular_speeds)
#    plt.figure(figsize=(12, 6))
#    plt.xlabel('Time, s from '+str(time_offset))
#    plt.ylabel('Angular speed, arcsec/s')
#    plt.plot(good_t, angular_speeds, 'r.-')
#    plt.tight_layout()
#    pdf.savefig() 
#    plt.close()
#
#
##    plt.figure(figsize=(12, 6), dpi=100)
##    plt.title('SRG axis location in equitorial coordinates')
##    ax = plt.axes(projection='astro degrees aitoff')
##    cmap = plt.cm.rainbow
##    norm = matplotlib.colors.Normalize(vmin=time[0]*0.8, vmax=time[-1]*1.2)
##
##    ax.grid()
##    current_c = SkyCoord(ra=ra[0]*u.degree, dec=dec[0]*u.degree, frame='icrs')
##    ax.plot_coord(current_c, 'o', color=cmap(norm(time[0])))
##    for c1,t in zip(coords[1:],time[1:]):
##        if c1.separation(current_c).degree*deg2arcsec > 10.:
##            current_c = c1
##            ax.plot_coord(current_c, 'o', color=cmap(norm(t)))
##    plt.show()
##    pdf.savefig() 
###    
#    return None
#
#    
#    
#import datetime
#module_names = ['T1','T2','T3','T4','T5','T6','T7']
#colorz       = ['k','r','g','b','m','c','lime']
#pdfname = 'artxc_quicklook_report_now.pdf'
#with PdfPages(pdfname) as pdf:
#    plt.figure(figsize=(9, 9))
#    plt.title('Out-of-FOV countrates, 5-60 keV, single events only')
#    for module,col in zip(module_names,colorz):
#        print ("Module "+module+" eventlist:")
#        evtlist_path = stem + module + '_el.fits'
#        t, te, r, re = get_bkg_lcurve(evtlist_path, module) 
#        plt.errorbar(t, r, yerr=re, xerr=te, label =module,color=col)
#    plt.legend()
#    plt.xlabel('TIME, s')
#    plt.ylabel('Counts/s per pixel')
#    pdf.savefig() 
#    plt.close()
###    
#    for module in module_names:
#        print ("Module "+module+" eventlist:")
#        evtlist_path = stem + module + '_el.fits'
#        get_detimage(evtlist_path, module, pdf)
#        get_detspectrum(evtlist_path, module, pdf)
#        
#    
#    d = pdf.infodict()
#    d['Title'] = 'Quicklook ART-XC report'
#    d['Author'] = 'oper'
#    d['Subject'] = 'ART-XC quicklook data'
#    d['CreationDate'] = datetime.datetime.today()
#    d['ModDate'] = datetime.datetime.today()
