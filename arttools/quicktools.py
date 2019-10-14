from astropy.io import fits
import numpy as np
from arttools.caldb import get_caldb
import matplotlib.pyplot as plt
from arttools.time import gti_intersection
from arttools.mask import edges
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as mc
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic
from astropy.time import Time
from astropy.wcs import WCS
import subprocess

def get_hk_gti(time, voltage, gti):
    '''
    input: arrays of timestamps and voltages from HK and GTIs
    '''
    t_hk, v_hk = time, voltage
    hv_mask    = v_hk<-95.
    t_indexes  = edges(hv_mask)
    t_indexes[1::2]-= 1
    if t_indexes[-1][-1] == len(t_hk):
        t_indexes[-1][-1]-=1
    hk_gti     = np.reshape(np.take(t_hk, np.ravel(t_indexes)), (-1,2))        
    return gti_intersection(gti, hk_gti)

def ingti(gti, t):
    return np.any(np.bitwise_and((t >= gti[:,0]),(t <= gti[:,1])))

def gtifilter(events_times, gti):
    mask = np.array([ingti(gti, t) for t in events_times])    
    return events_times[mask], mask

def get_median_processed_evt_ratio(evtcounter, proc_evtcounter, t_hk):
    delta_evt, delta_procevt = evtcounter[1:]-evtcounter[:-1], proc_evtcounter[1:]-proc_evtcounter[:-1]
    delta_mask = np.bitwise_and(delta_evt>0, delta_procevt>0)
    delta_rat  = delta_procevt/delta_evt
    delta_rat  = delta_rat[delta_mask]
#    t_delta    = 0.5*(t_hk[1:]+t_hk[:-1])
    print ('Median processed-to-all-events ratio is ',np.median(delta_rat),' with std of',np.std(delta_rat))
    return np.median(delta_rat)


def time_to_board(time):
    REFTIME = Time(51543.875, format='mjd')
    return (time - REFTIME).sec
def boardtime_to_date(seconds):
    return (Time(51543.875, format='mjd') + seconds*u.s).iso


def get_cl_events(filepath,module,teln):
#    """ Read eventfile, get events and corresponding GTIs"""
##    try:
    evtlist = fits.open(filepath)
##   read GTI 
    gti_start   = np.array(evtlist['GTI'].data['START'])
    gti_stop    = np.array(evtlist['GTI'].data['STOP'])
    gti_total   = np.sum(gti_stop-gti_start)
    gti         = np.array([gti_start,gti_stop]).T
    print (gti.shape, gti[0]) 
    print('Total livetime ',np.round(gti_total,2), ' s, not DEADTIME corrected')
    t_hk, v_hk = evtlist['HK'].data['TIME'], evtlist['HK'].data['HV']
#    evtcounter, proc_evtcounter = np.array(evtlist['HK'].data['EVENTS'], dtype=np.float),\
#                                  np.array(evtlist['HK'].data['EVENTS_PROCESSED'], dtype=np.float)
#    median_ratio = get_median_processed_evt_ratio(evtcounter, proc_evtcounter, t_hk)
    print ('Making new GTI with HK data..')    
    newgti = get_hk_gti(t_hk,v_hk, gti)


##   read events and select 
    evtimes              = np.array(evtlist['EVENTS'].data['TIME'])
    evtimes, gtimask     = gtifilter(evtimes, newgti)
    evenergies  = np.array(evtlist['EVENTS'].data['ENERGY'])[gtimask]
    evgrade     = np.array(evtlist['EVENTS'].data['GRADE'])[gtimask]
    evrawx      = np.array(evtlist['EVENTS'].data['RAW_X'])[gtimask]
    evrawy      = np.array(evtlist['EVENTS'].data['RAW_Y'])[gtimask]
    evflag      = np.array(evtlist['EVENTS'].data['FLAG'])[gtimask]
    evtlist.close()
    return evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti


def get_spectrum(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, filepath, module, teln, pdf):
    gti_total = np.sum(gti[:,-1]-gti[:,0])
    gti_start = gti[:,0]
    gti_stop = gti[:,-1]
    #Read CALDB entry with OOF mask
    oofmask = fits.getdata(get_caldb('OOFPIX', teln),1)
    #Exclude boundary strips from mask, since they are noisy and will introduce additional noise in background
    oofmask[0,:]  = 1
    oofmask[-1,:] = 1
    oofmask[:,0]  = 1
    oofmask[:,-1] = 1
    infov_pix_n   = 48**2 - np.sum(oofmask)
    oofmask[0,:]  = 0
    oofmask[-1,:] = 0
    oofmask[:,0]  = 0
    oofmask[:,-1] = 0
    oofov_pix_n   = np.sum(oofmask)
##  filter out good inFOV events   
    grademask   = np.bitwise_and(evgrade>=0,evgrade<=8)
    grademask   = np.bitwise_and(grademask, evflag==0)
    good_evts   = evenergies[grademask]
    n_good_evts = len(good_evts)

    # Now all pthotons with GRADE==-1 and RAWX,Y in 1..46 are inside detector ears
    grademask   = np.bitwise_and(evgrade>=0,evgrade<=8)
    grademask   = np.bitwise_and(grademask, evflag==2)
    rawxmask     =  np.bitwise_and(evrawx>0, evrawx<47)
    rawymask     =  np.bitwise_and(evrawy>0, evrawy<47)
    totalmask    =  np.bitwise_and(grademask,np.bitwise_and(rawxmask,rawymask))
    bkg_evts     =  evenergies[totalmask]

    energybins = np.logspace(np.log10(4),2,81)
    emeans, ewidths = (energybins[:-1]+energybins[1:])*0.5, (-energybins[:-1]+energybins[1:])*0.5  
    fov_hist, tmpbinz = np.histogram(good_evts, bins=energybins) 
    bkg_hist, tmpbinz = np.histogram(bkg_evts, bins=energybins)
    bkg_hist = bkg_hist*(np.sum(fov_hist[50:])/np.sum(bkg_hist[50:]))
    fov, bkg, em, ew = fov_hist, bkg_hist, emeans, ewidths
    plt.figure(figsize=(9, 9))
    plt.title('Spectra of single and double events for '+teln)
    fov, bkg, em, ew = fov_hist, bkg_hist, emeans, ewidths
    plt.errorbar(em, fov/ew, yerr=np.sqrt(fov)/ew, xerr=ew,
                 label =teln+' FOV', color='k', ls='',fmt='')
    plt.step(em+ew, fov/ew, color='k',lw=1.2)
    plt.errorbar(em, bkg/ew, yerr=np.sqrt(bkg)/ew, xerr=ew,
                 color='r', ls='',fmt='',alpha=0.5,label=teln+' normalized background')
    plt.step(em+ew, bkg/ew, color='r',lw=1.2, ls=':',alpha=0.5)
    plt.loglog()
    plt.xlim(4,40)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))    
    plt.legend()
    plt.xlabel('Energy, keV')
    plt.ylabel('Counts/keV')
    pdf.savefig() 
    plt.close()
    return fov_hist,bkg_hist,emeans, ewidths 



def get_lcurve(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, filepath,module,teln, pdf):
    gti_total = np.sum(gti[:,-1]-gti[:,0])
    gti_start = gti[:,0]
    gti_stop = gti[:,-1]

#    try:
#        pzfile = fits.open('pz.fits')
#        obsid, obsstart, obsstop = pzfile[1]['EXPERIMENT'],pzfile[1]['START'],pzfile[1]['STOP']
        

##  filter out good inFOV events   
    Elow, Ehigh = 4., 11.
    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
    grademask   = np.bitwise_and(evgrade>=0,evgrade<=8)
    grademask   = np.bitwise_and(grademask, evflag==0)
    goodphmask  = np.bitwise_and(energymask, grademask)
    good_evts   = evtimes[goodphmask]
    n_good_evts = len(good_evts)
    totalmask   = np.copy(goodphmask)
    print('Selected GOOD events:', n_good_evts, ' with GRADE in [0,8] AND (4keV>=E>=11keV)')

    #Read CALDB entry with OOF mask
    oofmask = fits.getdata(get_caldb('OOFPIX', teln),1)
    #Exclude boundary strips from mask, since they are noisy and will introduce additional noise in background
    oofmask[0,:]  = 1
    oofmask[-1,:] = 1
    oofmask[:,0]  = 1
    oofmask[:,-1] = 1
    infov_pix_n   = np.sum(np.logical_not(oofmask).astype(np.bool))
    oofmask[0,:]  = 0
    oofmask[-1,:] = 0
    oofmask[:,0]  = 0
    oofmask[:,-1] = 0
    oofov_pix_n   = np.sum(oofmask)
    # Now all pthotons with GRADE==-1 and RAWX,Y in 1..46 are inside detector ears
    grademask   = np.bitwise_and(evgrade>=0,evgrade<=8)
    grademask   = np.bitwise_and(grademask, evflag==2)
    energymask   =  np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
    rawxmask     =  np.bitwise_and(evrawx>0, evrawx<47)
    rawymask     =  np.bitwise_and(evrawy>0, evrawy<47)
    totalmask    =  np.bitwise_and(np.bitwise_and(grademask,energymask),np.bitwise_and(rawxmask,rawymask))
    bkg_evts     =  evtimes[totalmask]
    n_bkg_evts = len(evtimes[totalmask])
    print('Selected BKG events:', n_bkg_evts, ' with (4keV>=E>=11keV)')

        
    mean_good_rate = n_good_evts/gti_total    
    mean_bkg_rate  = (n_bkg_evts*infov_pix_n/oofov_pix_n)/gti_total    

    print('Mean GOOD countrate: ', np.round(mean_good_rate,3), 'cts/s')
    print('Mean normalized BKG countrate: ', np.round(mean_bkg_rate,3), 'cts/s')

    #Make detector lightcurve correcting for DEADTIME of known and unknown events    
    timebin = 20. #seconds. This binsize should allow for individual bright sources to be visible during survey  
    starttime, endtime = gti_start[0],gti_stop[-1]
    timebins  = np.arange(starttime, endtime,timebin)
    if timebins[-1]!=endtime:
        timebins = np.concatenate((timebins, [endtime]))
    mean_times  = (timebins[:-1] + timebins[1:])*0.5    
    delta_times = (timebins[1:]-timebins[:-1])*0.5    
    lc_good_evts, tmpedges = np.histogram(good_evts, bins = timebins)
    lc_bkg_evts, tmpedges  = np.histogram(bkg_evts, bins = timebins)
    lc_raw_evts, tmpedges  = np.histogram(evtimes, bins = timebins)

    # LIVETIME = SUM(GTI in tstart...tstop) - N_evts*ART-XC_DEADTIME/mean_effiiciency
    # where ART-XC_DEADTIME is constant 0.77 ms and mean_efficiency is ratio of counted to all events
    DEADTIME = 0.77/1000. #s
    def calc_livetime(gti, tstart, tstop):
        lti      = gti_intersection(gti, np.array([[tstart, tstop]]))
        livetime = np.sum(lti.T[1]-lti.T[0])
        return livetime
    def get_mean_processed_evt_ratio(evtcounter, proc_evtcounter, t_hk,tstart, tstop):
        timemask = np.bitwise_and(t_hk>=tstart,t_hk<=tstop)
        evtcounter, proc_evtcounter = evtcounter[timemask], proc_evtcounter[timemask]
        delta_evt, delta_procevt = evtcounter[1:]-evtcounter[:-1], proc_evtcounter[1:]-proc_evtcounter[:-1]
        delta_mask = np.bitwise_and(delta_evt>0, delta_procevt>0)
        delta_rat  = delta_procevt/delta_evt
        delta_rat  = delta_rat[delta_mask]
        return np.mean(delta_rat)
    evtlist = fits.open(filepath)
    t_hk = evtlist['HK'].data['TIME']
    evtcounter, proc_evtcounter = np.array(evtlist['HK'].data['EVENTS'], dtype=np.float),\
                                  np.array(evtlist['HK'].data['EVENTS_PROCESSED'], dtype=np.float)

    livetimes = np.array([ calc_livetime(gti, tstart, tstop) for tstart, tstop in zip(timebins[:-1],timebins[1:]) ])
    mean_ratios = np.array([get_mean_processed_evt_ratio(evtcounter, proc_evtcounter, t_hk, tstart, tstop) for tstart, tstop in zip(timebins[:-1],timebins[1:])])
    deadtimes = (DEADTIME*lc_raw_evts)/mean_ratios
    exposures = livetimes - deadtimes
    #Now we can calculate countrates    
    good_rate    = np.divide(lc_good_evts, exposures)
    good_rate_err= np.divide(np.sqrt(lc_good_evts), exposures) 
    isosta, isosto = boardtime_to_date(gti_start[0]), boardtime_to_date(gti_stop[-1]) 
    plt.figure(figsize=(15,6))
    plt.title('All-FOV lightcurve, '+teln+', 4-11 keV, single&double event,s live/dead time corrected', size=14)
    plt.errorbar(mean_times, good_rate, xerr=delta_times, yerr=good_rate_err, color='k',ls='')
    plt.step(mean_times+delta_times, good_rate, color='k',lw=1., alpha=0.5)
    plt.axvline(gti_start[0], color='darkred', ls='dashed', lw=0.8)
    t_offset= np.min([2000, 0.1*np.sum(gti_stop[-1]-gti_start[0])])
    plt.text(gti_start[0]-t_offset, (np.nanmax(good_rate))*0.5, str(isosta), rotation='vertical', color='darkred',verticalalignment='top', horizontalalignment='center')
    plt.axvline(gti_stop[-1], color='darkred', ls='dashed', lw=0.8)
    plt.text(gti_stop[-1]+t_offset, (np.nanmax(good_rate))*0.5, str(isosto), rotation='vertical', color='darkred',verticalalignment='top', horizontalalignment='center')
    plt.ylabel('Counts/s')
    plt.xlabel('Time, s')
    plt.xlim(gti_start[0]-t_offset-1000, gti_stop[-1]+t_offset+1000)
    plt.gca().xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    plt.yscale('log')
    pdf.savefig() 
    plt.close()





def get_rawmap(selected_rawx, selected_rawy, pdf, teln):
    rawbins = np.linspace(1,47,47)
    fig = plt.figure(figsize=(9,9))
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]
    ax_scatter = plt.axes(rect_scatter)
    ax_scatter.hist2d(selected_rawx, selected_rawy, bins=[rawbins,rawbins],cmap='magma',norm=mc.LogNorm())
    ax_scatter.set_xlabel('RAWX')
    ax_scatter.set_ylabel('RAWY')
    ax_histx = plt.axes(rect_histx)
    ax_histx.tick_params(direction='in', labelbottom=False)
    histx, tmpbinz, tmppatches = ax_histx.hist(selected_rawx, bins=rawbins, histtype='step', color='b', lw=2., log=True)
    ax_histx.set_xlim((0, 48))
    ax_histx.set_ylim((0.8*np.min(histx), 1.2*np.max(histx)))
    ax_histx.set_ylabel("Events", size=12)
    fig.suptitle('Detector plane image, '+teln+', 4-11 keV, single&double events', size=14)
    ax_histy = plt.axes(rect_histy)
    ax_histy.tick_params(direction='in', labelleft=False)
    histy, tmpbinz, tmppatches = ax_histy.hist(selected_rawy, bins=rawbins, histtype='step', color='b', lw=2., log=True , orientation='horizontal')
    ax_histy.set_ylim((0, 48))
    ax_histy.set_xlim((0.8*np.min(histy), 1.2*np.max(histy)))
    ax_histy.set_xlabel("Events", size=12)
    pdf.savefig() 
    plt.close()




def get_radec(gyropath, gti, pdf, pngname):
    gti_start = gti[:,0]
    gti_stop = gti[:,-1]
    gyro = fits.open(gyropath)
    ra, dec, time = gyro[1].data['RA'],gyro[1].data['DEC'],gyro[1].data['TIME']
    timeints  = time[1:] - time[:-1]

    plt.figure(figsize=(9, 9))
    xmin, xmax = np.max([gti_start[0]-1800, time[0]]),np.min([gti_stop[-1]+1800, time[-1]])
    figtitle = 'GYRO data'  
    if len(time[time<(gti_start[0]-86400)]):
        figtitle+='\n<!>NULL or invalid values in GYRO file!'
    if np.nanmax(timeints)>5.0:
        figtitle+='\n<!>GAPS in GYRO file!'

    time_mask = np.bitwise_and(time>= gti_start[0]-1800,time<= gti_stop[-1]+1800)
    time,ra,dec = time[time_mask], ra[time_mask], dec[time_mask]
    coords = SkyCoord(ra=ra*u.degree, dec=dec*u.degree, frame='icrs')
    timeints  = time[1:] - time[:-1]
    meantimes = (time[1:] + time[:-1])*0.5

    deg2arcsec     = 3600.
    good_times     = timeints>0.
    meantimes      = meantimes[good_times]
    timeints       = timeints[good_times]
    offsets        = coords[:-1:].separation(coords[1::])
    offsets        = offsets[good_times]*deg2arcsec
    angular_speeds = (offsets/timeints)

    plt.title(figtitle)
    plt.plot(time, ra-180., 'r.', label='RA-180 deg')
    plt.plot(time, dec, 'b.', label='DEC')
    plt.legend()
    plt.xlabel('Time, s')
    plt.ylabel('Coordinate, degree')
    plt.xlim(xmin, xmax)
    plt.savefig(pngname, dpi=400,format='png')
    plt.close()


    plt.figure(figsize=(10, 6))
    plt.title("GYRO: using 1 point per minute!")
    plt.xlabel('Time, s')
    plt.ylabel('Angular speed, arcsec/s')
    plt.plot(meantimes[::60], angular_speeds[::60], 'r.-')
    plt.yscale('log')
    plt.xlim(xmin, xmax)
    plt.tight_layout()
    pdf.savefig() 
    plt.close()

    header = {
        'NAXIS': 2,
        'NAXIS1': 720,
        'NAXIS2': 380,
        'CRPIX1': 360.,
        'CRPIX2': 180.,
        'CRVAL1': 0.,
        'CRVAL2': 0.,
        'CDELT1': -0.5,
        'CDELT2': 0.5,
        'CTYPE1': 'GLON-MOL',
        'CTYPE2': 'GLAT-MOL'}
    wcs  = WCS(header)
    #expoimg = np.zeros(720,360)

    plt.figure(figsize=(12,7))
    ax = plt.subplot(projection=wcs)
    cmap = plt.cm.rainbow
    time = time - time[0]
    norm = mc.Normalize(vmin=time[0]*0.95, vmax=time[-1]*1.05)
    for c,t in zip(coords[::100], time[::100]):
        ax.plot_coord(c.transform_to('galactic'),'.', color=cmap(norm(t))) 
    ax.plot_coord(coords[0].transform_to('galactic'),'.', color=cmap(norm(time[0])), label = 'Time from start: '+str(int(time[0]))) 
    ax.plot_coord(coords[-1].transform_to('galactic'),'.', color=cmap(norm(time[-1])), label = 'Time from start: '+str(int(time[-1]))) 

    lon = ax.coords['glon']
    lat = ax.coords['glat']
    lon.set_axislabel('Galactic Longitude')
    lat.set_axislabel('Galactic Latitude')
    lon.set_ticks(spacing=30. * u.degree)
    lat.set_ticks(spacing=30. * u.degree)
    ax.grid(lw=1.)
    ax.coords[0].set_axislabel('Galactic Longitude')
    ax.coords[1].set_axislabel('Galactic Latitude')
    plt.legend()
    pdf.savefig() 
    plt.close()




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
    
