from astropy.io import fits
import numpy as np
from arttools.caldb import get_caldb
import matplotlib.pyplot as plt
from arttools.time import gti_intersection

def get_lcurve(filepath,module,teln):
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
##   read events and select 
    evtimes     = np.array(evtlist['EVENTS'].data['TIME'])
    evenergies  = np.array(evtlist['EVENTS'].data['ENERGY'])
    evntgrade   = np.array(evtlist['EVENTS'].data['GRADE'])
    evrawx      = np.array(evtlist['EVENTS'].data['RAW_X'])
    evrawy      = np.array(evtlist['EVENTS'].data['RAW_Y'])
##  filter out good inFOV events   
    Elow, Ehigh = 4., 11.
    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
    grademask   = evntgrade>=0
    goodphmask  = np.bitwise_and(energymask, grademask)
    good_evts   = evtimes[goodphmask]
    n_good_evts = len(good_evts)
    totalmask   = np.copy(goodphmask)
    print('Selected GOOD events:', n_good_evts, ' with GRADE>=0 AND (4keV>=E>=11keV)')

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
    grademask    =  evntgrade==-1
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
    delta_times = (timebins[1:]-timebins[:-1])    
    lc_good_evts, tmpedges = np.histogram(good_evts, bins = timebins)
    lc_bkg_evts, tmpedges  = np.histogram(bkg_evts, bins = timebins)
    lc_raw_evts, tmpedges  = np.histogram(evtimes, bins = timebins)

    # LIVETIME = SUM(GTI in tstart...tstop) - N_evts*ART-XC_DEADTIME/mean_effiiciency
    # where ART-XC_DEADTIME is constant 0.77 ms and mean_efficiency is ratio of counted to all events
    def calc_livetime(gti, tstart, tstop):
        lti      = gti_intersection(gti, np.array([[tstart, tstop]]))
        livetime = np.sum(lti.T[1]-lti.T[0])
        return livetime
    livetimes = np.array([ calc_livetime(gti, tstart, tstop) for tstart, tstop in zip(timebins[:-1],timebins[1:]) ])

#    plt.figure()
#    plt.errorbar(mean_times, lc_good_evts, xerr=delta_times, color='k', ls='',zorder=10)
#    plt.errorbar(mean_times, lc_bkg_evts*(infov_pix_n/oofov_pix_n), xerr=delta_times, color='r', ls='',zorder=10)
#    plt.show()

#    times, times_err, rates, rates_err = [],[],[],[]
#    while c_time+timebin < endtime:
#        n_bkg_photons = len(cl_events[np.bitwise_and(cl_events>=c_time, cl_events<(c_time + timebin))])
#        livetime = get_gti_livetime(c_time, c_time+timebin, gti_start, gti_stop)
#        livetime = get_deadtime_corr(c_time, c_time+timebin, livetime, evtimes)
#        print ('From ',c_time,' to ',c_time+timebin,' selected ',n_bkg_photons,' in livetime of ',livetime)
#        print ('Rate is ',np.round(n_bkg_photons/livetime,2 ),' cts')
#        times.append(c_time+0.5*timebin)
#        times_err.append(0.5*timebin)
#        rates.append(n_bkg_photons/(livetime*active_pix))
#        rates_err.append(np.sqrt(n_bkg_photons)/(livetime*active_pix))
#        c_time+=timebin
#
#    n_bkg_photons = len(cl_events[np.bitwise_and(cl_events>=c_time, cl_events<(endtime))])
#    livetime = get_gti_livetime(c_time, endtime, gti_start, gti_stop)
#    livetime = get_deadtime_corr(c_time, endtime, livetime, evtimes)
#    print ('From ',c_time,' to ',endtime,' selected ',n_bkg_photons,' in livetime of ',livetime)
#    print ('Rate is ',np.round(n_bkg_photons/livetime,2 ),' cts')
#    times.append((endtime+c_time)/2)
#    times_err.append((endtime-c_time)/2)
#    rates.append(n_bkg_photons/livetime)
#    rates_err.append(np.sqrt(n_bkg_photons)/livetime)
#    c_time+=timebin
#
#    evtlist.close()
#    return np.array(times),np.array(times_err),np.array(rates),np.array(rates_err)


#def  get_gti_livetime(tstart, tstop, gti_start, gti_stop):
#    #calculate total amount of live time in interval from tstart to tstop
#    # livetime = sum(gti)
#    livetime = 0.
#    gti_before = gti_start >= tstop
#    gti_after  = gti_stop<tstart
#    gti_mask   = np.bitwise_or(gti_before, gti_after)
#    gti_mask   = np.bitwise_not(gti_mask) 
#    gti_start, gti_stop =  gti_start[gti_mask], gti_stop[gti_mask]
#    for gti_start_time, gti_stop_time in zip(gti_start, gti_stop):
#        if gti_start_time<=tstart and gti_stop_time<=tstop:
#            dlivetime = (gti_stop_time - tstart)
#            livetime+=dlivetime
#        if tstart<=gti_start_time and gti_stop_time<=tstop:
#            dlivetime = (gti_stop_time - gti_start_time)
#            livetime+=dlivetime
#        if tstart<=gti_start_time and tstop<=gti_stop_time:
#            dlivetime = (tstop - gti_start_time)
#            livetime+=dlivetime
#        if gti_start_time<=tstart and tstop<=gti_stop_time:
#            dlivetime = (tstop - tstart)
#            livetime+=dlivetime
#    return livetime
#
#def  get_deadtime_corr(tstart, tstop, livetime, all_events):
#    # DEADTIME for ART-XC is fixed at 0.77 ms 
#    
#    DEADTIME_const = 0.77/1000. #seconds
#    mask           = np.bitwise_and(all_events>=tstart, all_events<tstop)
#    N_good_evts    = len(all_events[mask])
#    livetime_corr       = livetime - N_good_evts*DEADTIME_const
#    print ('DEADTIME correction is ',np.round((livetime/livetime_corr)*100, 1),'%')
#    return livetime_corr
#            
#    
#
#def get_bkg_lcurve(filepath,module):
#    """ Read eventfile, get events and corresponding GTIs"""
##    try:
#    evtlist = fits.open(filepath)
##   read GTI 
#    gti_start   = np.array(evtlist[2].data['START'])
#    gti_stop    = np.array(evtlist[2].data['STOP'])
#    gti_total   = np.sum(gti_stop-gti_start)
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
#    Elow, Ehigh = 5., 60.
#    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
#    pattmask    = np.bitwise_and(evntop==1, evnbot==1)
#    goodphmask  = np.bitwise_and(energymask, pattmask)
#    totalmask   = np.copy(goodphmask)
#
#    #Select photons outside of FOV using caldb file
#    detmask     =  get_CALDB_outfov(module)      
#    active_pix  = np.sum(detmask)
#    for rawx, rawy, i in zip(evrawx,evrawy,np.arange(len(totalmask))):
#        if detmask[rawx, rawy]==0:
#            totalmask[i] = 0
#
#    cl_events   = evtimes[totalmask]
#    total_bkg_evts = len(evtimes[totalmask])
#        
#    print('Using only ',Elow,'-',Ehigh,' keV, NTOP==NBOT==1 photons')
#    print('    outside illuminated area')
#    print('selected: ',len(cl_events), ' events from ',len(evtimes))
#    mean_rate = total_bkg_evts/gti_total    
#    print('Mean background countrate: ', np.round(mean_rate,2), 'cts/s')
#    
#    timebin = 100.
#    starttime, endtime = gti_start[0],gti_stop[-1]
#    c_time = starttime
#    times, times_err, rates, rates_err = [],[],[],[]
#    while c_time+timebin < endtime:
#        n_bkg_photons = len(cl_events[np.bitwise_and(cl_events>=c_time, cl_events<(c_time + timebin))])
#        livetime = get_gti_livetime(c_time, c_time+timebin, gti_start, gti_stop)
#        livetime = get_deadtime_corr(c_time, c_time+timebin, livetime, evtimes)
#        print ('From ',c_time,' to ',c_time+timebin,' selected ',n_bkg_photons,' in livetime of ',livetime)
#        print ('Rate is ',np.round(n_bkg_photons/livetime,2 ),' cts')
#        times.append(c_time+0.5*timebin)
#        times_err.append(0.5*timebin)
#        rates.append(n_bkg_photons/(livetime*active_pix))
#        rates_err.append(np.sqrt(n_bkg_photons)/(livetime*active_pix))
#        c_time+=timebin
#
#    n_bkg_photons = len(cl_events[np.bitwise_and(cl_events>=c_time, cl_events<(endtime))])
#    livetime = get_gti_livetime(c_time, endtime, gti_start, gti_stop)
#    livetime = get_deadtime_corr(c_time, endtime, livetime, evtimes)
#    print ('From ',c_time,' to ',endtime,' selected ',n_bkg_photons,' in livetime of ',livetime)
#    print ('Rate is ',np.round(n_bkg_photons/livetime,2 ),' cts')
#    times.append((endtime+c_time)/2)
#    times_err.append((endtime-c_time)/2)
#    rates.append(n_bkg_photons/livetime)
#    rates_err.append(np.sqrt(n_bkg_photons)/livetime)
#    c_time+=timebin
#
#    evtlist.close()
#    return np.array(times),np.array(times_err),np.array(rates),np.array(rates_err)
#
