from astropy.io import fits
import numpy as np

def get_lcurve(filepath,module):
#    """ Read eventfile, get events and corresponding GTIs"""
##    try:
    evtlist = fits.open(filepath)
##   read GTI 
    gti_start   = np.array(evtlist['GTI'].data['START'])
    gti_stop    = np.array(evtlist['GTI'].data['STOP'])
    gti_total   = np.sum(gti_stop-gti_start)
    print('Total livetime ',np.round(gti_total,2), ' s, not DEADTIME corrected')
##   read events and select 
    evtimes     = np.array(evtlist['EVENTS'].data['TIME'])
    evenergies  = np.array(evtlist['EVENTS'].data['ENERGY'])
    evntpatt    = np.array(evtlist['EVENTS'].data['GRADE'])
    evrawx      = np.array(evtlist['EVENTS'].data['RAW_X'])
    evrawy      = np.array(evtlist['EVENTS'].data['RAW_Y'])
##  filter out good inFOV events   
    Elow, Ehigh = 4., 11.
    energymask  = np.bitwise_and(evenergies>=Elow, evenergies<=Ehigh)
    pattmask    = evntpatt>=0
    goodphmask  = np.bitwise_and(energymask, pattmask)
    infov_evts  = len(evtimes[goodphmask])
    totalmask   = np.copy(goodphmask)
    print('Selected ', infov_evts, ' events with GRADE>=0 AND (4keV>=E>=11keV)')

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
