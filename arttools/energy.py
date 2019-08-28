from scipy.interpolate import interp1d
import numpy as np

def PHA_to_PI(PHA, strip, Temp, caldb):
    """
    converts digital signals in strips in to pulse invariants with  formula c0_0 + c0_1*T + (c1_0 + c1_1*T)*PHA + (c2_0 + c2_1*T)*PHA*PHA
    """
    energy = caldb["c0_0"][strip] + caldb["c1_0"][strip]*PHA + caldb["c2_0"][strip]*PHA*PHA + \
            (caldb["c0_1"][strip] + caldb["c1_1"][strip]*PHA + caldb["c2_1"][strip]*PHA*PHA)*Temp
    return energy

def random_uniform(PHA, strip, Temp, caldb):
    e1 = PHA_to_PI(PHA - 0.5, strip, Temp, caldb)
    e2 = PHA_to_PI(PHA + 0.5, strip, Temp, caldb)
    return np.random.uniform(e1, e2)

def filter_edge_strips(events):
    return np.all((events["RAW_X"] > 0, events["RAW_X"] < 47, events["RAW_Y"] > 0, events["RAW_Y"] < 47), axis=0)

def mkeventindex(strip):
    coord = np.empty((3, strip.size), np.int) #(strip, (3, 1))
    coord[0, :] = strip - 1
    coord[1, :] = strip
    coord[2, :] = strip + 1
    mask = (coord >= 0) & (coord < 48)
    coord[np.logical_not(mask)] = -1
    return coord, mask
    

def get_events_energy(eventlist, hkdata, caldb):
    """
    from the captured event, which described with 6 16bit int numbers (left,right,central; top and bot digital amplitudes), 
    and 6 double values for voltages and temperatures, using calibration data one have to restore the energy of the photon, 
    which resulted the triger and stored event

    The algorithm is caused by the design of the detector and was long tested in on-ground test with 29 urd,
    see this url:

    the energy reconstruction is done in following steps:

    (step 1, 2 and 3 equivalent for bottom and top strips)
    1. with the stored in caldb threashold for the digital signal, define wether to consider
        a signal in particular strip to be caused by the energy release, or by the stochastick signal variation around ground level.
    2. for the strips, which passed previous test, estimate energy in keV, captured by particular strip and sum them
    3. evaluate assumed dispersion of the signal in each strip 
        this step is performed with the help of two float values:
        FWHM - noise level for the particular strip measured for T = 0 C 
        and FWHM_T_deg - a linear coefficient, defining additional noise component 
        FWHM(T) = FWHM + FWHM_T_def*T1
    4. estimate weighted energy of the photon using "independent" energy estimation in the top and bottom strips.
    whoala, you got your energy dude
    """
    print("total events", eventlist.size)

    #eventlist = eventlist[filter_edge_strips(eventlist)] 

    T = interp1d(hkdata["TIME"], hkdata["TD1"], 
            bounds_error=False, kind="linear", 
            fill_value=(hkdata["TD1"][0], hkdata["TD1"][-1]))(eventlist["TIME"])

    botcal = caldb["BOT"].data
    topcal = caldb["TOP"].data

    rawx, maskb = mkeventindex(eventlist["RAW_X"])
    #rawx = np.tile(eventlist["RAW_X"], (3, 1))
    #maskb = np.ones(rawx.shape, np.bool)
    print(rawx.shape, maskb.size, maskb.sum())
    sigmab = botcal["fwhm_1"][rawx]*T + botcal["fwhm_0"][rawx]
    PHAB = np.array([eventlist["PHA_BOT_SUB1"], eventlist["PHA_BOT"], eventlist["PHA_BOT_ADD1"]])
    energb = random_uniform(PHAB, rawx, T, botcal) #PHA_to_PI(PHAB, rawx, T, botcal)
    maskb = np.logical_and(maskb, energb > botcal["THRESHOLD"][rawx])
    print("tot size", maskb.shape[1])
    print("bot dist", np.unique(maskb.sum(axis=0), return_counts=True))

    rawy, maskt = mkeventindex(eventlist["RAW_Y"])
    #rawy = np.tile(eventlist["RAW_X"], (3, 1))
    #maskt = np.ones(rawy.shape, np.bool)
    sigmat = topcal["fwhm_1"][rawy]*T + topcal["fwhm_0"][rawy]
    PHAT = np.array([eventlist["PHA_TOP_SUB1"], eventlist["PHA_TOP"], eventlist["PHA_TOP_ADD1"]])
    energt = random_uniform(PHAT, rawy, T, topcal) #PHA_to_PI(PHAT, rawy, T, topcal)
    maskt = np.logical_and(maskt, energt > topcal["THRESHOLD"][rawy])
    print("top dist", np.unique(maskt.sum(axis=0), return_counts=True))

    """
    drop all below threashold
    """
    atleastone = np.logical_and(np.any(maskb, axis=0), np.any(maskt, axis=0))
    #atleastone = np.logical_and(atleastone, masklf)
    print("drop by threshold", atleastone.size - atleastone.sum(), " from ", atleastone.size)
    energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt = (arr[:, atleastone] for arr in [energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt])
    print("all that left", energt.shape)
    xc = np.sum(energb*maskb*rawx, axis=0)/np.sum(maskb*energb, axis=0)
    yc = np.sum(energt*maskt*rawy, axis=0)/np.sum(maskb*energb, axis=0)
    centralzone = ((xc - 23.5)**2. + (yc - 23.5)**2.) < 25.**2.
    print("drop by shadow", centralzone.size - centralzone.sum())
    energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt = (arr[:, centralzone] for arr in [energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt])


    ebot = np.sum(energb*maskb, axis=0)
    sigmabotsq = np.sum(sigmab**2.*maskb, axis=0)
    etop = np.sum(energt*maskt, axis=0)
    sigmatopsq = np.sum(sigmat**2.*maskt, axis=0)
    emean = (ebot*sigmatopsq + etop*sigmabotsq)/(sigmatopsq + sigmabotsq)

    """
    urd 28 additional correction
    """
    emean = -0.2504 + 1.0082*emean - 6.10E-5*emean**2.
    return atleastone, emean


