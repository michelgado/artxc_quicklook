from scipy.interpolate import interp1d
import numpy as np

def PHA_to_PI(PHA, strip, Temp, caldb):
    """
    converts digital signals in strips in to pulse invariants with  formula c0_0 + c0_1*T + (c1_0 + c1_1*T)*PHA + (c2_0 + c2_1*T)*PHA*PHA
    """
    energy = caldb["c0_0"][strip] + caldb["c1_0"][strip]*PHA + caldb["c2_0"][strip]*PHA*PHA + \
            (caldb["c0_1"][strip] + caldb["c1_1"][strip]*PHA + caldb["c2_1"][strip]*PHA*PHA)*Temp
    return energy

def filter_edge_strips(events):
    return np.all((events["RAW_X"] > 0, events["RAW_X"] < 47, events["RAW_Y"] > 0, events["RAW_Y"] < 47), axis=0)

def get_events_energy(eventlist, hkdata, caldb):
    """
    from the captured event, which described with 6 16bit int numbers (left,right,central; top and bot digital amplitudes), 
    and 6 double values for voltages and temperatures, using calibration data one have to restore the energy of the photon, 
    which resulted the triger and stored event

    The algorithm is caused by the design of the detector and was long tested in on-ground test with 29 urd,
    see this url:

    the energy reconstruction is done in following steps:

    (step 1, 2 and 3 equivalent for bottom and top stips)
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

    eventlist = eventlist[filter_edge_strips(eventlist)] 

    T = interp1d(hkdata["TIME"], hkdata["TD1"], 
            bounds_error=False, 
            fill_value=(hkdata["TD1"][0], hkdata["TD1"][-1]))(eventlist["TIME"])

    botcal = caldb["BOT"].data
    topcal = caldb["TOP"].data

    rawx = np.tile(eventlist["RAW_X"], (3, 1))
    rawx[0, :] = rawx[0, :] - 1
    rawx[2, :] = rawx[2, :] + 1
    sigmab = botcal["fwhm_1"][rawx]*T + botcal["fwhm_0"][rawx]
    PHAB = np.array([eventlist["PHA_BOT_SUB1"], eventlist["PHA_BOT"], eventlist["PHA_BOT_ADD1"]])
    energb = PHA_to_PI(PHAB, rawx, T, botcal)
    maskb = energb > botcal["THRESHOLD"][rawx]

    rawy = np.tile(eventlist["RAW_Y"], (3, 1))
    rawy[0, :] = rawy[0, :] - 1
    rawy[2, :] = rawy[2, :] + 1
    sigmat = botcal["fwhm_1"][rawx]*T + botcal["fwhm_0"][rawy]
    PHAT = np.array([eventlist["PHA_TOP_SUB1"], eventlist["PHA_TOP"], eventlist["PHA_TOP_ADD1"]])
    energt = PHA_to_PI(PHAT, rawy, T, topcal)
    maskt = energt > topcal["THRESHOLD"][rawy]

    """
    drop all below threashold
    """
    atleastone = np.logical_and(np.any(maskb, axis=0), np.any(maskt, axis=0))
    print("drop", atleastone.size - atleastone.sum(), " from ", atleastone.size)
    print(atleastone.shape)
    print(energb.shape, rawx.shape, maskb.shape)
    energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt = (arr[:, atleastone] for arr in [energb, energt, sigmab, sigmat, rawx, rawy, maskb, maskt])

    ebot = np.sum(energb*maskb, axis=0)
    sigmabotsq = np.sum(energb*sigmab**maskb/ebot, axis=0)**2.

    etop = np.sum(energt*maskt, axis=0)
    sigmatopsq = np.sum(energt*sigmat**maskt/etop, axis=0)**2.

    emean = (ebot/sigmabotsq + etop/sigmatopsq)/(1./sigmatopsq + 1./sigmabotsq)
    return atleastone, emean

