from .caldb import get_highres_vign_model, get_optical_axis_offset_by_device, get_boresight_by_device, get_telescope_crabrates
from .atthist import make_small_steps_quats
from ._det_spatial import raw_xy_to_vec, vec_to_offset, rawxy_to_qcorr
from scipy.interpolate import interp1d
import numpy as np
from math import pi
from astropy.io import fits


urdcrates = get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}

def make_correcrted_arf(attdata, srcvec, filters, urddtc={}):
    psf = get_highres_vign_model()
    arf = fits.open("/srg/a1/work/andrey/ART-XC/ARTCALDB/artxc_arf_v000.fits")
    xoff, _ = vec_to_offset(np.array([np.ones(502), np.tan(np.arange(-50.1, 50.2, 0.2)*pi/180/60.), np.zeros(502)]).T)
    _, yoff = vec_to_offset(np.array([np.ones(502), np.zeros(502), np.tan(np.arange(-50.1, 50.2, 0.2)*pi/180/60.)[::-1]]).T)
    print(xoff.shape, yoff.shape, xoff[0])
    print(xoff, yoff)
    efa = psf[-1].data["EFFAREA"]/psf[-1].data["EFFAREA"][:, 250, 250][:, np.newaxis, np.newaxis]
    ew = 0.
    for urdn in filters:
        attloc = attdata*get_boresight_by_device(urdn)*rawxy_to_qcorr(*get_optical_axis_offset_by_device(urdn))
        ts, qval, dtn, locgti = make_small_steps_quats(attloc, gti=filters[urdn]["TIME"], timecorrection=urddtc.get(urdn, lambda x: 1.))
        xo, yo = vec_to_offset(qval.apply(srcvec, inverse=True))
        print(xo.shape, yo.shape)
        idxx = np.searchsorted(xoff, xo) - 1
        idxy = np.searchsorted(yoff, yo) - 1
        print(idxx, idxy)
        ew += np.sum(efa[:,idxx,idxy]*dtn[np.newaxis, :], axis=1)*urdcrates[urdn]/dtn.sum()

    #ew = ew/ew.sum()
    corr = interp1d(psf[-1].data["E"], ew, bounds_error=False, fill_value=1.)
    arf[1].data["SPECRESP"] = arf[1].data["SPECRESP"]*corr(np.sqrt(arf[1].data["ENERG_LO"]*arf[1].data["ENERG_HI"]))
    return arf

