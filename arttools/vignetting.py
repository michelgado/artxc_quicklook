from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd, OPAXOFFSET
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import cumtrapz
from ._det_spatial import get_shadowed_pix_mask, offset_to_raw_xy, DL, F, \
    offset_to_vec, vec_to_offset_pairs, vec_to_offset
from .telescope import URDNS
from .caldb import ARTQUATS
import numpy as np
from math import log10, pi, sin, cos

TINY = 1e-15

def make_vignetting_for_urdn(urdn, energy=7.2001, phot_index=None,
                             useshadowmask=True, ignoreedgestrips=True,
                             emin=0, emax=np.inf):
    vignfile = get_vigneting_by_urd(urdn)
    #TO DO: put max eff area in CALDB
    norm = 65.71259133631082 # = np.max(vignfile["5 arcmin PSF"].data["EFFAREA"])


    efint = interp1d(vignfile["5 arcmin PSF"].data["E"],
                     vignfile["5 arcmin PSF"].data["EFFAREA"],
                     axis=0)

    if not phot_index is None:
        s, e = np.searchsorted(vignfile["5 arcmin PSF"].data["E"], [emin, emax])
        es = np.copy(vignfile["5 arcmin PSF"].data["E"][s: e])
        vmap = np.copy(vignfile["5 arcmin PSF"].data["EFFAREA"][s:e])
        de = es[1:] - es[:-1]
        if phot_index != 1:
            vignmap = np.sum(vmap[:-1] - vmap[1:]*(es[:-1]/de)[:, np.newaxis, np.newaxis] + \
                             vmap[1:]/(2. - phot_index)*\
                             ((es[1:]**(2. - phot_index) - es[:-1]**(2. - phot_index))/de)[:, np.newaxis, np.newaxis], axis=0)*\
                            (1. - phot_index)/(es[-1]**(1. - phot_index) - es[0]**(1. - phot_index))
        else:
            vignmap = np.sum(vmap[:-1] - vmap[1:]*(es[:-1]/de)[:, np.newaxis, np.newaxis] + \
                             vmap[1:], axis=0)/np.log(es[-1]/es[0])

    else:
        vignmap = efint(energy)

    vignmap = vignmap/norm
    print("check vignetting map:", vignmap.max())

    x = np.tan(vignfile["Offset angles"].data["X"]*pi/180/60.)*F + (24. - OPAXOFFSET[urdn][0])*DL
    y = np.tan(vignfile["Offset angles"].data["Y"]*pi/180/60.)*F + (24. - OPAXOFFSET[urdn][1])*DL

    shmask = get_shadowmask_by_urd(urdn).astype(np.uint8) if useshadowmask else np.ones((48, 48), np.uint8)
    if ignoreedgestrips:
        shmask[[0, -1], :] = 0
        shmask[:, [0, -1]] = 0

    X, Y = np.meshgrid(x, y)
    rawx, rawy = offset_to_raw_xy(X.ravel(), Y.ravel())
    mask = np.all([rawx > -1, rawx < 48, rawy > -1, rawy < 48], axis=0)
    vignmap[np.logical_not(mask).reshape(vignmap.shape)] = 0.
    vignmap[mask.reshape(vignmap.shape)] *= shmask[rawx[mask], rawy[mask]]

    vmap = RegularGridInterpolator((x, y), vignmap[:, ::-1], bounds_error=False, fill_value=0.)
    return vmap


def make_overall_vignetting(energy=6., *args,
                            subgrid=20, urdweights={urdn:1. for urdn in URDNS},
                            **kwargs):
    if subgrid < 1:
        print("ahtung! subgrid defines splines of the translation of multiple vigneting file into one map")
        print("set subgrid to 2")
        subgrid = 2
    #x, y = np.meshgrid(np.linspace(-24., 24., 48*subgrid), np.np.linspace(-24., 24., 48*subgrid))
    xmin, xmax = -24.*DL, 24.*DL
    ymin, ymax = -24.*DL, 24.*DL

    vecs = offset_to_vec(np.array([xmin, xmax, xmax, xmin]),
                         np.array([ymin, ymin, ymax, ymax]))

    vmaps = {}
    for urdn in URDNS:
        quat = ARTQUATS[urdn]
        xlim, ylim = vec_to_offset(quat.apply(vecs))
        xmin, xmax = min(xmin, xlim.min()), max(xmax, xlim.max())
        ymin, ymax = min(ymin, ylim.min()), max(ymax, ylim.max())

    dd = DL/subgrid
    dx = dd - (xmax - xmin)%dd
    xmin, xmax = xmin - dx/2., xmax + dx
    dy = dd - (ymax - ymin)%dd
    ymin, ymax = ymin - dy/2., ymax + dy

    x, y = np.mgrid[xmin:xmax:dd, ymin:ymax:dd]
    shape = x.shape
    newvmap = np.zeros(shape, np.double)
    vecs = offset_to_vec(np.ravel(x), np.ravel(y))

    for urdn in URDNS:
        vmap = make_vignetting_for_urdn(urdn, energy, *args, **kwargs)
        quat = ARTQUATS[urdn]
        newvmap += vmap(vec_to_offset_pairs(quat.apply(vecs, inverse=True))).reshape(shape)*urdweights.get(urdn, 1.)

    vmap = RegularGridInterpolator((x[:, 0], y[0]), newvmap, bounds_error=False, fill_value=0)
    return vmap
