from .telescope import URDNS
import numpy as np
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask, get_shadowmask_by_urd
from ._det_spatial import offset_to_vec, raw_xy_to_vec, rawxy_to_qcorr
from .orientation import vec_to_pol, pol_to_vec, get_photons_vectors, make_align_quat
from .interval import Intervals
from .time import emptyGTI, GTI
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset
from .mosaic import SkyImage, SkyImageMP
from .atthist import make_small_steps_quats
from copy import copy
from math import pi, sin, cos
from functools import reduce
from astropy.wcs import WCS
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
#import matplotlib.pyplot as plt
from threading import Thread


MPNUM = cpu_count()//4
#OFFSETS = ???


def get_events_in_illumination_mask(urdn, srcax, urdevts, attdata):
    attloc = attdata*get_boresight_by_device(urdn)
    pvecs = get_photons_vectors(urdevts, urdn, attdata)
    opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
    opaxvecs = attloc(urdevts["TIME"]).apply(opax)
    qalign = make_align_quat(np.tile(srcax, (opaxvecs.shape[0], 1)), opaxvecs)

    wcs, offsets, imask = get_illumination_mask()

    angles = np.arccos(np.sum(srcax*opaxvecs, axis=-1))*180./pi*3600.
    offedges = np.array([offsets["OPAXOFFL"], offsets["OPAXOFFH"]]).T
    offset_intervals = Intervals(offedges)
    offset_intervals.merge_joint()
    mask = ~offset_intervals.mask_external(angles)
    sidx = np.argsort(angles[~mask])
    cedges = np.searchsorted(angles[~mask][sidx], offedges)
    pvecsis = qalign[~mask].apply(pvecs[~mask])[sidx]
    xy = []
    for i, crval in enumerate(offsets["CRVAL"]):
        wcs.wcs.crval[1] = crval/3600.
        w1 = WCS(wcs.to_header())
        s, e = cedges[i]
        xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis[s:e])).T, 1) - 0.5).astype(int)
        xy.append(xyl)
        plt.imshow(imask[i])
        plt.scatter(xyl[:, 0], xyl[:, 1], marker="x")
        plt.title(crval)
        plt.show()
    il = np.repeat(np.arange(len(xy)), [val.shape[0] for val in xy])
    xy = np.concatenate(xy)
    mask[~mask][sidx] = ~imask[il, xy[:, 0], xy[:, 1]]
    return mask

class IlluminationSource(object):
    def __init__(self, ra, dec, wcs, offsets, imask, app=300.):
        self.sourcevector = pol_to_vec(*np.deg2rad([ra, dec]))
        #wcs, offsets, imask = get_illumination_mask()
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask
        self.x, self.y = np.mgrid[0:imask.shape[1]:1, 0:imask.shape[2]:1]
        self.app = app
        self.cedges = None
        self.mask = None
        self.qalign = None
        self.sidx = None

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax):
        opaxvecs = quats.apply(opax)
        qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        offedges = np.array([self.offsets["OPAXOFFL"], self.offsets["OPAXOFFH"]]).T
        """
        offset_intervals = Intervals(offedges)
        offset_intervals.merge_joint()

        mask = offset_intervals.mask_external(angles)
        """
        mask = np.zeros(angles.size, np.bool)

        sidx = np.argsort(angles)
        cedges = np.searchsorted(angles[sidx], offedges)
        qloc = qalign[sidx]

        for i, crval in enumerate(self.offsets["CRVAL"]):
            s, e = cedges[i]
            if e == s:
                continue
            pvecsis = qloc[s:e].apply(pvecs[sidx[s:e]])
            self.wcs.wcs.crval[1] = crval/3600.
            w1 = WCS(self.wcs.to_header())
            xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis)).T, 1) - 0.5).astype(int)
            #imask = np.copy(self.imask[i])
            y0, x0 = (w1.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
            mask[sidx[s:e]] = self.imask[np.full(xyl.shape[0], i), xyl[:, 1], xyl[:, 0]] & ((xyl[:, 1] - x0)**2 + (xyl[:, 0] - y0)**2. > (self.app/w1.wcs.cdelt[0]/3600.)**2.)
            #imask[(self.x - x0)**2 + (self.y - y0)**2. < (self.app/w1.wcs.cdelt[0]/3600.)**2.] = False
            """
            print(y0, x0, w1.all_world2pix(np.array([[180., 0.],]), 1))
            plt.imshow(imask)
            plt.show()
            """
            #mask[sidx[s:e]] = imask[xyl[:, 1], xyl[:, 0]]
        return mask

    def setup_for_quats(self, quats, opax):
        opaxvecs = quats.apply(opax)
        self.qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        offedges = np.array([self.offsets["OPAXOFFL"], self.offsets["OPAXOFFH"]]).T

        self.sidx = np.argsort(angles)
        self.cedges = np.searchsorted(angles[self.sidx], offedges)

    def mask_vecs_with_setup(self, pvecs, qinitrot=None, mpnum=MPNUM, opax=None):
        pool = ThreadPool(mpnum)
        qvecrot = self.qalign if qinitrot is None else self.qalign*qinitrot
        mask = np.zeros((pvecs.shape[0], len(qvecrot)), np.bool)
        qloc = qvecrot[self.sidx]
        print(len(qloc))
        iidx = np.arange(mask.shape[1])
        for i, crval in enumerate(self.offsets["CRVAL"]):
            s, e = self.cedges[i]
            if s == e:
                continue
            self.wcs.wcs.crval[1] = crval/3600.
            w1 = WCS(self.wcs.to_header())
            pvecsis = np.array(pool.map(qloc[s:e].apply, pvecs)).reshape((-1, 3))
            xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis)).T, 1) - 0.5).astype(int)
            imask = np.copy(self.imask[i])
            y0, x0 = (w1.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
            imask[(self.x - x0)**2 + (self.y - y0)**2. < (self.app/w1.wcs.cdelt[0]/3600.)**2.] = False
            mask[:, self.sidx[s:e]] = imask[xyl[:, 1], xyl[:, 0]].reshape((-1, e - s))
        return mask

    def get_events_in_illumination_mask(self, urddata, urdn, attdata):
        attloc = attdata*get_boresight_by_device(urdn)
        opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
        gti = attloc.circ_gti(self.sourcevector, self.offsets["OPAXOFFH"][-1], opax)
        mask = gti.mask_external(urddata["TIME"])
        if np.any(mask):
            pvecs = get_photons_vectors(urddata[mask], urdn, attdata)
            mask[mask] = self.get_vectors_in_illumination_mask(attloc(urddata["TIME"][mask]), pvecs, opax)
        return mask

class IlluminationSources(object):
    def __init__(self, ra, dec, mpnum=MPNUM):
        wcs, offsets, imask = get_illumination_mask()
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask
        self.sources = [IlluminationSource(r, d, self.wcs, self.offsets, self.imask) for r, d in zip(np.asarray(ra).reshape(-1), np.asarray(dec).reshape(-1))]
        self.x, self.y = np.mgrid[0:48:1, 0:48:1]
        self.mpnum = mpnum

    def get_illumination_bti(self, attdata, urdns=URDNS):
        bti = emptyGTI
        for urdn in list(urdns):
            attloc = attdata*get_boresight_by_device(urdn)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for isource in self.sources:
                bti = bti | attloc.circ_gti(isource.sourcevector, self.offsets["OPAXOFFH"][-1], opax)
        return bti

    def get_events_in_illumination_mask(self, urddata, urdn, attdata):
        mask = np.any([source.get_events_in_illumination_mask(urddata, urdn, attdata) for source in self.sources], axis=0)
        return mask

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, qalignlist=False):
        mask = np.zeros(urddata.size, np.bool)
        for i, source in enumerate(self.sources):
            mask = np.logical_or(mask, source.get_vectors_in_illumination_mask(quats, pvecs, opax, None if not qalignlist else qalignlist[i]))
        return mask

    def get_illumination_expmap(self, wcs, attdata, urdgti, imgfilters, urdweights={}, dtcorr={}, mpnum=MPNUM, kind="stright", **kwargs):
        """
        note: deadtime correction is likely mandatory in the vicinity of illumination sources
        """
        vmap = get_ipsf_interpolation_func()
        sky = SkyImageMP(wcs, vmap, mpnum=mpnum)


        for urdn, gti in urdgti.items():
            taskargs = []
            ipsffunc = unpack_inverse_psf_specweighted_ayut(imgfilters[urdn], **kwargs)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            bti = self.get_illumination_bti(attdata, [urdn,])
            lgti = gti & bti
            print("filters and nonfiltered exposures", gti.exposure, lgti.exposure)
            if lgti.exposure == 0:
                continue
            ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=lgti, timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
            dtq = dtq*urdweights.get(urdn, 1./7.)
            for source in self.sources:
                source.setup_for_quats(qval, opax)

            shmask = get_shadowmask_by_urd(urdn)
            xloc, yloc = self.x[shmask], self.y[shmask]
            qcorr = rawxy_to_qcorr(xloc, yloc)
            vecs = raw_xy_to_vec(self.x[shmask], self.y[shmask])
            i, j = rawxy_to_opaxoffset(xloc, yloc, urdn)

            mask = np.any([source.mask_vecs_with_setup(vecs, qval, mpnum=mpnum) for source in self.sources], axis=0)
            print("urdn %d process (featuring %d workers):" % (urdn, mpnum))
            sky.interpolate_bunch([(qval[m]*q, dtq[m], ipsffunc(il, jl)) for m, q, il, jl in zip(mask, qcorr, i, j) if np.any(m)], kind=kind)

        sky._accumulate_img()
        sky._clean_img()
        return sky.img

def get_illumination_gtis(att, brightsourcevec, urdgti=None):
    offsetmasks = get_illumination_mask()
    if urdgti is None:
        urdgti = {urdn: [emptyGTI for i in range(offsets.size - 1)] for  urdn in URDNS}

    for urdn in URDNS:
        xyOA = get_optical_axis_offset_by_device(urdn)
        vOA = arttools._det_spatial.raw_xy_to_vec(*np.array(xyOA).reshape((2, 1)))[0]
        attloc = att*get_boresight_by_device(urdn)
        gti0 = attloc.circ_gti(brightsourcevec, offsets[0], ax=vOA)
        attloc = attloc.apply_gti(~gti0)
        for i, upper_margin in enumerate(offsets[1:]):
            gloc = attloc.circ_gti(brightsourcevec, offsets[0], ax=vOA)
            urdgti[urdn][i] = urdgti[urdn][i] | gloc
    return urdgti
