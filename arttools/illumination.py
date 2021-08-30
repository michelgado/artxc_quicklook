from .telescope import URDNS
import numpy as np
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask
from ._det_spatial import offset_to_vec, raw_xy_to_vec
from .orientation import vec_to_pol, pol_to_vec, get_photons_vectors, make_align_quat
from .interval import Intervals
from .time import emptyGTI, GTI
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut
from .mosaic import SkyImage, SkyImageMP
from math import pi, sin, cos
from functools import reduce
from astropy.wcs import WCS
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count
import matplotlib.pyplot as plt


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
        xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis[s:e])).T, 1) - 0.5).astype(np.int)
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
    def __init__(self, ra, dec, wcs, offsets, imask):
        self.sourcevector = pol_to_vec(*np.deg2rad([ra, dec]))
        #wcs, offsets, imask = get_illumination_mask()
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, qalign=None):
        opaxvecs = quats.apply(opax)
        if qalign is None:
            qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        offedges = np.array([self.offsets["OPAXOFFL"], self.offsets["OPAXOFFH"]]).T
        offset_intervals = Intervals(offedges)
        offset_intervals.merge_joint()

        mask = offset_intervals.mask_external(angles)

        sidx = np.argsort(angles[mask])
        cedges = np.searchsorted(angles[mask][sidx], offedges)
        pvecsis = qalign[mask].apply(pvecs[mask])[sidx]

        xy = []
        for i, crval in enumerate(self.offsets["CRVAL"]):
            self.wcs.wcs.crval[1] = crval/3600.
            w1 = WCS(wcs.to_header())
            s, e = cedges[i]
            xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis[s:e])).T, 1) - 0.5).astype(np.int)
            xy.append(xyl)
        il = np.repeat(np.arange(len(xy)), [val.shape[0] for val in xy])
        xy = np.concatenate(xy)
        mask[mask][sidx] = imask[il, xy[:, 0], xy[:, 1]]
        return mask

    def make_mask_function_for_quats(self, quats, opax):
        opaxvecs = quats.apply(opax)
        qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        offedges = np.array([self.offsets["OPAXOFFL"], self.offsets["OPAXOFFH"]]).T
        offset_intervals = Intervals(offedges)
        offset_intervals.merge_joint()

        mask = offset_intervals.mask_external(angles)

        sidx = np.argsort(angles[mask])
        cedges = np.searchsorted(angles[mask][sidx], offedges)

        def newfunc(pvecs):
            pvecsis = qalign[mask].apply(pvecs[mask])[sidx]
            xy = []
            for i, crval in enumerate(self.offsets["CRVAL"]):
                self.wcs.wcs.crval[1] = crval/3600.
                w1 = WCS(wcs.to_header())
                s, e = cedges[i]
                xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis[s:e])).T, 1) - 0.5).astype(np.int)
                xy.append(xyl)
            il = np.repeat(np.arange(len(xy)), [val.shape[0] for val in xy])
            xy = np.concatenate(xy)
            mask[mask][sidx] = imask[il, xy[:, 0], xy[:, 1]]
            return mask
        return newfunc

    def get_events_in_illumination_mask(self, urddata, urdn, attdata):
        attloc = attdata*get_boresight_by_device(urdn)
        opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
        gti = attloc.circ_gti(self.sourcevector, self.offsets["OPAXOFFH"][-1], opax)
        mask = gti.mask_external(urddata["TIME"])
        pvecs = get_photons_vectors(urddata[mask], urdn, attloc)
        mask[mask] = self.get_vectors_in_illumination_mask(attloc(urddata["TIME"][mask]), pvecs, opax)
        return mask

class IlluminationSources(object):
    def __init__(self, ra, dec, mpnum=MPNUM):
        wcs, offsets, imask = get_illumination_mask()
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask
        self.sources = [IlluminationSource(r, d, self.wcs, self.offsets, self.imask) for r, d in zip(ra, dec)]
        self.x, self.y = mgrid[0:48:1, 0:48:1]
        self.mpnum = mpnum

    def get_illumination_bti(self, attdata, urdns=URDNS):
        bti = emptyGTI
        for urdn in list(urdns):
            attloc = attdata*get_boresight_by_device(urdn)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for isource in self.sources:
                bti = bti | attloc.circ_gti(isource.sourcevector, self.offsets["OPAXOFFH"][-1], opax)
        return bti

    def get_events_in_illumination_mask(self, urddata, urdn):
        mask = np.zeros(urddata.size, np.bool)
        for source in self.sources:
            mask = np.logical_or(mask, source.get_events_in_illumination_mask(urddata, urdn, attdata))
        return mask

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, qalignlist=False):
        mask = np.zeros(urddata.size, np.bool)
        for i, source in enumerate(self.sources):
            mask = np.logical_or(mask, source.get_vectors_in_illumination_mask(quats, pvecs, opax, None if not qalignlist else qalignlist[i]))
        return mask

    def get_illumination_expmap(self, wcs, attdata, urdgti, imgfilters, urdweights={}, dtcorr={}, **kargs):
        """
        note: deadtime correction is likely mandatory in the vicinity of illumination sources
        """
        vmap = get_ipsf_interpolation_func()
        sky = SkyImageMP(wcs, vmap)
        for urdn, gti in urdgti.items():
            ipsffunc = unpack_inverse_psf_specweighted_ayut(imgfilters[urdn], **kwargs)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            lgti = gti & self.get_illumination_bti(attdata, urdn)
            ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=lgti, timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
            dtq = dtq*urdweights.get(urdn, 1./7.)
            mfunctions = [source.make_mask_function_for_quats(qval, opax) for source in self.sources]

            shmask = get_shadowmask_by_urd(urdn)
            xloc, yloc = self.x[shmask], self.y[shmask]
            qcorr = offset_to_qcorr(*rawxy_to_opaxoffset(xloc, yloc))
            vecs = raw_xy_to_vec(self.x[shmask], self.y[shmask])
            i, j = rawxy_to_opaxoffset(xloc, yloc)
            for il, jl, q, v in zip(i, j, qcorr, vecs):
                mask = reduce(np.logical_or, [m(v) for m in mfunctions])
                sky._set_core_values(ipsffunc(il, jl))
                sky.interpolate(qval[mask], dtn[mask], False)
        sky._accumulate_img()
        sky._clean_img()





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
