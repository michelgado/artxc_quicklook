from .telescope import URDNS
import numpy as np
import tqdm
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask, get_shadowmask_by_urd
from ._det_spatial import offset_to_vec, raw_xy_to_vec, rawxy_to_qcorr
from .orientation import vec_to_pol, pol_to_vec, get_photons_vectors, make_align_quat
from .interval import Intervals
from .time import emptyGTI, GTI
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset, unpack_inverse_psf_with_weights
from .mosaic2 import SkyImage
from .atthist import make_small_steps_quats, make_wcs_steps_quats
from copy import copy
from math import pi, sin, cos
from functools import reduce
from astropy.wcs import WCS
from multiprocessing.pool import ThreadPool
from multiprocessing import cpu_count, Pool
#import matplotlib.pyplot as plt
from threading import Thread
import time
from scipy.spatial.transform import Rotation


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

localillumsource = None

class IlluminationSource(object):
    def __init__(self, ra, dec, wcs, offsets, imask, app=300.):
        self.sourcevector = pol_to_vec(*np.deg2rad([ra, dec]))
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask
        self.x, self.y = np.mgrid[0:imask.shape[1]:1, 0:imask.shape[2]:1]
        self.app = app
        self.cedges = None
        self.mask = None
        self.qalign = None
        self.sidx = None

    @staticmethod
    def illumination_slice(w, imask, pvecs, crval, app, srcvec, opaxvecs):
        qalign = make_align_quat(np.tile(srcvec, (opaxvecs.shape[0], 1)), opaxvecs)

        w.wcs.crval[1] = crval/3600.
        w = WCS(w.to_header())
        xyl = (w.all_world2pix(np.rad2deg(vec_to_pol(qalign.apply(pvecs))).T, 1) - 0.5).astype(int)
        y0, x0 = (w.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
        return np.logical_and(imask[xyl[:, 1], xyl[:, 0]], ((xyl[:, 1] - x0)**2 + (xyl[:, 0] - y0)**2. > (app/w.wcs.cdelt[0]/3600.)**2.))


    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, mpnum=1):
        opaxvecs = quats.apply(opax)
        #qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        #angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        angles = -np.sum(self.sourcevector*opaxvecs, axis=-1)
        offedges = -np.array([np.cos(self.offsets["OPAXOFFL"]*pi/180./3600.), np.cos(self.offsets["OPAXOFFH"]*pi/180./3600.)]).T


        sidx = np.argsort(angles)
        cedges = np.maximum(np.searchsorted(angles[sidx], offedges) - 1, 0)
        mask = np.zeros(sidx.size, np.bool)

        if mpnum > 1:
            pool = Pool(mpnum)
            for (s, e), res in zip(cedges, pool.starmap(self.illumination_slice, [(self.wcs, imask, pvecs[sidx[s:e]], crval, self.app, self.sourcevector, opaxvecs[sidx[s:e]]) for imask, (s, e), crval in zip(self.imask, cedges, self.offsets["CRVAL"]) if e != s])):
                mask[sidx[s:e]] = res
        else:
            qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)
            for i, crval in enumerate(self.offsets["CRVAL"]):
                s, e = cedges[i]
                if e == s:
                    continue
                pvecsis = qalign[sidx[s:e]].apply(pvecs[sidx[s:e]])
                self.wcs.wcs.crval[1] = crval/3600.
                w1 = WCS(self.wcs.to_header())
                xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis)).T, 1) - 0.5).astype(int)
                y0, x0 = (w1.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
                mask[sidx[s:e]] = self.imask[np.full(xyl.shape[0], i), xyl[:, 1], xyl[:, 0]] & ((xyl[:, 1] - x0)**2 + (xyl[:, 0] - y0)**2. > (self.app/w1.wcs.cdelt[0]/3600.)**2.)
        return mask

    def setup_for_quats(self, quats, opax):
        opaxvecs = quats.apply(opax)
        self.qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)
        angles = -np.sum(self.sourcevector*opaxvecs, axis=-1)
        print(angles)
        offedges = -np.array([np.cos(self.offsets["OPAXOFFL"]*pi/181./3600.), np.cos(self.offsets["OPAXOFFH"]*pi/180./3600.)]).T
        self.sidx = np.argsort(angles)
        #self.cedges = np.maximum(np.searchsorted(angles[self.sidx], offedges) - 1, 0)
        self.cedges = np.searchsorted(angles[self.sidx], offedges)
        print("setup done", self.sourcevector)

    @staticmethod
    def setup_initializer(illum_source, qinitrot):
        global localillumsource
        localillumsource = illum_source
        if qinitrot is not None:
            localillumsource.qalign = localillumsource.qalign*qinitrot

    @staticmethod
    def get_mask_for_vector_with_setup(vector):
        global localillumsource
        qvecrot = localillumsource.qalign
        sidx = localillumsource.sidx
        qloc = qvecrot[sidx]
        mask = np.zeros(len(qvecrot), np.bool)
        for i, crval in enumerate(localillumsource.offsets["CRVAL"]):
            s, e = localillumsource.cedges[i]
            if s == e:
                continue
            localillumsource.wcs.wcs.crval[1] = crval/3600.
            w1 = WCS(localillumsource.wcs.to_header())
            pvecsis = qloc[s:e].apply(vector)
            xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis)).T, 1) - 0.5).astype(int)
            imask = np.copy(localillumsource.imask[i])
            y0, x0 = (w1.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
            imask[(localillumsource.x - x0)**2 + (localillumsource.y - y0)**2. < (localillumsource.app/w1.wcs.cdelt[0]/3600.)**2.] = False
            mask[sidx[s:e]] = imask[xyl[:, 1], xyl[:, 0]]
        return mask

    def mask_vecs_with_setup(self, pvecs, qinitrot=None, mpnum=MPNUM, opax=None):
        pool = Pool(mpnum, initializer=self.setup_initializer, initargs=(self, qinitrot))
        return np.array(pool.map(self.get_mask_for_vector_with_setup, pvecs))

    def mask_vecs_with_setup2(self, pvecs, qinitrot=None, mpnum=MPNUM, opax=None):
        pool = ThreadPool(mpnum)
        qvecrot = self.qalign if qinitrot is None else self.qalign*qinitrot
        mask = np.zeros((pvecs.shape[0], len(qvecrot)), np.bool)
        qloc = qvecrot[self.sidx]
        for i, crval in enumerate(self.offsets["CRVAL"]):
            s, e = self.cedges[i]
            if s == e:
                continue
            self.wcs.wcs.crval[1] = crval/3600.
            w1 = WCS(self.wcs.to_header())
            pvecsis = np.array(pool.map(qloc[s:e + 1].apply, pvecs)).reshape((-1, 3))
            xyl = (w1.all_world2pix(np.rad2deg(vec_to_pol(pvecsis)).T, 1) - 0.5).astype(int)
            imask = np.copy(self.imask[i])
            y0, x0 = (w1.all_world2pix(np.array([[180., 0.],]), 1) - 0.5).astype(int)[0]
            imask[(self.x - x0)**2 + (self.y - y0)**2. < (self.app/w1.wcs.cdelt[0]/3600.)**2.] = False
            mask[:, self.sidx[s:e]] = imask[xyl[:, 1], xyl[:, 0]].reshape((-1, e - s))
        return mask

    def get_events_in_illumination_mask(self, urddata, urdn, attdata, mpnum=1):
        opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
        gti = attdata.circ_gti(self.sourcevector, self.offsets["OPAXOFFH"][-1], get_boresight_by_device(urdn).apply(opax))
        mask = gti.mask_external(urddata["TIME"])
        if np.any(mask):
            pvecs = get_photons_vectors(urddata[mask], urdn, attdata)
            attloc = attdata.apply_gti(gti)*get_boresight_by_device(urdn)
            mask[mask] = self.get_vectors_in_illumination_mask(attloc(urddata["TIME"][mask]), pvecs, opax, mpnum)
        return mask


class DataDistributer(object):
    def __init__(self, stack_pixels=False):
        self.ipsffuncs = []
        self.it = []
        self.jt = []
        self.maskt  = []
        self.dtqt = []
        self.qlist = []
        self.qct = []
        self.size = 0
        self.stack_pixels = stack_pixels

    def set_stack_pixels(self, stack_pixels):
        self.stack_pixels = stack_pixels

    def add(self, i, j, mask, dtq, qlist, qcorr, ipsffunc):
        self.it.append(i)
        self.jt.append(j)
        self.maskt.append(mask)
        self.qlist.append(qlist)
        self.dtqt.append(dtq)
        self.qct.append(qcorr)
        self.ipsffuncs.append(ipsffunc)
        self.size = max(self.size, i.size)

    def get_size(self):
        ijt = np.array([np.concatenate(ar) for ar in [self.it, self.jt]])
        m = np.concatenate([np.any(m, axis=1) for m in self.maskt])
        iju, iidx = np.unique(ijt, axis=1, return_inverse=True)
        mres = np.zeros(iju.shape[1], bool)
        np.add.at(mres, iidx, m)
        return np.sum(mres)

    def __iter__(self):
        if self.stack_pixels:
            ipsffunc = self.ipsffuncs[0]
            ijt = np.array([np.concatenate(ar) for ar in [self.it, self.jt]])
            iju, uidx = np.unique(ijt, axis=1, return_inverse=True)
            self.size = iju.shape[1]
            cshift = np.repeat(np.cumsum([0, ] + [i.size for i in self.it[:-1]]), [i.size for i in self.it])
            c1 = np.cumsum([i.size for i in self.it])
            for k, ijpair in enumerate(iju.T):
                iidx = np.where(k == uidx)[0]
                didx = np.searchsorted(c1, iidx)
                lidx = iidx - cshift[iidx]
                mloc = [self.maskt[d] for d in didx]
                if not np.any([np.any(m[iloc]) for m, iloc in zip(mloc, lidx)]):
                    continue
                qloc = [self.qlist[d] for d in didx]
                qct = [self.qct[d] for d in didx]
                dtl = [self.dtqt[d] for d in didx]
                q = Rotation(np.concatenate([(ql[m[iloc]]*qc[iloc]).as_quat() for ql, m, qc, iloc in zip(qloc, mloc, qct, lidx) if np.any(m[iloc])], axis=0))
                dtq = np.concatenate([dt[m[iloc]] for dt, m, iloc in zip(dtl, mloc, lidx)])
                yield (q, dtq, ipsffunc(*ijpair))
        else:
            for i, j, mask, qlist, qc, dtq, ipsffunc in zip(self.it, self.jt, self.maskt, self.qlist, self.qct, self.dtqt, self.ipsffuncs):
                for k in range(i.size):
                    if np.any(mask[k]):
                        yield qlist[mask[k]]*qc[k], dtq[mask[k]], ipsffunc(i[k], j[k])



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

    def get_events_in_illumination_mask(self, urddata, urdn, attdata, mpnum=1):
        mask = np.any([source.get_events_in_illumination_mask(urddata, urdn, attdata, mpnum) for source in self.sources], axis=0)
        return mask

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, qalignlist=False):
        mask = np.zeros(urddata.size, np.bool)
        for i, source in enumerate(self.sources):
            mask = np.logical_or(mask, source.get_vectors_in_illumination_mask(quats, pvecs, opax, None if not qalignlist else qalignlist[i]))
        return mask

    def prepare_data_for_computation(self, wcs, attdata, imgfilters, urdweights={}, dtcorr={}, psfweightfunc=None, mpnum=10, **kwargs):

        filts = list(imgfilters.values())
        matchpsf = all([(filts[0]["ENERGY"] == f["ENERGY"]) & (filts[0]["GRADE"] == f["GRADE"]) for f in filts[:]])
        data = DataDistributer(matchpsf)

        for urdn, f in imgfilters.items():
            if psfweightfunc is None:
                ipsffunc = unpack_inverse_psf_specweighted_ayut(imgfilters[urdn].filter, **kwargs)
            else:
                ipsffunc = unpack_inverse_psf_with_weights(psfweightfunc)
            gti = f.filter["TIME"]
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            bti = self.get_illumination_bti(attdata, [urdn,])
            lgti = gti & bti

            print("nonfiltered and filtered exposures", gti.exposure, lgti.exposure)
            if lgti.exposure == 0:
                continue
            ts, qval, dtq, locgti = make_wcs_steps_quats(wcs, attdata*get_boresight_by_device(urdn), gti=lgti, timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
            dtq = dtq*urdweights.get(urdn, 1./7.)
            for source in self.sources:
                source.setup_for_quats(qval, opax)

            shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            xloc, yloc = self.x[shmask], self.y[shmask]
            qcorr = rawxy_to_qcorr(xloc, yloc)
            vecs = raw_xy_to_vec(xloc, yloc)
            i, j = rawxy_to_opaxoffset(xloc, yloc, urdn)

            mask = np.any([source.mask_vecs_with_setup(vecs, qval, mpnum=mpnum) for source in self.sources], axis=0)
            data.add(i, j, mask, dtq, qval, qcorr, ipsffunc)
        return data

    def get_illumination_expmap(self, wcs, attdata, imgfilters, urdweights={}, dtcorr={}, psfweightfunc=None, mpnum=MPNUM, kind="direct", **kwargs):
        """
        note: deadtime correction is likely mandatory in the vicinity of illumination sources
        """
        vmap = get_ipsf_interpolation_func()
        sky = SkyImage(wcs, vmap, mpnum=mpnum)
        list(sky.clean_image())
        data = self.prepare_data_for_computation(wcs, attdata, imgfilters, urdweights=urdweights, dtcorr=dtcorr, psfweightfunc=psfweightfunc, mpnum=mpnum, **kwargs)

        if kind == "direct":
            sky.rmap_convolve_multicore(data, total=data.get_size())
        elif kind == "fft_convolve":
            sky.fft_convolve_multiple(data)
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
