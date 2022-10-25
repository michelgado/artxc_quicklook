from .telescope import URDNS
import numpy as np
import tqdm
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask, get_shadowmask_by_urd
from ._det_spatial import offset_to_vec, raw_xy_to_vec, rawxy_to_qcorr
from .orientation import get_photons_vectors, make_align_quat
from .filters import Intervals
from .time import emptyGTI, GTI
from .aux import DistributedObj
from .vector import vec_to_pol, pol_to_vec, normalize
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset, unpack_inverse_psf_with_weights, get_pix_overall_countrate_constbkg_ayut
from .mosaic2 import SkyImage
from .atthist import make_small_steps_quats, make_wcs_steps_quats, hist_orientation_for_attdata
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


MPNUM = cpu_count()//4 + 1
#OFFSETS = ???


def get_events_in_illumination_mask(urdn, srcax, urdevts, attdata):
    """
    for provided wcs and opticalaxis offset grid OPAXOFFL computes weather the events are within illumination mask
    """
    attloc = attdata*get_boresight_by_device(urdn)
    pvecs = get_photons_vectors(urdevts, urdn, attdata)
    opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]


    opaxvecs = attloc(urdevts["TIME"]).apply(opax)

    p1 = normalize(opaxvecs - srcax[np.newaxis, :]*np.sum(srcax*opaxvecs, axis=1)[:, np.newaxis])
    p2 = np.cross(srcax, p1) #, axis=1)
    #print(p2.shape)
    svec = normalize(np.array([-np.sum(srcax*pvecs, axis=1), np.sum(p2*pvecs, axis=1), np.sum(p1*pvecs, axis=1)]).T)
    #wcs, offsets, imask = get_illumination_mask()

    offidx = np.searchsorted(-np.cos(offsets["OPAXOFFL"]*pi/180./3600.), -np.sum(opaxvecs*srcax, axis=1)) - 1
    #print(offidx)
    radec = np.rad2deg(vec_to_pol(svec))
    #print(radec)
    #print(radec.shape)

    y, x = (wcs.all_world2pix(np.rad2deg(vec_to_pol(svec)).T, 0) + 0.5).astype(int).T
    #print(y, x)
    mask = imask[offidx, x, y].astype(bool)
    return mask


class AzimuthMaskSet(object):
    """
    the mask checks, wether the provided vector located within the specific
    azimuth angle interval for specific ring
    it contains regular grid for
    1) optical axis -- source offsets
    2) source -- event offsets

    the mask works as follows:
        it separate all events over their belonging to the particular offset ring, defined by the 1) grid
        within opax--src ring events separated over their offset from source on the set of rings based on the second grid
        for each ring a set of azimuthal angles intervals defined [[begining, end], [beginning, end], ....]
        if source vector is within this interval, a possitive mask is applied to it
    """
    def __init__(self, srcoaxofflb, evtsrclb, srcevtbounds):
        """
        srcoaxofflb -- lower boundary for optical axis - source offset
        evtsrclb -- lower boundary for soruce -- events vectors offset
        srcevtbounds - a set of source centered segments (lower offset boundary and interavls within which mask is True)
        """
        a1 = -np.cos(srcoaxofflb)
        self.srcaxgrid = a1
        self.evtsrgrid = -np.cos(evtsrclb)
        grid = np.repeat(np.arange(self.srcaxgrid.size)*self.evtsrgrid.size, [len(c)*2 for c in srcevtbounds])
        bounds = np.concatenate(srcevtbounds, axis=0)
        self.grid = grid + np.searchsorted(self.evtsrgrid+1e-10, np.repeat(-np.cos(bounds[:, 0]), 2)) + bounds[:, 1:].ravel()/2./pi

    def check_vecs(self, srcvec, axvec, evtvec):
        """
        srcvec - (1, 3) - single vector of the source
        axvec - attributed to fk5 optical axis vector (N, 3)
        evtsvec - atributed to the each optical axis direction set of events vectors (N, M, 3)
        """
        a1 = -np.sum(srcvec*axvec, axis=1)
        p1 = normalize(axvec + a1[:, np.newaxis]*srcvec[np.newaxis, :])
        idx1 = (np.searchsorted(self.srcaxgrid, -np.sum(axvec*srcvec, axis=1)) - 1)*self.evtsrgrid.size
        p2 = np.cross(srcvec, p1) #, axis=1)

        a2 = np.sum(srcvec*evtvec, axis=-1)
        idx2 = np.searchsorted(self.evtsrgrid, -a2) - 1
        if evtvec.ndim == 2:
            angle = np.arctan2(np.sum(p2*evtvec, axis=1), np.sum(p1*evtvec, axis=1))
        else:
            angle = np.arctan2(np.sum(p2[:, np.newaxis, :]*evtvec, axis=-1), np.sum(p1[:, np.newaxis, :]*evtvec, axis=-1))

        vals = idx1 + idx2 + (angle + pi)/2./pi if evtvec.ndim == 2 else idx1[:, np.newaxis] + idx2 + (angle + pi)/2./pi
        return np.searchsorted(self.grid, vals)%2 == 1


class IlluminationSource(object):
    def __init__(self, ra, dec):
        self.sourcevector = pol_to_vec(*np.deg2rad([ra, dec]))

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, mpnum=1):
        opaxvecs = quats.apply(opax)

        #qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)

        #angles = np.arccos(np.sum(self.sourcevector*opaxvecs, axis=-1))*180./pi*3600.
        angles = -np.sum(self.sourcevector*opaxvecs, axis=-1)
        #print(angles)


        mask =  (angles < -cos(self.offsets["OPAXOFFH"][-1]*pi/180./3600.))
        #print(mask.sum())
        pvecs = pvecs[mask]

        p1 = normalize(opaxvecs - self.sourcevector[np.newaxis, :]*np.sum(self.sourcevector*opaxvecs, axis=1)[:, np.newaxis])
        p2 = np.cross(self.sourcevector, p1) #, axis=1)
        #print(p1.shape, p2.shape, pvecs.shape)
        svec = normalize(np.array([-np.sum(self.sourcevector*pvecs, axis=1), np.sum(p2*pvecs, axis=1), np.sum(p1*pvecs, axis=1)]).T)
        offidx = np.searchsorted(-np.cos(self.offsets["OPAXOFFL"]*pi/180./3600.), -np.sum(opaxvecs*self.sourcevector, axis=1)) - 1
        y, x = (self.wcs.all_world2pix(np.rad2deg(vec_to_pol(svec)).T, 0) + 0.5).astype(int).T
        x = np.maximum(np.minimum(x, self.imask.shape[1] - 1), 0)
        y = np.maximum(np.minimum(y, self.imask.shape[0] - 1), 0)

        #mask[mask] = imask[offidx, x, y].astype(bool)
        mask[mask] = self.imask[offidx, x, y].astype(bool)

        return mask

    def setup_for_quats(self, quats, opax):
        opaxvecs = quats.apply(opax)
        self.qalign = make_align_quat(np.tile(self.sourcevector, (opaxvecs.shape[0], 1)), opaxvecs)
        angles = -np.sum(self.sourcevector*opaxvecs, axis=-1)
        offedges = -np.array([np.cos(self.offsets["OPAXOFFL"]*pi/180./3600.), np.cos(self.offsets["OPAXOFFH"]*pi/180./3600.)]).T
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
        res = []
        for m in tqdm.tqdm(pool.imap(self.get_mask_for_vector_with_setup, pvecs), total=pvecs.shape[0]):
            res.append(m)
        return np.array(res)

    def mask_vecs_with_setup2(self, pvecs, qinitrot=None, mpnum=MPNUM, opax=None):
        pool = ThreadPool(mpnum) if mpnum > 1 else type.__new__(type, "MockPool", (), {"map": map})
        qvecrot = self.qalign if qinitrot is None else self.qalign*qinitrot
        mask = np.zeros((pvecs.shape[0], len(qvecrot)), np.bool)
        qloc = qvecrot[self.sidx]
        for i, crval in tqdm.tqdm(enumerate(self.offsets["CRVAL"]), total=self.offsets.shape[0]):
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

    def get_events_in_illumination_mask(self, urddata, urdn, attdata, mpnum=1):
        opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
        gti = attdata.circ_gti(self.sourcevector, self.offsets["OPAXOFFH"][-1], get_boresight_by_device(urdn).apply(opax))
        mask = gti.mask_external(urddata["TIME"])
        if np.any(mask):
            pvecs = get_photons_vectors(urddata[mask], urdn, attdata)
            attloc = attdata.apply_gti(gti)*get_boresight_by_device(urdn)
            mask[mask] = self.get_vectors_in_illumination_mask(attloc(urddata["TIME"][mask]), pvecs, opax, mpnum)
        return mask

    """
    def get_vectors_in_illumination_mask(self, pvecs, urdn, attdata, mpnum=1):
        opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
        gti = attdata.circ_gti(self.sourcevector, self.offsets["OPAXOFFH"][-1], get_boresight_by_device(urdn).apply(opax))
        mask = gti.mask_external(urddata["TIME"])
        if np.any(mask):
            pvecs = pvecs[mask]
            attloc = attdata.apply_gti(gti)*get_boresight_by_device(urdn)
            mask[mask] = self.get_vectors_in_illumination_mask(attloc(urddata["TIME"][mask]), pvecs, opax, mpnum)
        return mask
    """

class MosaicForEachPix(DistributedObj):

    def __init__(self, locwcs, qvals, dtq, shape=None, mpnum=4, barrier=None):
        kwargs = locals()
        kwargs.pop("self")
        kwargs.pop("__class__")
        initpool = Thread(target=super().__init__, kwargs=kwargs) #don't wait for pool initialization since it takes approximately the same time
        initpool.start()
        #super().__init__(**kwargs)

        self.locwcs = locwcs
        self.shape = shape if not shape is None else [(0, int(locwcs.wcs.crpix[1]*2 + 1)), (0, int(locwcs.wcs.crpix[0]*2 + 1))]
        self.img = np.zeros(np.diff(self.shape, axis=1).ravel(), float)
        self.qvals = qvals
        self.dtq = dtq

    @DistributedObj.for_each_argument
    def spread_events(self, x, y, weight, mask, randomize=True):
        v = raw_xy_to_vec(np.ones(mask.sum())*x, np.ones(mask.sum())*y, randomize=randomize)
        xy = self.locwcs.all_world2pix(np.rad2deg(vec_to_pol(self.qval[mask].apply(v))).T, 0).astype(int) + [self.shape[1][0], self.shape[0][0]]
        np.add.at(self.img, (xy[:, 1], xy[:, 0]), self.dtq[mask]*weight)

    @DistributedObj.for_each_process
    def get_img(self):
        return self.img

    def get_clean(self):
        self.img[:, :] = 0.

    @DistributedObj.for_each_process
    def set_qval_and_dtq(self, qval, dtq):
        self.qval = qval
        self.dtq = dtq

    """
    def __iter__(self):
        ones = np.ones(len(self.qlist))
        for x, y, w, m in zip(self.x, self.y, self.weights, self.mask):
            vecs = qlist[mask].apply(raw_xy_to_vec(ones[mask]*x, ones[mask]*y, randomize=True))
            return vecs, w*self.dtq[mask]
    """


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
        if len(self.it) == 0:
            return 0
        ijt = np.array([np.concatenate(ar) for ar in [self.it, self.jt]])
        m = np.concatenate([np.any(m, axis=1) for m in self.maskt])
        iju, iidx = np.unique(ijt, axis=1, return_inverse=True)
        mres = np.zeros(iju.shape[1], bool)
        np.add.at(mres, iidx, m)
        return np.sum(mres)

    def __iter__(self):
        if self.stack_pixels:
            ipsffunc = self.ipsffuncs[0]
            if len(self.it) > 0:
                ijt = np.array([np.concatenate(ar) for ar in [self.it, self.jt]])
            else:
                ijt = np.empty((2, 0), float)
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
    def __init__(self, ra, dec, mpnum=MPNUM, clustered=False):
        wcs, offsets, imask = get_illumination_mask()
        self.wcs = wcs
        self.offsets = offsets
        self.imask = imask
        self.ra = ra
        self.dec = dec
        self.sources = [IlluminationSource(r, d, self.wcs, self.offsets, self.imask) for r, d in zip(np.asarray(ra).reshape(-1), np.asarray(dec).reshape(-1))]
        self.x, self.y = np.mgrid[0:48:1, 0:48:1]
        self.mpnum = mpnum
        self.clustered = clustered

        maxoffset = offsets[-1][2]
        m = self.imask[-1].astype(bool)
        self.wcs.wcs.crval[1] = maxoffset/3600.
        w1 = WCS(self.wcs.to_header())
        """
        source position vector for wcs is placed at 180 0
        have to provide it from caldb
        """
        svec = pol_to_vec(np.array([pi,]), np.array([0.,]))[0]
        x, y = np.mgrid[0:m.shape[0]:1, 0:m.shape[1]:1]
        r, d = w1.all_pix2world(np.array([y[m], x[m]]).T, 0).T
        v = pol_to_vec(np.deg2rad(r), np.deg2rad(d))
        amax = np.arccos(np.min(np.sum(v*svec, axis=1)))

        if not self.clustered:
            clusters = np.arange(len(self.sources))
            for i in range(len(self.sources) - 1):
                tvec = self.sources[i].sourcevector
                for j in range(i + 1,len(self.sources)):
                    gvec = self.sources[j].sourcevector
                    if np.arccos(np.sum(tvec*gvec)) < 2.*amax:
                        clusters[(clusters == clusters[i]) | (clusters == clusters[j])] = min(clusters[i], clusters[j])

            self.clusters = clusters
        else:
            self.clusters = [0,]

    def get_clusters(self):
        return [self.__class__(self.ra[self.clusters == c], self.dec[self.clusters == c], self.mpnum, clustered=True) for c in np.unique(self.clusters)]


    def get_illumination_bti(self, attdata, urdns=URDNS):
        bti = []
        for urdn in list(urdns):
            attloc = attdata*get_boresight_by_device(urdn)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for isource in self.sources:
                bti.append(attdata.circ_gti(isource.sourcevector, self.offsets["OPAXOFFH"][-1], opax))

        return bti

    def get_events_in_illumination_mask(self, urddata, urdn, attdata, mpnum=1):
        mask = np.any([source.get_events_in_illumination_mask(urddata, urdn, attdata, mpnum) for source in self.sources], axis=0)
        return mask

    def get_vectors_in_illumination_mask(self, vecs, urdn, attdata, mpnum=1):
        mask = np.any([source.get_vectors_in_illumination_mask(vecs, urdn, attdata, mpnum) for source in self.sources], axis=0)
        return mask

    def get_vectors_in_illumination_mask(self, quats, pvecs, opax, qalignlist=False):
        mask = np.zeros(urddata.size, np.bool)
        for i, source in enumerate(self.sources):
            mask = np.logical_or(mask, source.get_vectors_in_illumination_mask(quats, pvecs, opax, None if not qalignlist else qalignlist[i]))
        return mask

    def prepare_data_for_computation(self, wcs, attdata, imgfilters, urdweights={}, dtcorr={}, psfweightfunc=None, histattdata=False, mpnum=10, **kwargs):
        filts = list(imgfilters.values())
        matchpsf = all([(filts[0]["ENERGY"] == f["ENERGY"]) & (filts[0]["GRADE"] == f["GRADE"]) for f in filts[:]])
        data = DataDistributer(matchpsf)

        for urdn in imgfilters:
            print("setup %d urdn illumination" % urdn)
            if psfweightfunc is None:
                ipsffunc = unpack_inverse_psf_specweighted_ayut(imgfilters[urdn].filters, **kwargs)
            else:
                ipsffunc = unpack_inverse_psf_with_weights(psfweightfunc)
            gti = imgfilters[urdn].filters["TIME"]
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            btis = self.get_illumination_bti(attdata, [urdn,])
            bti = reduce(lambda a, b: a | b, btis)
            lgti = gti & bti

            print("nonfiltered and filtered exposures", gti.exposure, lgti.exposure)
            if lgti.exposure == 0:
                continue
            if histattdata:
                dtq, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), lgti, wcs=wcs, timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
                msrcs = [np.ones(dtq.size, bool) for _ in range(len(self.sources))]
            else:
                ts, qval, dtq, locgti = make_wcs_steps_quats(wcs, attdata*get_boresight_by_device(urdn), gti=lgti, timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
                print("ts.size", ts.size, len(qval))
                msrcs = [b.mask_external(ts) for b in btis]

            dtq = dtq*urdweights.get(urdn, 1./7.)

            for ms, source in zip(msrcs, self.sources):
                #print("computing illumination masks over pixels", ms.size, ms.sum())
                source.setup_for_quats(qval[ms], opax)

            shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            xloc, yloc = self.x[shmask], self.y[shmask]
            qcorr = rawxy_to_qcorr(xloc, yloc)
            vecs = raw_xy_to_vec(xloc, yloc)
            i, j = rawxy_to_opaxoffset(xloc, yloc, urdn)

            mask = np.zeros((vecs.shape[0], len(qval)), bool)
            for ms, source in zip(msrcs, self.sources):
                #mres = source.mask_vecs_with_setup(vecs, qval[ms], mpnum=mpnum)
                mres = source.mask_vecs_with_setup(vecs, qval[ms], mpnum=mpnum)
                mask[:, ms] = np.logical_or(mask[:, ms], mres)

            #mask = np.any([source.mask_vecs_with_setup(vecs, qval, mpnum=mpnum) for source in self.sources], axis=0)
            data.add(i, j, mask, dtq, qval, qcorr, ipsffunc)
        return data

    def get_illumination_expmap(self, wcs, attdata, imgfilters, urdweights={}, dtcorr={}, psfweightfunc=None, mpnum=MPNUM, kind="direct", **kwargs):
        """
        note: deadtime correction is likely mandatory in the vicinity of illumination sources
        """

        """
        if not self.clustered:
            isrcs = self.get_clusters()
            return sum(s.get_illumination_expmap(wcs, attdata, imgfilters, urdweights, dtcorr, psfweightfunc, mpnum, kind, **kwargs) for s in isrcs)
        """

        vmap = get_ipsf_interpolation_func()
        sky = SkyImage(wcs, vmap, mpnum=mpnum)
        list(sky.clean_image())


        print("put illuminated pixels on sky")
        if kind == "direct":
            data = self.prepare_data_for_computation(wcs, attdata, imgfilters, urdweights=urdweights, dtcorr=dtcorr, psfweightfunc=psfweightfunc, mpnum=mpnum, **kwargs)
            sky.rmap_convolve_multicore(data, total=data.get_size())
        elif kind == "fft_convolve":
            gparkinds = attdata.get_axis_movement_speed_gti(lambda x: x < pi/180.*5./3600.) # select all parkind and compute with interpolations there
            pgti = {urdn: imgfilters[urdn]["TIME"] & gparkinds for urdn in imgfilters}
            mgti = {urdn: imgfilters[urdn]["TIME"] & ~gparkinds for urdn in imgfilters}
            for urdn in imgfilters:
                imgfilters[urdn]["TIME"] = pgti[urdn]
            print("compute parking with interpolations, overall exposure:", reduce(lambda a, b: a | b, pgti.values()).exposure)
            data = self.prepare_data_for_computation(wcs, attdata, imgfilters, urdweights=urdweights, dtcorr=dtcorr, psfweightfunc=psfweightfunc, histattdata=True, mpnum=mpnum, **kwargs)
            if data.get_size() > 0:
                sky.rmap_convolve_multicore(data, total=data.get_size())
            for urdn in imgfilters:
                imgfilters[urdn]["TIME"] = mgti[urdn]
            data = self.prepare_data_for_computation(wcs, attdata, imgfilters, urdweights=urdweights, dtcorr=dtcorr, psfweightfunc=psfweightfunc, mpnum=mpnum, **kwargs)
            if data.get_size() > 0:
                sky.fft_convolve_multiple(data, total=data.get_size())
        return sky.img

    def make_pixmap_projection(self, wcs, attdata, imgfilters, shape=None, mpnum=MPNUM, dtcorr={}, urdweights={}, **kwargs):
        x, y = np.mgrid[0:48:1, 0:48:1]

        #sky = SkyImage(wcs, None, shape, mpnum=mpnum)
        sky = MosaicForEachPix(wcs, None, None, shape, mpnum)
        print("sky done")

        for urdn in imgfilters:
            opix = get_pix_overall_countrate_constbkg_ayut(imgfilters[urdn], **kwargs)
            shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            x0, y0 = get_optical_axis_offset_by_device(urdn)
            opax = raw_xy_to_vec(*np.array([x0 - 0.5, y0 - 0.5]).reshape((2, -1)))[0]
            i, j = np.round(x + 0.5 - x0).astype(int)[shmask], np.round(y + 0.5 - y0).astype(int)[shmask]
            xl, yl = x[shmask], y[shmask]
            vecs = raw_xy_to_vec(xl, yl)

            weights = opix(i, j)
            ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=imgfilters[urdn]["TIME"], timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
            list(sky.set_qval_and_dtq(qval, dtq))

            for source in self.sources:
                source.setup_for_quats(qval, opax)
            print("source setup done")

            mask = np.zeros((vecs.shape[0], len(qval)), bool)
            for source in self.sources:
                mres = source.mask_vecs_with_setup(vecs, qval, mpnum=mpnum)
                mask[:,:] = np.logical_or(mask[:, :], mres)
            print("mask estimation done")

            for _ in tqdm.tqdm(sky.spread_events(zip(xl, yl, weights, ~mask)), total=xl.size):
                pass
        return sum(sky.get_img())


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
