from .telescope import URDNS
import numpy as np
import tqdm
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask, get_shadowmask_by_urd
from ._det_spatial import offset_to_vec, raw_xy_to_vec, rawxy_to_qcorr, urddata_to_offset, urd_to_vec, raw_xy_to_offset, vec_to_offset_pairs
from .orientation import get_photons_vectors, make_align_quat
from .filters import Intervals, IndependentFilters
from .time import emptyGTI, GTI
from .aux import DistributedObj
from .vector import vec_to_pol, pol_to_vec, normalize
from .vignetting import DetectorVignetting, DEFAULVIGNIFUN
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset, unpack_inverse_psf_with_weights, get_pix_overall_countrate_constbkg_ayut, get_urddata_opaxofset_map
from .mosaic2 import SkyImage, WCSSky
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
    svec = normalize(np.array([-np.sum(srcax*pvecs, axis=1), np.sum(p2*pvecs, axis=1), np.sum(p1*pvecs, axis=1)]).T)

    offidx = np.searchsorted(-np.cos(offsets["OPAXOFFL"]*pi/180./3600.), -np.sum(opaxvecs*srcax, axis=1)) - 1
    radec = np.rad2deg(vec_to_pol(svec))

    y, x = (wcs.all_world2pix(np.rad2deg(vec_to_pol(svec)).T, 0) + 0.5).astype(int).T
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
    def __init__(self, srcoaxofflb, evtsrclb, srcevtbounds, axvec=None, app=None):
        """
        srcoaxofflb -- lower boundary for optical axis - source offset
        evtsrclb -- lower boundary for soruce -- events vectors offset
        srcevtbounds - a set of source centered segments (lower offset boundary and interavls within which mask is True)
        """
        minit = np.ones(srcoaxofflb.size, bool) if app is None else srcoaxofflb < app
        a1 = -np.cos(srcoaxofflb)
        self.srcaxgrid = a1[minit]
        self.offit = Intervals(np.tile(srcoaxofflb[minit], (2, 1)).T + [0., np.median(np.diff(srcoaxofflb[minit]))*1.0001,])
        self.evtsrgrid = -np.cos(evtsrclb)
        grid = np.repeat(np.arange(self.srcaxgrid.size)*self.evtsrgrid.size, [len(c)*2 for c, m in zip(srcevtbounds, minit) if m])
        bounds = np.concatenate([s for s, m in zip(srcevtbounds, minit) if m], axis=0)
        self.grid = grid + np.searchsorted(self.evtsrgrid+1e-10, np.repeat(-np.cos(bounds[:, 0]), 2)) + bounds[:, 1:].ravel()/2./pi
        self.axvec = axvec


    def get_offsetgrid_gti(self, att, srcvec, axvec=None):
        if axvec is None:
            if self.axvec is None:
                raise ValueError("optical axis is not set")
            axvec = self.axvec

        g = GTI([])
        for s, e in self.offit.arr:
            g = g | (~att.circ_gti(srcvec, s*180/pi*3600., axvec) & att.circ_gti(srcvec, e*180/pi*3600., axvec))
        return g


    def get_srcoffset_mask(self, srcvec, axvec=None):
        if axvec is None:
            if self.axvec is None:
                raise ValueError("optical axis is not set")
            axvec = self.axvec

        a1 = -np.sum(srcvec*axvec, axis=1)
        mask = Intervals(-np.cos(self.offit.arr)).mask_external(a1)
        return mask


    def check_vecs(self, srcvec, evtvec, axvec=None):
        """
        axvec - attributed to optical axis vector (1, 3)
        srcvec - (N, 3) - vector of the source rotated back to the telescope coordinate system
        evtsvec - atributed to the each source direction set of events vectors (N, M, 3)
        """
        if axvec is None:
            if self.axvec is None:
                raise ValueError("optical axis is not set")
            axvec = self.axvec


        a1 = -np.sum(srcvec*axvec, axis=-1)
        mask = Intervals(-np.cos(self.offit.arr)).mask_external(a1)

        a1 = a1[mask]
        srcvec = srcvec[mask]
        evtvec = evtvec[mask]

        p1 = normalize(axvec + a1[:, np.newaxis]*srcvec)
        idx1 = (np.searchsorted(self.srcaxgrid, -np.sum(axvec*srcvec, axis=1)) - 1)*self.evtsrgrid.size
        p2 = np.cross(srcvec, p1) #, axis=1)

        a2 = np.sum(srcvec*evtvec, axis=-1) if evtvec.ndim == 2 else np.sum(srcvec[:, np.newaxis, :]*evtvec, axis=-1)
        idx2 = np.searchsorted(self.evtsrgrid, -a2) - 1
        if evtvec.ndim == 2:
            angle = np.arctan2(np.sum(p2*evtvec, axis=1), np.sum(p1*evtvec, axis=1))
        else:
            angle = np.arctan2(np.sum(p2[:, np.newaxis, :]*evtvec, axis=-1), np.sum(p1[:, np.newaxis, :]*evtvec, axis=-1))

        if evtvec.ndim == 2:
            vals = idx1 + idx2 + (angle + pi)/2./pi
            mask[mask] = np.searchsorted(self.grid, vals)%2 == 1
        else:
            vals = idx1[:, np.newaxis] + idx2 + (angle + pi)/2./pi
            mlong = np.tile(mask, (evtvec.shape[1], 1)).T
            mlong[mask] = np.searchsorted(self.grid, vals)%2 == 1
            mask = mlong
        return mask

class StrayLight(object):
    FPinhole = 3e3
    def __init__(self, patches):
        self.patches = np.array(patches)
        #print(self.patches.shape)
        #print(np.concatenate(self.patches, axis=1).T.shape)
        """
        self.xb = self.patches[:, 0, :]
        self.xb = self.xb[np.argsort[self.xb[:, 0]]]
        self.xbidx = np.argsort(self.xb[:, 1])

        self.yb = self.patches[:, 1, :]
        self.yb = self.yb[np.argsort(self.yb[:, 0])]
        self.ybidx = np.argsort(self.xb[:, 1])
        """
        if self.patches.size > 0:
            vecs = offset_to_vec(*np.concatenate(self.patches, axis=1))
            offangles = np.arccos(vecs[0])
            self.amin, self.amax = offangles.min(), offangles.max()
        else:
            self.amin, self.amax = 0., 0.

    def check_points(self, scvec, x, y):
        if self.patches.size > 0:
            xo = x - scvec[:, 1]*self.FPinhole/scvec[:, 0]
            mask = (xo[:, np.newaxis] > self.patches[np.newaxis, :, 0, 0]) & (xo[:, np.newaxis] < self.patches[np.newaxis, :, 0, 1])
            mi = np.any(mask, axis=1)
            yo = (y if np.asarray(y).ndim == 0 else y[mi]) + scvec[mi, 2]*self.FPinhole/scvec[mi, 0]
            mask[mi,:] = mask[mi,:] & ((yo[:, np.newaxis] > self.patches[np.newaxis, :, 1, 0]) & (yo[:, np.newaxis] < self.patches[np.newaxis, :, 1, 1]))
            return np.any(mask, axis=1)
        else:
            return np.zeros(scvec.shape[0], bool)
        #print(self.patches)
        """
        for (xl, xh), (yl, yh) in self.patches:
            xo, yo = x - scvec[:, 1]*self.FPinhole/scvec[:, 0], y + scvec[:, 2]*self.FPinhole/scvec[:, 0]
            mask = np.logical_or(mask, np.all([xo > xl, xo < xh, yo > yl, yo < yh], axis=0))

        return mask
        """

    def get_offsetgrid_gti(self, attdata, srcvec):
        g = GTI([])
        return (attdata.circ_gti(srcvec, self.amax*180./pi*3600.) & ~attdata.circ_gti(srcvec, self.amin*180/pi*3600.))


#==================================================================================================================
#a set of specific method for interpolation over sky

def set_global_state(sobj, *args, **kwargs):
    for name, val in kwargs.items():
        sobj.__setattr__(name, val)

def set_detmap(sobj, filters):
    sobj.__setattr__("detmap", {urdn: DetectorVignetting(unpack_inverse_psf_specweighted_ayut(f.filters)) for urdn, f in filters.items()})

def make_s_interpolation(sobj, qval, dtn, urdn, mask):
    sobj.detmap[urdn]._clean_img()
    sobj.detmap[urdn].produce_vignentting(sobj.urdpixcoord[urdn][0][mask], sobj.urdpixcoord[urdn][1][mask], sobj.ijt[urdn][0][mask], sobj.ijt[urdn][1][mask])
    sobj.vmap.values = sobj.detmap[urdn].img
    sobj.interpolate_vmap_for_qval(qval, dtn)

#==================================================================================================================

class Isource(object):
    def __init__(self, name, ra, dec, imask={}, smask={}):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.imask = imask
        self.smask = smask
        self.vec = pol_to_vec(*np.deg2rad([ra, dec]).reshape((2, 1)))[0]


class IlluminationSources(DistributedObj):

    def __init__(self, isources, mpnum=4, barrier=None):
        """
        ifilts is expect to be a set of illumination filters, attributed to each particular telescope had
        """
        self.isources = isources
        """
        self.srcsvec = pol_to_vec(*np.deg2rad([raset, decset]))
        self.smask = np.ones(self.srcvec.shape[0], bool) if smask is None else smask

        self.srcmask = np.ones(self.srcsvec.shape[0], bool)
        self.ifiltsset = ifiltsset
        self.slset = slset
        self.glist = None
        """
        super().__init__(mpnum, barrier, isources=isources) #, decset=decset, ifiltsset=ifiltsset, slset=slset) #=rmap, mask=mask, **kwargs)


    def check_pixel_in_illumination(self, urdn, x, y, qloc):
        xof, yof = raw_xy_to_offset(x, y)
        pixvec = offset_to_vec(xof, yof)
        pixvec = np.lib.stride_tricks.as_strided(pixvec, shape=(len(qloc), 3), strides=(0,pixvec.strides[0]))
        mask = np.zeros(len(qloc), bool)
        for isrc in self.isources:
            srcvecs = qloc.apply(isrc.vec, inverse=True)
            if urdn in isrc.imask:
                mask[~mask] = isrc.imask[urdn].check_vecs(srcvecs[~mask], pixvec[~mask])
            if urdn in isrc.smask:
                mask[~mask] = isrc.smask[urdn].check_points(srcvecs[~mask], xof, yof)
        return mask


    def get_illumination_mask(self, attdata, urddata):

        #mask = g.mask_external(urddata["TIME"])
        u, iu = np.unique(urddata[["RAW_X", "RAW_Y"]], return_inverse=True)
        vecs = urd_to_vec(u)[iu]
        #vecs = urd_to_vec(urddata) #get_photons_vectors(urddata, urddata.urdn, attdata)
        mask = np.zeros(vecs.shape[0], bool)
        qloc = attdata.for_urdn(urddata.urdn)(urddata["TIME"])
        x, y = urddata_to_offset(urddata)
        urdn = urddata.urdn

        for isrc in self.isources:
            srcvecs = qloc.apply(isrc.vec, inverse=True)
            if urdn in isrc.imask:
                mask[~mask] = isrc.imask[urdn].check_vecs(srcvecs[~mask], vecs[~mask])
            if urdn in isrc.smask:
                mask[~mask] = isrc.smask[urdn].check_points(srcvecs[~mask], x[~mask], y[~mask])
        return mask

    @DistributedObj.for_each_argument
    def get_snapshot_mask(self, urdn, q): #, srcmask=None):
        #if srcmask is None:
        #    srcmask = self.srcmask
        mask = np.zeros(self.urdpixvecs[urdn].shape[0], bool)
        xo, yo = self.urdpixoffst[urdn]
        i, j = self.ijt[urdn]
        """
        for srcs, m in zip(self.srcsvec, srcmask):
            if not m:
                continue
        """

        for isrc in self.isources:
            vloc = q.apply(isrc.vec, inverse=True)
            vloc = np.lib.stride_tricks.as_strided(vloc, shape=self.urdpixvecs[urdn].shape, strides=(0, vloc.strides[0]))
            if urdn in isrc.imask:
                mask[~mask] = isrc.imask[urdn].check_vecs(vloc[:(~mask).sum()], self.urdpixvecs[urdn][~mask])
            if urdn in isrc.smask:
                mask[~mask] = isrc.smask[urdn].check_points(vloc[:(~mask).sum()], xo[~mask], yo[~mask])
        return mask if np.any(mask) else None


    def get_overall_gti(self, attdata):
        glist = [] #GTI([]) for _ in self.srcvecs]
        for isrc in self.isources:
            g = GTI([])
            for urdn, imask in isrc.imask.items():
                g = g | imask.get_offsetgrid_gti(attdata.for_urdn(urdn), isrc.vec)
                if urdn in isrc.smask:
                    g = g | isrc.smask[urdn].get_offsetgrid_gti(attdata.for_urdn(urdn), isrc.vec)
            glist.append(g)
        return glist


    def get_pixmap_tasks(self, tc, dt, qval, urdnsfilters, urdweights, pixresponse):
        """
        produces a set of tasks, which can be consumed by the mosaic task
        """
        srcsvecs = {}
        qlist = {}
        for urdn in urdnsfilters:
            qloc = qval*get_boresight_by_device(urdn)
            qlist[urdn] = qloc
            srcsvecs[urdn] = [qloc.apply(isrc.vec, inverse=True) for isrc in self.isources]

        x, y = np.mgrid[0:48:1, 0:48:1]
        ijt = {}
        shmasks = {}
        opaxofmap = {}
        detidxlist = {}
        for urdn in urdnsfilters:
            shmasks[urdn] = urdnsfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            imap, jmap = get_urddata_opaxofset_map(urdn)
            opaxofmap[urdn] = (imap, jmap)
            detidxlist[urdn] = x[shmasks[urdn]], y[shmasks[urdn]]
            ijt[urdn] = np.array([imap[detidxlist[urdn][0]], jmap[detidxlist[urdn][1]]]).T

        tijt = np.concatenate([v for v in ijt.values()], axis=0)
        iju = np.unique(tijt, axis=0)
        dj = iju[:, 1].max() - iju[:, 0].min()
        for urdn in urdnsfilters:
            ijt[urdn] = ijt[urdn][:, 0]*dj + ijt[urdn][:, 1]

        def prepeare_pix_task(ij):
            i, j = ij
            #print("ij", i, j, tc.size, dt.size, len(qval))
            qc = []
            dtt = []

            mask = np.zeros(tc.size, bool)
            for urdn in urdnsfilters:
                idx = np.searchsorted(ijt[urdn], i*dj + j)
                if idx < ijt[urdn].size and ijt[urdn][idx] == i*dj + j:
                    evtvec = np.repeat(raw_xy_to_vec(detidxlist[urdn][0][idx:idx+1], detidxlist[urdn][1][idx:idx+1]), tc.size, axis=0)
                    xo, yo = np.repeat(raw_xy_to_offset(detidxlist[urdn][0][idx:idx+1], detidxlist[urdn][1][idx:idx+1]), tc.size, axis=1)
                    for isrc in self.isources: #srcs, smask in zip(srcsvecs[urdn], self.smask):
                        if urdn in isrc.imask:
                            mask[~mask] = isrc.imask[urdn].check_vecs(srcs[~mask], evtvec[~mask])
                        if urdn in isrc.smask:
                            mask[~mask] = isrc.smask[urdn].check_points(srcs[~mask], xo[~mask], yo[~mask])
                    #print(mask.size, mask.sum())
                    if np.any(mask):
                        qc.append((qlist[urdn][mask]*rawxy_to_qcorr(detidxlist[urdn][0][idx], detidxlist[urdn][1][idx])).as_quat())
                        dtt.append(dt[mask]*urdweights.get(urdn, 1/7.))

            if len(dtt) > 0:
                dtt = np.concatenate(dtt)
                qc = Rotation(np.concatenate(qc))
                return qc, dtt, pixresponse(i, j)
            else:
                return [], np.empty(0, float), None

        pool = ThreadPool(4)
        return pool.imap_unordered(prepeare_pix_task, iju), iju.shape[0]


    def prepare_snapshot_tasks(self, tc, dt, qval, urdnsfilters, urdweights, cspec=None):

        iifuns = {urdn: unpack_inverse_psf_specweighted_ayut(f.filters) for urdn, f in urdnsfilters.items()}
        vmap = {urdn: DetectorVignetting(iifuns[urdn]) for urdn in iifuns}
        x, y = np.mgrid[0:48:1,0:48:1]

        ijt = {}
        urdpixmask = {}
        urdpixcoord = {}
        urdpixvecs = {}
        qboresight = {}

        for urdn in urdnsfilters:
            urdpixmask[urdn] = urdnsfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            imap, jmap = get_urddata_opaxofset_map(urdn)
            urdpixcoord[urdn] = x[urdpixmask[urdn]], y[urdpixmask[urdn]]
            urdpixvecs[urdn] = raw_xy_to_vec(urdpixcoord[urdn][0], urdpixcoord[urdn][1])
            ijt[urdn] = imap[urdpixcoord[urdn][0]], jmap[urdpixcoord[urdn][1]]
            qboresight[urdn] = get_boresight_by_device(urdn)

        """
        produces a set of tasks, which can be consumed by the mosaic task
        """
        #for q, dtn in zip(qval, dt):
        def prepare_slice_vmap(invars): #(q, dtn, urdn)):
            q, dtn, urdn = invars
            qloc = q*qboresight[urdn]
            mask = np.zeros(urdpixvecs[urdn].shape[0], bool)
            xo, yo = urdpixcoord[urdn]
            i, j = ijt[urdn]
            for isrc in self.isources:
                vloc = qloc.apply(isrc.vec, inverse=True)
                vloc = np.lib.stride_tricks.as_strided(vloc, shape=urdpixvecs[urdn].shape, strides=(0, vloc.strides[0]))
                for urdn in set(list(isrc.imask) + list(isrc.smask)):
                    if urdn in isrc.imask:
                        mask[~mask] = isrc.imask[urdn].check_vecs(vloc[:(~mask).sum()], urdpixvecs[urdn][~mask])
                    if urdn in isrc.smask:
                        mask[~mask] = isrc.smask[urdn].check_points(vloc[:(~mask).sum()], xo[~mask], yo[~mask])
            if np.any(mask):
                vmap[urdn]._clean_img()
                img = vmap[urdn].produce_vignentting(xo[mask], yo[mask], i[mask], j[mask])
                res = ([q,], [dtn*urdweights[urdn],], img)
                res = ([], [], None)
            else:
                res = ([], [], None)
            return res

        tp = ThreadPool(5)
        return tp.imap_unordered(prepare_slice_vmap, ((q, dtn, urdn) for urdn in urdnsfilters for q, dtn in zip(qval, dt)))


    def get_illumination_expmap(self, locwcs, attdata, imgfilters, dtcorr={28: lambda x: np.full(x.size, 1.)}, urdweights={}, mpnum=20, kind="fft_convolve", subres=3):
        gti = reduce(lambda a, b: a | b, [f.filters['TIME'] for f in imgfilters.values()])
        res = 0.
        if kind == "fft_convolve":
            pointing_gti = attdata.get_axis_movement_speed_gti(lambda x: x < 4.*pi/180./3600.)
            gti = gti & ~pointing_gti
            print("pointings", pointing_gti.exposure, gti.exposure)
            if pointing_gti.exposure > 0.:
                res = self.get_illumination_expmap(locwcs, attdata, {urdn: f.filters & IndependentFilters({"TIME": pointing_gti}) for urdn, f in imgfilters.items()}, dtcorr, urdweights, mpnum, kind="direct")


        tel, gaps, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
        tc = (tel[1:] + tel[:-1])[gaps]/2.
        qval = attdata(tc)
        dtn = np.diff(tel)[gaps]*dtcorr[28](tc)

        #tc, qval, dtn, gloc =  make_wcs_steps_quats(locwcs, attdata, gti = gti, timecorrection=dtcorr[28])
        iifun = get_ipsf_interpolation_func()
        iicore = unpack_inverse_psf_specweighted_ayut(imgfilters[28])
        if kind=="fft_convolve":
            data, total = self.get_pixmap_tasks(tc, dtn, qval, imgfilters, urdweights, iicore)
            if subres > 1:
                subres = subres//2*2 + 1 # make subre odd
                lwcs = WCS(locwcs.to_header())
                lwcs.wcs.cdelt = locwcs.wcs.cdelt/subres
                lwcs.wcs.crpix = [locwcs.wcs.crpix[0]*subres + subres//2, locwcs.wcs.crpix[1]*subres + subres//2]
                lwcs = WCS(lwcs.to_header())
                shape = [(0, int(locwcs.wcs.crpix[1]*2 + 1)*subres), (0, int(locwcs.wcs.crpix[0]*2 + 1)*subres)]
                sky = WCSSky(lwcs, iifun, shape=shape, mpnum=mpnum)
            else:
                sky = WCSSky(locwcs, iifun, mpnum=mpnum)
            sky.fft_convolve_multiple(data, total=total, subres=subres)
            sky.accumulate_img()
            if subres > 1:
                subres = subres//2*2 + 1 # make subre odd
                res = res + sky.img.reshape((sky.img.shape[0]//subres, subres, sky.img.shape[1]//subres, subres)).sum(axis=(1, 3)) #np.lib.stride_tricks.as_strided(sky.img, shape=[sky.img.shape[0]//subres, sky.img.shape[1]//subres, subres, subres], strides = [sky.img.strides[0], sky.img.strides[1], sky.img.strides[0], sky.img.strides[1]])
            else:
                res = res + np.copy(sky.img)
        else:
            sky = WCSSky(locwcs, DEFAULVIGNIFUN, mpnum=mpnum)
            data = self.prepare_snapshot_tasks(tc, dtn, qval, imgfilters, urdweights)
            sky.rmap_convolve_multicore(data, total=tc.size)
            res = res + np.copy(sky.img)
        return res #np.copy(sky.img)

    def make_exposures(self, srcvec, te, att, filters, app=120., urdweights={}, dtcorr={}):
        """
        repeats the utility of expmap.make_exposures -- i.e. computes corrected for vignetting observed time within te time bins towards fk5 vector srcvec,
        but in case of this implementation, the exposures are computed only for the pixels, which are within illumination of stray light mask
        """
        urdgtis = {urdn: f.filters["TIME"] & Intervals(te[[0, -1]]) for urdn, f in filters.items()}

        gti = reduce(lambda a, b: a | b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
        print("gti exposure", gti.exposure)
        #ts, qval, dtq, locgti = make_small_steps_quats(att, gti=gti, tedges=te)
        tel, gaps, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
        #tc = (tel[1:] + tel[:-1])[gaps]/2.
        #qval = attdata(tc)
        #dtq = np.diff(tel)[gaps]*dtcorr[28](tc)

        dtn = np.zeros(te.size - 1, np.double)
        x, y = np.mgrid[0:48:1, 0:48:1]
        vecs = raw_xy_to_vec(x.ravel(), y.ravel()).reshape(list(x.shape) + [3,])
        vfun = get_ipsf_interpolation_func()
        for urdn in urdgtis:
            if urdgtis[urdn].arr.size == 0:
                continue
            iifun = unpack_inverse_psf_specweighted_ayut(filters[urdn])
            teu, gaps = (urdgtis[urdn] & locgti).make_tedges(tel)
            dtu = np.diff(teu)[gaps]
            tcc = (teu[1:] + teu[:-1])/2.
            tc = tcc[gaps]
            idx = np.searchsorted(te, tc) - 1
            qlist = att(tc)*get_boresight_by_device(urdn)
            vsrc = qlist.apply(srcvec, inverse=True)
            shmask = filters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            dtc = np.ones(tc.size) if not urdn in dtcorr else dtcorr[urdn](tc)

            for xo, yo, v in zip(x[shmask], y[shmask], vecs[shmask]):
                mask = np.sum(v*vsrc, axis=1) > cos(app*pi/180./3600.)
                if np.any(mask):
                    imask = self.check_pixel_in_illumination(urdn, xo, yo, qlist[mask])
                    mask[mask] = imask
                    if np.any(mask):
                        i, j = rawxy_to_opaxoffset(xo, yo, urdn)
                        qc = rawxy_to_qcorr(xo, yo)
                        vfun.values = iifun(i, j)
                        vlist = vfun(vec_to_offset_pairs(qc.apply(vsrc[mask], inverse=True)))
                        np.add.at(dtn, idx[mask], vlist*dtu[mask]*urdweights.get(urdn, 1/7.)*dtc[mask])
        return dtn



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

