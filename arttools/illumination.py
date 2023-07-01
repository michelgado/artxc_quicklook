from .telescope import URDNS
import numpy as np
import tqdm
from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device, get_illumination_mask, get_shadowmask_by_urd, get_stray_light_mask_by_urd
from ._det_spatial import offset_to_vec, raw_xy_to_vec, rawxy_to_qcorr, urddata_to_offset, urd_to_vec, raw_xy_to_offset, vec_to_offset_pairs, offset_to_raw_xy, vec_to_offset
from .orientation import get_photons_vectors, make_align_quat
from .filters import Intervals, IndependentFilters
from .time import emptyGTI, GTI
from .aux import DistributedObj
from .vector import vec_to_pol, pol_to_vec, normalize
from .vignetting import DetectorVignetting, DEFAULVIGNIFUN, sensitivity_second_order
from .background import get_background_surface_brigtnress
from .telescope import URDNS
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset, \
    unpack_inverse_psf_with_weights, get_pix_overall_countrate_constbkg_ayut, get_urddata_opaxofset_map, naive_bispline_interpolation, \
    unpack_inverse_psf_datacube_specweight_ayut, naive_bispline_interpolation_specweight
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

class CylindricalProjectionBooleanMask(object):
    """
    this class
    """
    def __init__(self, azgrid, offgrid, ortgrid, mask, opax=None, maxcoffset=None):
        self.ca = -np.cos(azgrid/3600.*pi/180.)
        self.of = np.cos((90. - offgrid/3600.)*pi/180.)
        self.og = np.cos((90. - ortgrid/3600.)*pi/180.)
        self.mask = mask
        self.maxcoffset = -self.ca[-1] if maxcoffset is None else np.cos(maxcoffset*pi/180.)
        self.opax=opax

    def set_axvec(self, opax):
        self.opax = opax

    def set_maxcoffset(self, maxcoffset):
        self.maxcoffset = np.cos(maxcoffset*pi/180.*3600.)


    def get_offsetgrid_gti(self, att, srcvec, opax=None):
        if opax is None:
            if self.opax is None:
                raise ValueError("optical axis is not set")
            opax = self.opax
        return  att.circ_gti(srcvec, self.maxcoffset, opax)

    def check_vecs(self, srcvec, evtvec, opax=None):
        """
        opax - attributed to optical axis vector (1, 3)
        srcvec - (N, 3) - vector of the source rotated back to the telescope coordinate system
        evtsvec - atributed to the each source direction set of events vectors (N, M, 3)
        """

        if opax is None:
            if self.opax is None:
                raise ValueError("optical axis is not set")
            opax = self.opax

        csep = np.sum(opax*srcvec, axis=-1)
        mask = csep > self.maxcoffset

        if srcvec.ndim == 1 and evtvec.ndim == 2:
            if not mask:
                return np.zeros(evtvec.shape[0], bool)

            zidx = np.searchsorted(self.ca, -csep)
            voff = normalize(opax - srcvec*csep)
            vort = np.cross(srcvec, voff, axis=-1)
            mask = np.ones(evtvec.shape[0], bool)

        if srcvec.ndim == 2:
            if not np.any(mask):
                return mask
            zidx = np.searchsorted(self.ca, -csep[mask])
            voff = normalize(opax - srcvec[mask, :]*csep[mask, np.newaxis])
            vort = np.cross(srcvec[mask], voff, axis=-1)
            evtvec = evtvec[mask] if evtvec.ndim == 2 else evtvec

        xidx = np.searchsorted(self.og, np.sum(vort*normalize(evtvec - voff*np.sum(evtvec*voff, axis=-1)[:, np.newaxis]), axis=-1))
        yidx = np.searchsorted(self.of, np.sum(voff*normalize(evtvec - vort*np.sum(evtvec*vort, axis=-1)[:, np.newaxis]), axis=-1))

        mask[mask] = self.mask[zidx, yidx, xidx]
        return mask


class StrayLight(object):
    """
    this code produce code, which masks events, according to their coordinates in the detector plane if they are located within rectangular masks in projection to the upper pannel
    for example if we have an event with detector coordiantes (in mm) X and Y and source vector in this coordinate system with components (x, y, z) then the event falls within mask
    if X + FPinhole*x/z, Y + FPinhole*y/z lies within one of specified rectangles, containing holes of the upper pannel
    ** there actually two upper pannels separated by few cm with different holes patterns, however the holes masks were produced based on the observation of the bright source from different dirrection,
    they therefore, can be quite excessive (since the shape of the hole can actually changes depending on the orientation, but this approach is computationally and memmory effective
    """
    FPinhole = 3e3 #distance in mm toward the upper pannel, covering the ART-XC stray field of view, but containing some holes
    def __init__(self, patches):
        self.patches = np.array(patches)
        if self.patches.size > 0:
            vecs = offset_to_vec(*(np.max(np.abs(self.patches), axis=-1) + raw_xy_to_offset(48, 48)).T)
            self.amax = np.arccos(vecs[:, 0].min())
            vecs = offset_to_vec(*(np.min(np.abs(self.patches), axis=-1) - raw_xy_to_offset(48, 48)).T)
            self.amin = np.arccos(vecs[:, 0].max())
            #self.amin, self.amax = max(0., offangles.min() - pi/180.*/3.), offangles.max() + pi/180./3.
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

    def get_offsetgrid_gti(self, attdata, srcvec):
        g = GTI([])
        return (attdata.circ_gti(srcvec, self.amax*180./pi*3600.) & ~attdata.circ_gti(srcvec, self.amin*180/pi*3600.))


#==================================================================================================================

class Isource(object):
    def __init__(self, name, ra, dec, imask={}, smask={}, maxcoffset=None, usestraylight=True):
        self.name = name
        self.ra = ra
        self.dec = dec
        self.vec = pol_to_vec(*np.deg2rad([ra, dec]).reshape((2, 1)))[0]
        self.imask = {}
        self.smask = {}
        self.maxcoffset = maxcoffset
        self.usestraylight = usestraylight
        for urdn in URDNS:
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            self.imask[urdn] = imask.get(urdn, CylindricalProjectionBooleanMask(*get_illumination_mask(), opax=opax, maxcoffset=maxcoffset))
            self.smask[urdn] = smask.get(urdn, StrayLight(get_stray_light_mask_by_urd(urdn)))


    def check_location_in_illumination(self, urdn, xof, yof, qloc, pixvec=None, mask=None):
        """
        check veather the specific offset in the detectors plane lies within illumination and stray light masks
        """
        srcvec = qloc.apply(self.vec, inverse=True)
        if pixvec is None:
            pixvec = offset_to_vec(xof, yof)


        if qloc.single:
            if mask is None:
                mask = self.imask[urdn].check_vecs(srcvec, pixvec)
            else:
                mask[~mask] = self.imask[urdn].check_vecs(srcvec, pixvec[~mask])
            if self.usestraylight:
                mask[~mask] = self.smask[urdn].check_points(np.tile(srcvec, ((~mask).sum(), 1)), xof[~mask] if type(xof) is np.ndarray else xof, yof[~mask] if type(yof) is np.ndarray else yof)
            return mask

        if mask is None:
            mask = np.zeros(len(qloc), bool)


        mask[~mask] = self.imask[urdn].check_vecs(srcvec[~mask], pixvec[~mask] if pixvec.ndim == 2 else pixvec)
        if self.usestraylight:
            mask[~mask] = self.smask[urdn].check_points(srcvec[~mask], xof[~mask] if type(xof) is np.ndarray else xof, yof[~mask] if type(yof) is np.ndarray else yof)
        return mask

    def get_gti(self, urdn, attdata):
        attloc = attdata.for_urdn(urdn)
        gti = self.imask[urdn].get_offsetgrid_gti(attloc, self.vec)
        if self.usestraylight:
            gti = gti | self.smask[urdn].get_offsetgrid_gti(attloc, self.vec)
        return gti



class IlluminationSources(object):
    def __init__(self, isources, filters=None, **kwargs):
        self.isources = isources
        self.x = np.lib.stride_tricks.as_strided(np.arange(48, dtype=np.uint8), shape=(48, 48), strides=(1, 0)).ravel()
        self.y = np.lib.stride_tricks.as_strided(np.arange(48, dtype=np.uint8), shape=(48, 48), strides=(0, 1)).ravel()
        self.detvecs = raw_xy_to_vec(self.x, self.y)
        self.xof, self.yof = raw_xy_to_offset(self.x, self.y)
        if not filters is None:
            self.set_urdn_data_filters(filters)

    def set_urdn_data_filters(self, filters, scales=None, brates=None, vfun=None, cspec=None):
        self.filters = filters
        self.detmap = {urdn: DetectorVignetting(unpack_inverse_psf_specweighted_ayut(f.filters, cspec=cspec)) for urdn, f in filters.items()}
        self.shmasks = {urdn: f.filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)]) for urdn, f in filters.items()}
        if not brates is None:
            for urdn in self.detmap:
                brate = get_background_surface_brigtnress(urdn, filters[urdn], normalize=True)*brates[urdn]
                self.detmap[urdn].set_bkgratemap(brate)
        if not scales is None:
            for urdn in self.detmap:
                self.detmap[urdn].set_vignscale(scales[urdn])
        if not vfun is None:
            for urdn in self.detmap:
                self.detmap[urdn].set_vignetting_functions(vfun)


    def check_pixel_in_illumination(self, urdn, xof, yof, qloc, pixvec=None, mask=None):
        if mask is None:
            mask = np.zeros(xof.size, bool)
        for isrc in self.isources:
            mask = isrc.check_location_in_illumination(urdn, xof, yof, qloc, pixvec, mask)
        return mask

    def get_vignetting_in_illumination(self, urdn, qloc):
        mask = ~self.shmasks[urdn].ravel()
        mask = self.check_pixel_in_illumination(urdn, self.x, self.y, qloc, self.detvecs, mask)
        mask = mask & self.shmasks[urdn].ravel() #mask
        if not np.any(mask):
            return False
        self.detmap[urdn]._clean_img()
        imap, jmap = get_urddata_opaxofset_map(urdn)
        self.detmap[urdn].produce_vignentting(self.x[mask], self.y[mask], imap[self.x[mask]], jmap[self.y[mask]])
        return True

    def get_gti(self, attdata, urdn=28):
        gti = GTI([])
        for isrc in self.isources:
            gti = gti | isrc.get_gti(urdn, attdata)

    def provide_pixels_in_vignetting(self, attdata, filters):
        for urdn in filters:
            gti = self.get_gti(attdata, urdn) & filters[urdn].filters["TIME"]
            te, gaps = make_small_steps_quats(attdata.for_urdn(urdn))

    def get_illumination_mask(self, attdata, urddata):
        if urddata.data.size == 0:
            return np.zeros(0, bool)
        attloc = attdata.for_urdn(urddata.urdn)
        qloc = attloc(urddata["TIME"])
        xof, yof = raw_xy_to_offset(urddata["RAW_X"], urddata["RAW_Y"])
        pixvec = offset_to_vec(xof, yof)
        return self.check_pixel_in_illumination(urddata.urdn, xof, yof, qloc, pixvec)

    def make_exposures(self, direction, te, attdata, urdfilters, urdweights={}, mpnum=MPNUM, dtcorr={}, app=120., cspec=None):
        """
        estimate exposure within timebins te, for specified directions
        """
        if app is None:
            app= 300.
        cgti = attdata.circ_gti(direction, 25.*60.)
        urdgtis = {urdn: f.filters["TIME"] & cgti for urdn, f in urdfilters.items()}

        gti = reduce(lambda a, b: a | b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
        print("gti exposure", gti.exposure)
        tel, gaps, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
        tc = (tel[1:] + tel[:-1])[gaps]/2.

        shiftsize = int(min(app, 300)//45 + 1)
        print("shiftsize", shiftsize)
        xc, yc = np.mgrid[-shiftsize: shiftsize + 1: 1, -shiftsize: shiftsize + 1: 1] # size of the pixel is 45 arcsec
        detmask = np.zeros((48 + 2*shiftsize, 48 + shiftsize*2), bool)


        dtn = np.zeros(te.size - 1, np.double)

        x, y = np.mgrid[0:48:1, 0:48:1]
        vecs = raw_xy_to_vec(x.ravel(), y.ravel())
        for urdn in urdgtis:
            if urdgtis[urdn].arr.size == 0:
                continue

            ipsfdata = unpack_inverse_psf_datacube_specweight_ayut(urdfilters[urdn].filters, cspec, app)
            shmask = urdfilters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])

            teu, gaps = (urdgtis[urdn] & locgti).make_tedges(tel)
            dtu = np.diff(teu)[gaps]
            tcc = (teu[1:] + teu[:-1])/2.
            tc = tcc[gaps]
            qlist = attdata(tc)*get_boresight_by_device(urdn)
            vsrc = qlist.apply(direction, inverse=True)

            xs, ys = offset_to_raw_xy(*vec_to_offset(vsrc))
            xs, ys = xs[:, np.newaxis] + xc.ravel()[np.newaxis, :], ys[:, np.newaxis] + yc.ravel()[np.newaxis, :]
            mask = np.logical_and.reduce([xs > -1, xs < 48, ys > -1, ys < 48])
            mask[mask] = shmask[xs[mask], ys[mask]]
            cvals = mask.sum(axis=1)
            xs, ys = xs[mask], ys[mask]
            qlist = Rotation(np.repeat(qlist.as_quat(), cvals, axis=0))
            millum = self.check_pixel_in_illumination(urdn, xs, ys, qlist, vecs.reshape((48, 48, 3))[xs, ys])
            mupd, w = naive_bispline_interpolation_specweight(xs[millum], ys[millum], np.repeat(vsrc, cvals, axis=0)[millum], ipsfdata, urdn)
            millum[millum] = mupd

            if np.any(mupd):
                idx = np.searchsorted(te, np.repeat(tc, cvals)[millum]) - 1
                dtu = np.repeat(dtu, cvals)[millum]
                dtc = np.repeat(dtcorr[urdn](tc), cvals)[millum] if urdn in dtcorr else 1.
                print("urdn", urdn, " time", np.sum(dtu*w))
                np.add.at(dtn, idx, w*dtu*dtc*urdweights.get(urdn, 1./7.))
                print("urdweights", urdweights.get(urdn, 1./7.))
        print("dtn sum", dtn.sum())
        return dtn


class WCSSkyWithIllumination(WCSSky, IlluminationSources): #, IlluminationSources):

    def __init__(self, isources, filters, **kwargs):
        IlluminationSources.__init__(self, isources, filters)
        super().__init__(isources=isources, filters=filters, **kwargs)
        #WCSSky.__init__(self, isources, *args, filters=filters, **kwargs)

    @DistributedObj.for_each_process
    def update_filters(self, filters, **kwargs):
        self.set_urdn_data_filters(filters, **kwargs)

    @DistributedObj.for_each_process
    def set_urdn(self, urdn):
        self.urdn = urdn

    @DistributedObj.for_each_argument
    def interpolate_vmap_for_qval(self, qval, scale, update_corners=False):
        if not self.get_vignetting_in_illumination(self.urdn, qval):
            return False
        self.vmap.values = self.detmap[self.urdn].img
        if update_corners:
            self.update_corners()
        img, rmap, vecs = self._get_cutout(qval)
        if vecs.size == 0:
            return False
        xyl = vec_to_offset_pairs(qval.apply(vecs, inverse=True))
        vm = self.vmap(xyl)
        img += self.action(vm, scale, rmap)
        return np.any(vm > 0.)

    def get_expmap(self, attdata, urdfilters, urdweights={}, dtcorr={}, cspec=None):
        urdgtis = {urdn: f.filters["TIME"] for urdn, f in urdfilters.items()}
        self.clean_image()
        self.update_filters({urdn: f.filters for urdn, f in urdfilters.items()}, cspec=cspec)

        for urdn in urdgtis:
            self.set_urdn(urdn)
            gti = urdgtis[urdn]
            if gti.exposure == 0:
                print("urd %d has no individual gti, continue" % urdn)
                continue
            print("urd %d, exposure %.1f, progress:" % (urdn, gti.exposure))
            exptime, qval, locgti = hist_orientation_for_attdata(attdata.for_urdn(urdn), gti, \
                                                                timecorrection=dtcorr.get(urdn, lambda x: 1), \
                                                                wcs=self.locwcs)
            for _ in tqdm.tqdm(self.interpolate_vmap_for_qval(zip(qval, exptime*urdweights.get(urdn, 1/7.))), total=exptime.size):
                pass

        self.accumulate_img()
        return np.copy(self.img)

    def get_theta_sq_component(self, attdata, urdfilters, brates, urdweights={}):
        urdgtis = {urdn: f.filters["TIME"] for urdn, f in urdfilters.items()}
        self.update_filters({urdn: f.filters for urdn, f in urdfilters.items()}, brates=brates, scales={urdn: urdweights.get(urdn, 1/7.) for urdn in urdfilters}, vfun=sensitivity_second_order)

        for urdn in urdgtis:
            self.set_urdn(urdn)
            gti = urdgtis[urdn]
            if gti.exposure == 0:
                print("urd %d has no individual gti, continue" % urdn)
                continue
            print("urd %d, exposure %.1f, progress:" % (urdn, gti.exposure))
            exptime, qval, locgti = hist_orientation_for_attdata(attdata.for_urdn(urdn), gti, \
                                                                wcs=self.locwcs)
            for _ in tqdm.tqdm(self.interpolate_vmap_for_qval(zip(qval, exptime)), total=exptime.size):
                pass

        self.accumulate_img()
        return np.copy(self.img)




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

