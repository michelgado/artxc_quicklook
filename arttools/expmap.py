from .caldb import get_boresight_by_device, get_optical_axis_offset_by_device
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttInvHist, make_small_steps_quats
from .mosaic2 import SkyImage, WCSSky
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting
from .psf import get_ipsf_interpolation_func, unpack_inverse_psf_specweighted_ayut, rawxy_to_opaxoffset, get_pix_overall_countrate_constbkg_ayut
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from .vector import vec_to_pol, pol_to_vec
from .mask import edges
from .lightcurve import weigt_time_intervals
from itertools import repeat
from ._det_spatial import vec_to_offset_pairs, raw_xy_to_vec, rawxy_to_qcorr
from .telescope import URDNS
from functools import reduce
from multiprocessing import cpu_count, Pool, Process, Queue, RawArray, Barrier
from threading import Thread, Lock
from scipy.spatial.transform import Rotation
import numpy as np
import tqdm
from math import cos, pi

MPNUM = cpu_count()

def make_mosaic_expmap_mp_worker(shape, wcs, vmapvals, x, y, qin, qout):
    vmap = np.copy(np.frombuffer(vmapvals).reshape((x.size, y.size)))
    vmap = RegularGridInterpolator((x, y), vmap)
    sky = SkyImage(wcs, shape)

    while True:
        val = qin.get()
        if val == -1:
            break
        qval, expval = val
        sky.put_stright_on(qval, expval)
    qout.put(sky.img)

def collect(res, qout, lock):
    val = qout.get()
    lock.acquire()
    res[0] = res[0] + val
    lock.release()


def make_mosaic_expmap_mp_executor(shape, wcs, vmap, qvals, exptime, mpnum):
    qin = Queue(100)
    qout = Queue()
    lock = Lock()
    res = [0., ]
    threads = [Thread(target=collect, args=(res, qout, lock)) for _ in range(mpnum)]
    for thread in threads:
        thread.start()

    vmapvals = RawArray(vmap.values.dtype.char, vmap.values.size)
    np.copyto(np.frombuffer(vmapvals).reshape(vmap.values.shape), vmap.values)

    pool = [Process(target=make_mosaic_expmap_mp_worker, args=(shape, wcs, vmapvals, vmap.grid[0], vmap.grid[1], qin, qout)) for _ in range(mpnum)]
    for p in pool:
        p.start()

    for i in range(exptime.size):
        qin.put([qvals[i], exptime[i]])
        sys.stderr.write('\rdone {0:%}'.format(i/(exptime.size - 1)))

    for p in pool:
        qin.put(-1)

    for thread in threads:
        thread.join()

    return res[0]

def make_expmap_for_attdata(sky, attdata, imgfilters, dtcorr={}, kind="direct", urdweights={}, **kwargs):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    urdgtis = {urdn: f.filters["TIME"] for urdn, f in imgfilters.items()}


    if kind not in ["direct", "convolve"]:
        raise ValueError("only  convolve and direct option for exposure mosiac is available")


    overall_gti = emptyGTI

    dtcc = {urdn: 1. for urdn in URDNS}
    if dtcorr:
        print('check dtcorr', list(urdgtis.keys()))
        if all([urdn in urdgtis for urdn in URDNS]):
            overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
            print('overall exposure before dtm tolerance', overall_gti.exposure)
            for urdn in URDNS:
                dtm = np.median(dtcorr[urdn].y)
                mdtc = np.abs(dtcorr[urdn].y - dtm)/dtm < 0.01 # dt corr tollerance let it be 1%
                eloc = edges(mdtc) + [0, -1]
                eloc = eloc[eloc[:, 0] != eloc[:, 1]]
                overall_gti = overall_gti & GTI(dtcorr[urdn].x[eloc])
                print('overall gti after %d' % urdn, overall_gti.exposure)
                dtcc[urdn] = dtm
    else:
        overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])

    print("overal exposure", overall_gti.exposure)

    if overall_gti.exposure > 0:
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti, wcs=None if not hasattr(sky, "locwcs") else sky.locwcs)
        vmap = make_overall_vignetting(imgfilters, urdweights={urdn: w*dtcc[urdn] for urdn, w in urdweights.items()}, **kwargs)
        sky.set_vmap(vmap)
        print("exptime sum", exptime.sum())
        print("produce overall urds expmap")
        if kind == "direct":
            sky.direct_convolve(qval, exptime)
        elif kind == "convolve":
            sky.fft_convolve(qval, exptime)

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d, exposure %.1f, progress:" % (urdn, gti.exposure))
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti, \
                                                             timecorrection=dtcorr.get(urdn, lambda x: 1), \
                                                             wcs=None if not hasattr(sky, "locwcs") else sky.locwcs)
        vmap = make_vignetting_for_urdn(urdn, imgfilters[urdn].filters, **kwargs)
        sky.set_vmap(vmap)
        if kind == "direct":
            sky.direct_convolve(qval, exptime*urdweights.get(urdn, 1.))
        elif kind == "convolve":
            sky.fft_convolve(qval, exptime*urdweights.get(urdn, 1.))
        print(" done!")
    return np.copy(sky.img)

def make_expmap_for_wcs(wcs, attdata, imgfilters, shape=None, mpnum=MPNUM, dtcorr={}, kind="direct", urdweights={}, **kwargs):
    if shape is None:
        ysize, xsize = int(wcs.wcs.crpix[0]*2 + 1), int(wcs.wcs.crpix[1]*2 + 1)
        shape = [(0, xsize), (0, ysize)]
    print(wcs)
    sky = WCSSky(wcs, shape=shape, mpnum=mpnum)
    return make_expmap_for_attdata(sky, attdata, imgfilters, dtcorr=dtcorr, kind=kind, urdweights=urdweights, **kwargs)




def make_exposures(direction, te, attdata, urdfilters, urdweights={}, mpnum=MPNUM, dtcorr={}, illum_filters=None, **kwargs):
    """
    estimate exposure within timebins te, for specified directions
    """
    urdgtis = {urdn: f.filters["TIME"] for urdn, f in urdfilters.items()}

    tec, mgaps, se, scalefunc, cumscalefunc = weigt_time_intervals(urdgtis)
    gti = reduce(lambda a, b: a | b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
    print("gti exposure", gti.exposure)
    ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
    tel = np.empty(ts.size*2, np.double)
    tel[::2] = ts - dtq/2.
    tel[1::2] = ts + dtq/2.
    tel = np.sort(tel)
    tetot, gaps = gti.make_tedges(tel)
    dtn = np.zeros(te.size - 1, np.double)
    x, y = np.mgrid[0:48:1, 0:48:1]
    for urdn in urdgtis:
        if urdgtis[urdn].arr.size == 0:
            continue
        teu, gaps = urdgtis[urdn].make_tedges(tel)
        dtu = np.diff(teu)[gaps]
        tcc = (teu[1:] + teu[:-1])/2.
        tc = tcc[gaps]
        qlist = attdata(tc)*get_boresight_by_device(urdn)
        vmap = make_vignetting_for_urdn(urdn, urdfilters[urdn].filters, **kwargs)
        dtc = dtcorr.get(urdn, lambda x: 1.)(tc)
        vval = vmap(vec_to_offset_pairs(qlist.apply(direction, inverse=True)))*urdweights.get(urdn, 1.)*dtc
        idx = np.searchsorted(te, tc) - 1
        mloc = (idx >= 0) & (idx < te.size - 1)
        np.add.at(dtn, idx[mloc], vval[mloc]*dtu[mloc])
    print("dtn sum", dtn.sum())
    return te, dtn

def make_exposures_for_app(direction, te, attdata, urdfilters, urdweights={}, mpnum=MPNUM, dtcorr={}, app=None, illum_filters=None, **kwargs):
    urdgtis = {urdn: f.filters["TIME"] for urdn, f in urdfilters.items()}
    tec, mgaps, se, scalefunc, cumscalefunc = weigt_time_intervals(urdgtis)
    ipsffun = get_ipsf_interpolation_func()
    gti = reduce(lambda a, b: a | b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
    #print("gti exposure", gti.exposure)
    ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
    #print("dtq sum", dtq.sum())
    tel = np.empty(ts.size*2, np.double)
    tel[::2] = ts - dtq/2.
    tel[1::2] = ts + dtq/2.
    tel = np.sort(tel)
    tetot, gaps = gti.make_tedges(tel)
    dtn = np.zeros(te.size - 1, np.double)
    x, y = np.mgrid[0:48:1, 0:48:1]
    cosa = cos(app*pi/180./3600.)
    for urdn in urdgtis:
        if urdgtis[urdn].arr.size == 0:
            continue
        teu, gaps = urdgtis[urdn].make_tedges(tel)
        dtu = np.diff(teu)[gaps]
        tcc = (teu[1:] + teu[:-1])/2.
        tc = tcc[gaps]
        qlist = attdata(tc)*get_boresight_by_device(urdn)
        vmap = make_vignetting_for_urdn(urdn, urdfilters[urdn].filters, app=app, **kwargs)
        dtc = dtu*dtcorr.get(urdn, lambda x: 1.)(tc)*urdweights.get(urdn, 1.)
        vval = vmap(vec_to_offset_pairs(qlist.apply(direction, inverse=True)))*dtc
        idx = np.searchsorted(te, tc) - 1
        mloc = (idx >= 0) & (idx < te.size - 1)
        np.add.at(dtn, idx[mloc], vval[mloc])
        if not illum_filters is None:
            #pool = ThreadPool(mpnum)
            ipsff = unpack_inverse_psf_specweighted_ayut(urdfilters[urdn].filters, **kwargs)
            shmask = urdfilters[urdn].filters.meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
            xloc, yloc = x[shmask], y[shmask]
            vec = raw_xy_to_vec(xloc, yloc)
            opax = raw_xy_to_vec(*np.array(get_optical_axis_offset_by_device(urdn)).reshape((2, 1)))[0]
            for source in illum_filters.sources:
                source.setup_for_quats(qlist, opax)
            mask = np.any([source.mask_vecs_with_setup(vec, qlist, mpnum=mpnum) for source in illum_filters.sources], axis=0)
            m2 = np.array([(np.sum(vec*q.apply(direction, inverse=True), axis=1) > cosa) for q in qlist]).T
            mask = np.logical_and(mask, m2)
            mask = np.logical_and(mask, mloc[np.newaxis, :])
            i, j = rawxy_to_opaxoffset(xloc, yloc, urdn)
            qcorr = rawxy_to_qcorr(xloc, yloc)
            print("illumination exposure", mask.sum(), mask.size - mask.sum(), len(qlist))
            for il, jl, qc, m, v in zip(i, j, qcorr, mask, vec):
                if not np.any(m):
                    continue
                #print(il, jl, m.sum())
                ipsffun.values = ipsff(il, jl)
                vals = -ipsffun(vec_to_offset_pairs((qlist[m]*qc).apply(direction, inverse=True)))*dtc[m]
                np.add.at(dtn, idx[m], vals)
    return te, dtn

def make_expmap_for_healpix(attdata, urdgtis, mpnum=MPNUM, dtcorr={}, subscale=4):
    if dtcorr:
        overall_gti = emptyGTI
        emap = 0.
    else:
        overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
        exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti)
        vmap = make_overall_vignetting()
        print("produce overall urds expmap")
        emap = AttHealpixhist.make_mp(2048, vmap, exptime, qval, mpnum, subscale=subscale*2)
        print("\ndone!")

    for urd in urdgtis:
        gti = urdgtis[urd] & ~overall_gti
        if gti.size == 0:
            print("urd %d has no individual gti, continue" % urd)
            continue
        print("urd %d progress:" % urd)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urd), gti)
        vmap = make_vignetting_for_urdn(urd)
        emap = AttHealpixHist.make_mp(2048, vmap, exptime, qval, mpnum, subscale=subscale) + emap
        print(" done!")
    make_vignetting_for_urdn.clear_cache()
    return emap


def initpool(shmask, shape, weights, wcs, barr, ifilter, opx):
    global wprod, xll, yll, xlsize, locimg, lwcs, barrier, illum_filter, opax
    barrier = barr
    lwcs = wcs
    x, y = np.mgrid[0:48:1, 0:48:1]
    x, y = x[shmask], y[shmask]
    xlsize = shmask.sum()
    locimg = np.zeros(shape, float)
    wprod = np.tile(weights, 1000)
    xll, yll = np.tile(x, 1000), np.tile(y, 1000)
    illum_filter = ifilter
    opax = opx

def poolworker(args):
    global wprod, xll, yll, xlsize, locimg, lwcs, illum_filter, opax
    qlist, dtq = args
    qlist = Rotation(qlist)
    pvecs = raw_xy_to_vec(xll, yll, randomize=True)
    for source in illum_filter.sources:
        source.setup_for_quats(qlist, opax)

    mask = np.zeros((pvecs.shape[0], len(qlist)), bool)
    for source in illum_filter.sources:
        mres = source.mask_vecs_with_setup2(pvecs, qlist, mpnum=1)
        mask[:, :] = np.logical_or(mask, mres)

    pvecs = np.concatenate([qlist[m].apply(v) for v, m in zip(pvecs, ~mask)], axis=0)
    #qprod = Rotation(np.repeat(qlist, xlsize, axis=0)) #ts.size - sl*1000].as_quat(), xl.size, axis=0))
    rd = np.rad2deg(vec_to_pol(pvecs)) #qprod.apply()))
    xy = (lwcs.all_world2pix(rd.T, 0) + 0.5).astype(int)
    np.add.at(locimg, (xy[:, 1], xy[:, 0]), np.repeat(dtq, xlsize)*wprod)

def acquire(args):
    global locimg, barrier
    barrier.wait()
    return locimg

def make_pixmap_projection(wcs, attdata, imgfilters, shape=None, mpnum=MPNUM, dtcorr={}, urdweights={}, illum_filter=None, **kwargs):
    x, y = np.mgrid[0:48:1, 0:48:1]

    shape = shape if not shape is None else [(0, int(wcs.wcs.crpix[1]*2 + 1)), (0, int(wcs.wcs.crpix[0]*2 + 1))]
    img = np.zeros(np.diff(shape, axis=1).ravel(), float)
    for urdn in imgfilters:
        opix = get_pix_overall_countrate_constbkg_ayut(imgfilters[urdn], **kwargs)
        x0, y0 = get_optical_axis_offset_by_device(urdn)
        opax = raw_xy_to_vec(*np.array([[x0,], [y0],]))[0]
        shmask = imgfilters[urdn].meshgrid(["RAW_Y", "RAW_X"], [np.arange(48), np.arange(48)])
        i, j = np.round(x + 0.5 - x0).astype(int)[shmask], np.round(y + 0.5 - y0).astype(int)[shmask]
        xl, yl = x[shmask], y[shmask]
        weights = opix(i, j)
        ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=imgfilters[urdn]["TIME"], timecorrection=dtcorr.get(urdn, lambda x: np.ones(x.size)))
        barrier = Barrier(mpnum)
        pool = Pool(mpnum, initializer=initpool, initargs=(shmask, img.shape, weights, wcs, barrier, illum_filter, opax))
        for _ in tqdm.tqdm(pool.imap_unordered(poolworker, ((qval[sl*1000:(sl + 1)*1000].as_quat(), dtq[sl*1000:(sl + 1)*1000]) for sl in range(dtq.size//1000))), total=ts.size//1000):
            pass
        img += sum(pool.imap_unordered(acquire, repeat(None, pool._processes)))

        if ts.size//1000 > 0:
            xlsize = xl.size
            slast = ts.size%1000
            wprod = np.tile(weights, slast)
            xl, yl = np.tile(xl, slast), np.tile(yl, slast)
            qprod = Rotation(np.repeat(qval[-slast:].as_quat(), xlsize, axis=0)) #ts.size - sl*1000].as_quat(), xl.size, axis=0))
            rd = np.rad2deg(vec_to_pol(qprod.apply(raw_xy_to_vec(xl, yl, randomize=True))))
            xy = (wcs.all_world2pix(rd.T, 0) + 0.5).astype(int)
            np.add.at(img, (xy[:, 1], xy[:, 0]), np.repeat(dtq[-slast:], xlsize)*wprod)
    return img
