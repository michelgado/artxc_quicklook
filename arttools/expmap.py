from .caldb import get_boresight_by_device
from .atthist import hist_orientation_for_attdata, AttWCSHist, AttHealpixHist, AttInvHist, make_small_steps_quats
from .mosaic import SkyImage
from .vignetting import make_vignetting_for_urdn, make_overall_vignetting, make_vignetting_from_inverse_psf
from .time import gti_intersection, gti_difference, GTI, emptyGTI
from .lightcurve import weigt_time_intervals
from ._det_spatial import vec_to_offset_pairs
from .telescope import URDNS
from functools import reduce
from multiprocessing import cpu_count, Pool, Process, Queue, RawArray
from threading import Thread, Lock
import numpy as np

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

def make_expmap_for_wcs(wcs, attdata, urdgtis, imgfilters, shape=None, mpnum=MPNUM, dtcorr={}, kind="direct", urdweights={}, **kwargs):
    """
    produce exposure map on the provided wcs area, with provided GTI and attitude data

    There are two hidden nonobvious properties of the input data expected:
    1) gti is expected to be a dict with key is urd number
        and value is elevant for this urd gti in the form of Nx2 numpy array
    2) wcs is expected to be astropy.wcs.WCS class,
        crpix is expected to be exactly the central pixel of the image
    """
    if shape is None:
        ysize, xsize = int(wcs.wcs.crpix[0]*2 + 1), int(wcs.wcs.crpix[1]*2 + 1)
        shape = [(0, xsize), (0, ysize)]

    if kind not in ["direct", "convolve"]:
        raise ValueError("only  convolve and direct option for exposure mosiac is available")
    print(wcs)
    print(shape)

    sky = SkyImage(wcs, shape=shape)

    if dtcorr:
        overall_gti = emptyGTI
    else:
        overall_gti = reduce(lambda a, b: a & b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
        print(overall_gti.exposure)
        if overall_gti.exposure > 0:
            exptime, qval, locgti = hist_orientation_for_attdata(attdata, overall_gti)
            vmap = make_overall_vignetting(imgfilters, urdweights=urdweights, **kwargs)
            print("exptime sum", exptime.sum())
            print("produce overall urds expmap")
            sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values)
            if kind == "direct":
                sky.interpolate_mp(qval[:], exptime[:], mpnum)
            elif kind == "convolve":
                sky.convolve(qval, exptime, mpnum)
            #emap = AttInvHist.make_mp(wcs, vmap, exptime, qval, mpnum)
            #emap = make_mosaic_expmap_mp_executor(shape, wcs, vmap, qval, exptime, mpnum)
            print("\ndone!")

    for urdn in urdgtis:
        gti = urdgtis[urdn] & ~overall_gti
        if gti.exposure == 0:
            print("urd %d has no individual gti, continue" % urdn)
            continue
        print("urd %d progress:" % urdn)
        exptime, qval, locgti = hist_orientation_for_attdata(attdata*get_boresight_by_device(urdn), gti, \
                                                             dtcorr.get(urdn, lambda x: 1))
        vmap = make_vignetting_for_urdn(urdn, imgfilters[urdn], **kwargs)
        sky._set_core(vmap.grid[0], vmap.grid[1], vmap.values*urdweights.get(urdn, 1))
        if kind == "direct":
            sky.interpolate_mp(qval[:], exptime[:], mpnum)
        elif kind == "convolve":
            sky.convolve(qval, exptime, mpnum)
        #emap = make_mosaic_expmap_mp_executor(shape, wcs, vmap, qvals, exptime, mpnum) + emap
        #emap = AttInvHist.make_mp(wcs, vmap, exptime, qval, mpnum) + emap
        print(" done!")
    return sky.img

def make_exposures(direction, te, attdata, urdgtis, urdfilters, urdweights={}, mpnum=MPNUM, dtcorr={}, **kwargs):
    tec, mgaps, se, scalefunc, cumscalefunc = weigt_time_intervals(urdgtis)
    gti = reduce(lambda a, b: a | b, [urdgtis.get(URDN, emptyGTI) for URDN in URDNS])
    print("gti exposure", gti.exposure)
    ts, qval, dtq, locgti = make_small_steps_quats(attdata, gti=gti, tedges=te)
    print("dtq sum", dtq.sum())
    tel = np.empty(ts.size*2, np.double)
    tel[::2] = ts - dtq/2.
    tel[1::2] = ts + dtq/2.
    tel = np.sort(tel)
    tetot, gaps = gti.make_tedges(tel)
    dtn = np.zeros(te.size - 1, np.double)
    for urdn in urdgtis:
        if urdgtis[urdn].arr.size == 0:
            continue
        teu, gaps = urdgtis[urdn].make_tedges(tel)
        dtu = np.diff(teu)[gaps]
        tcc = (teu[1:] + teu[:-1])/2.
        tc = tcc[gaps]
        qlist = attdata(tc)*get_boresight_by_device(urdn)
        vmap = make_vignetting_for_urdn(urdn, urdfilters[urdn], **kwargs)
        dtc = dtcorr.get(urdn, lambda x: 1.)(tc)
        vval = vmap(vec_to_offset_pairs(qlist.apply(direction, inverse=True)))*urdweights.get(urdn, 1.)*dtc
        idx = np.searchsorted(te, tc) - 1
        mloc = (idx >= 0) & (idx < te.size - 1)
        np.add.at(dtn, idx[mloc], vval[mloc]*dtu[mloc])
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
