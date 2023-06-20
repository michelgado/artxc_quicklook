#!/opt/soft/psoft/python_venv/bin/python3
from arttools.plot import make_mosaic_for_urdset_by_gti, get_attdata, make_sky_image, make_energies_flags_and_grades
from arttools.orientation import read_bokz_fits, AttDATA
from arttools.planwcs import make_wcs_for_attdata, split_survey_mode
from arttools.time import tGTI, get_gti, GTI, emptyGTI, deadtime_correction
from arttools.expmap import make_expmap_for_wcs
from arttools.background import make_bkgmap_for_wcs
import tqdm
from arttools.telescope import URDNS
import arttools
import sys
import subprocess
import os
from scipy.spatial import cKDTree
from math import pi, sqrt, log, log10, sin, cos
import numpy as np
from astropy.io import fits
from functools import reduce
from itertools import repeat
import re
from scipy.interpolate import interp1d
from copy import copy
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Barrier
from scipy.optimize import minimize, root
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import gamma, gammainc
from scipy.ndimage import gaussian_filter
import time
import pymc3
from astropy.table import Table, QTable
from astropy import units as au
import pandas
from astropy.time import Time, TimeDelta
import datetime


vmap = None

surveys, surveygtis = None, None
tstart, fnames, atts = None, None, None

urdcrates = arttools.caldb.get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}

imgfilters = {urdn: arttools.filters.IndependentFilters({"ENERGY": arttools.filters.Intervals([4., 30.]),
                                                        "GRADE": arttools.filters.RationalSet(range(10)),
                                                        ("RAW_X", "RAW_Y"): arttools.filters.get_shadowmask_filter(urdn)}) for urdn in arttools.telescope.URDNS}

bkgfilters = arttools.filters.IndependentFilters({"ENERGY": arttools.filters.Intervals([40., 100.]),
                                                "RAW_X": arttools.filters.InversedRationalSet([0, 47]),
                                                "RAW_Y": arttools.filters.InversedRationalSet([0, 47]),
                                                "GRADE": arttools.filters.InversedRationalSet([])})



def pfunc(x, e):
    g = gammainc(x[0], e*x[1])
    return np.diff(g)


def load_surve_att():
    global tstart, fnames, atts
    if tstart is None:
        tstart, fnames, atts = pickle.load(open("/srg/a1/work/andrey/ART-XC/all_sky_survey/surveyatt.pkl", "rb"))
    return tstart, fnames, atts

def load_survey_gtis():
    global surveys, surveygtis
    if surveys is None:
        surveys = np.loadtxt("/srg/a1/work/andrey/ART-XC/all_sky_survey/surveys_times.txt")
        surveygtis = {i + 1: arttools.time.GTI([surveys[i],np.inf if i > surveys.size - 2 else surveys[i + 1]]) for i in range(surveys.size)}
    return surveys, surveygtis

def radec_to_name(r, d):
    return "SRGA_J" + "%02d%02d%02d.%1d" % ((r/360.*24.), (r*24./360.*60)%60, (r*24/360*3600.)%60, (r*24/360*36000.)%10) + "%+02d%02d%02d" % ((d), abs(d*60)%60, abs(d*3600.)%60)

def make_names(ra, dec):
    names = []
    for r, d in zip(ra, dec):
        names.append(radec_to_name(r, d))
    return names


def update_att(flist, tstart, fnames, atts):
    attlist = []
    gnew = arttools.time.emptyGTI
    if tstart is None:
        tstart, fnames, atts = load_surve_att()

    for i, ddir in enumerate(flist):
        if ddir.rstrip() in fnames:
            continue
        gfname = [os.path.join(ddir.rstrip(), "L0", l) for l in os.listdir(os.path.join(ddir.rstrip(), "L0")) if "gyro.fits" in l][0]
        att = arttools.orientation.get_attdata(gfname)
        if (att.gti & ~atts.gti & ~gnew).exposure == 0:
            continue
        g1 = att.get_axis_movement_speed_gti(lambda x: (x < 110.) & (x > 75.)) & ~atts.gti & ~gnew
        if g1.exposure == 0:
            continue
        te, gaps = g1.arange(600)
        att = arttools.orientation.AttDATA(te, att(te), gti=g1)
        gnew = gnew | att.gti
        attlist.append(att)
        tstart.append(te[0])
        fnames.append(ddir.rstrip())
    atts = arttools.orientation.AttDATA.concatenate([atts,] + attlist)

    tstart = np.array(tstart)
    idx = np.argsort(tstart)
    fnames = [fnames[i] for i in idx]
    pickle.dump([tstart, fnames, atts], open("/srg/a1/work/andrey/ART-XC/all_sky_survey/surveyatt.pkl", "wb"))
    return tstart, fnames, atts

def mprob_quantiles(sample, val=None, quantiles=[0.9,]):
    if val is None:
        val = np.median(np.random.choice(sample, 10))
    idx = np.argsort(np.abs(val - sample))
    ss = sample[idx]
    return [(ss[:int(q*ss.size)].min(), ss[:int(q*ss.size)].max()) for q in list(quantiles)]


def init_ipsf_weight_pool():
    global vmap
    vmap = arttools.psf.get_ipsf_interpolation_func()

def get_ipsf_weight(args):
    i, j, e, ax, q = args
    vmap.values = arttools.psf.unpack_inverse_psf_ayut(i, j, e)
    return vmap(arttools._det_spatial.vec_to_offset(q.apply(ax, inverse=True)))


def make_spec(ra, dec, survey=None, flist=None, usergti=tGTI):

    ra, dec = float(ra), float(dec)
    print("processing spec for (ra, dec, survey):", ra, dec, survey)
    srcname = radec_to_name(ra, dec)
    fname = srcname[:12]
    if not survey is None:
        fname = fname + ("_s%s" % survey)

    if os.path.exists("%s.pha" % fname):
        print("pha file exisys, skip", fname)
        return None

    with open("%s.log" % fname, "w") as report:
        report.write("processing %.5f %.5f survey: %s\n" % (ra, dec, survey))

        survtstart, survfnames, attsurvey = load_surve_att()

        ax = arttools.orientation.pol_to_vec(*np.deg2rad([ra, dec]))
        gloc = attsurvey.circ_gti(ax, 25.*60.)

        if not survey is None:
            surveys, surveygtis = load_survey_gtis()
            sprod = [int(t) for t in survey.rsplit(",")]
            gloc = gloc & reduce(lambda a, b: a | b, [surveygtis[i] for i in sprod])

        report.write("gti: \n")
        for tt in gloc.arr:
            t1 = Time(arttools.caldb.MJDREF, format="mjd") + TimeDelta(tt, format="sec")
            report.write("%s %s\n" % (t1[0].iso, t1[1].iso))

        if not flist is None:
            allfiles = [l.rstrip() for l in open(flist)]
            attfiles = [l for l in allfiles if "gyro.fits" in l]
            urdfiles = [l for l in allfiles if "urd.fits" == l[-8:]]
        else:
            dirs = [survfnames[i] for i in np.unique(np.searchsorted(survtstart, gloc.arr.ravel())) - 1]
            attfiles = []
            urdfiles = []
            for d in dirs:
                attfiles += [os.path.join(d, "L0", l) for l in os.listdir(os.path.join(d, "L0")) if "gyro.fits" in l]
                urdfiles += [os.path.join(d, "L1b", l) for l in os.listdir(os.path.join(d, "L1b")) if "urd.fits" == l[-8:]]

        if os.path.exists("%s_evt.pkl" % fname):
            att, urdgti, bkgtimes, urdevtt, bkggti = pickle.load(open("%s_evt.pkl" % fname, "rb"))
        else:

            att = None
            att = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(l) for l in attfiles])
            gti = att.circ_gti(ax, 25*60.) & gloc
            print("used gti", gti)

            bdata = []
            bkggti = {}
            urdevt = {}

            for urdn in imgfilters:
                imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([4., 30.])

            for fname in urdfiles:
                print(fname)
                pfile = fits.open(fname)
                urdn = fits.getheader(fname.replace("L1b", "L0"), 1)["URDN"]
                lgti = arttools.time.GTI.from_hdu(pfile["GTI"]) + [-2, 2]  + [10, -10] & gti
                if lgti.exposure == 0:
                    continue
                imgfilters[urdn]["TIME"] = imgfilters[urdn].get("TIME", arttools.time.emptyGTI) | lgti
                data = np.copy(pfile[1].data)
                b = np.copy(data["TIME"])[bkgfilters.apply(data)]
                bkggti[urdn] = (lgti + [-250, 250]) & arttools.time.GTI.from_hdu(pfile["GTI"])
                print(bkggti[urdn].exposure)
                bdata.append(b[bkggti[urdn].mask_external(b)])
                data = data[imgfilters[urdn].apply(data)]
                vecs = arttools.orientation.get_photons_vectors(data, urdn, att)
                mask = np.sum(vecs*ax, axis=1) > cos(pi/180.*2200./3600.)
                urdevt[urdn] = urdevt.get(urdn, []) + [data[mask],]

            fname = srcname[:12]
            if not survey is None:
                fname = fname + ("_s%s" % survey)

            urdgti = {urdn: imgfilters[urdn]["TIME"] for urdn in imgfilters}
            bkgtimes = np.sort(np.concatenate(bdata))
            urdevtt = {urdn: np.concatenate(d) for urdn, d in urdevt.items()}
            tgti = reduce(lambda a, b: a | b, urdgti.values())
            pickle.dump([att.apply_gti(tgti), urdgti, bkgtimes, urdevtt, bkggti], open("%s_evt.pkl" % fname, "wb"))
            report.write("events stored in %s\n" % ("%s_evt.pkl" % fname))
            report.write("events format: pickle file with list containig [attdata, urdgtis, bkgeventarrivaltimes, urdevents, bkggti]\n")


        if os.path.exists("%s.fits.gz" % fname):
            pmap = fits.getdata("%s.fits.gz" % fname)
            locwcs = WCS(fits.getheader("%s.fits.gz" % fname, 1))
        else:
            imgfilters4_12 = {urdn: copy(f) for urdn, f in imgfilters.items()}
            for urdn in imgfilters4_12:
                imgfilters4_12[urdn]["ENERGY"] = arttools.filters.Intervals([4., 12.])

            urdevt = {urdn: urdevtt[urdn][imgfilters4_12[urdn].apply(urdevtt[urdn])] for urdn in urdevtt}
            print("4-12 events", {urdn: d.size for urdn, d in urdevt.items()})

            tgti = reduce(lambda a, b: a | b, urdgti.values())
            locwcs = arttools.planwcs.make_tan_wcs(ra*pi/180., dec*pi/180., sizex=15, sizey=15, pixsize=5./3600., alpha=0.)
            #locwcs = arttools.planwcs.make_wcs_for_attdata(att, gti=tgti, pixsize=5./3600.)
            #locwcs.wcs.crpix = [31, 31]


            qlist = [arttools.orientation.get_events_quats(urdevt[urdn], urdn, att)*arttools._det_spatial.get_qcorr_for_urddata(urdevt[urdn]) for urdn in arttools.telescope.URDNS if urdn in urdevt and urdevt[urdn].size > 0]
            qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

            i, j = zip(*[arttools.psf.urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in arttools.telescope.URDNS if urdn in urdevt])
            i, j = np.concatenate(i), np.concatenate(j)

            imgdata = arttools.telescope.concat_data_in_order(urdevt)
            eenergy = arttools.telescope.concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

            urdbkg = arttools.background.get_background_lightcurve(bkgtimes, bkggti, bkgfilters, 2000., imgfilters4_12)

            bkgrates = {urdn: arttools.background.get_local_bkgrates(urdn, urdbkg[urdn], imgfilters4_12[urdn], urdevt[urdn]) for urdn in arttools.telescope.URDNS}
            bkgrates = arttools.telescope.concat_data_in_order(bkgrates)

            photprob = arttools.background.get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
            photprob = arttools.telescope.concat_data_in_order(photprob)

            vmapl = arttools.psf.get_ipsf_interpolation_func()
            pkoef = photprob/bkgrates
            print("sky initialized")

            ije, sidx, ss, sc = arttools.psf.select_psf_grups(i, j, eenergy)

            sky = arttools.mosaic2.SkyImage(locwcs, vmapl, mpnum=20)
            emap = arttools.expmap.make_expmap_for_wcs(locwcs, att, urdgti, imgfilters4_12, urdweights=urdcrates) #, urdweights=urdcrates) #emin=4., emax=12., phot_index=1.9)
            tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(arttools.psf.unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]


            mask = emap > 1.
            sky.set_mask(mask)

            ctot = np.maximum(gaussian_filter(img1.astype(float), 3.)*2.*pi*9., 1.)
            sky.set_action(arttools.mosaic2.get_source_photon_probability)
            rmap = np.maximum(ctot, 1.)/np.maximum(emap, 1.)
            sky.set_rmap(np.maximum(ctot, 1.)/np.maximum(emap, 1.))

            for _ in range(61):
                sky.clean_image()
                ctasks = [(q, s, c) if np.all(m) else (q[m], s[m], c) for (q, s, c), m in zip(ctasks, sky.rmap_convolve_multicore(ctasks, ordered=True, total=len(ctasks))) if np.any(m)]
                sky.accumulate_img()
                mold = np.copy(sky.mask)
                sky.set_mask(np.all([sky.mask, ~((sky.img < ctot) & (sky.img < 0.5)), np.abs(sky.img - ctot) > np.maximum(ctot, 2)*5e-3], axis=0))
                print("img mask", sky.mask.size, sky.mask.sum(), "total events", np.sum([t[1].size for t in ctasks]))
                print("zero photons cts: ", sky.mask.sum(), "conv hist", np.histogram(np.abs(ctot[sky.mask] - sky.img[sky.mask])/ctot[sky.mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 10000.]))
                ctot[mold] = np.copy(sky.img[mold])
                sky.set_rmap(ctot/np.maximum(emap, 1.))
                sky.img[:, :] = 0.
                if not np.any(sky.mask):
                    break
            sky.clean_image()
            sky.set_action(arttools.mosaic2.get_zerosource_photstat)
            sky.set_rmap(ctot/np.maximum(emap, 1.))
            sky.set_mask(np.ones(sky.mask.shape, bool))
            sky.img[:, :] = 0.
            sky.rmap_convolve_multicore(tasks, total=len(tasks))

            pmap = -ctot - sky.img
            fits.ImageHDU(data =pmap/log(10.), header=locwcs.to_header()).writeto("%s.fits.gz" % fname, overwrite=True)
            report.write("image stored in %s\n" % ("%s.fits.gz" % fname))
        x, y = np.unravel_index(np.argmax(pmap), pmap.shape)
        print("old coordinates", ra, dec)
        report.write("old coordinates %.5f %.5f\n" % (ra, dec))
        ra, dec = locwcs.all_pix2world(np.array([y + 1, x + 1]).reshape((-1, 2)), 1)[0]
        print("new coordinates", ra, dec)
        report.write("new coordinates %.5f %.5f\n" % (ra, dec))

        ax = arttools.orientation.pol_to_vec(*np.deg2rad([ra, dec]))
        srcvec = ax

        for urdn in urdevtt:
            vecs = arttools.orientation.get_photons_vectors(urdevtt[urdn], urdn, att)
            mask = np.sum(vecs*ax, axis=1) > cos(pi/180.*120./3600.)
            urdevtt[urdn] = urdevtt[urdn][mask]

        tgti = reduce(lambda a, b: a | b, urdgti.values())

        phot = []
        amod = []
        arf = []

        emin, emax = np.array([[4., 12.], [4., 8.], [8., 12.], [12., 20.]]).T
        for k, (el, eh) in enumerate(zip(emin, emax)):

            report.write("processing energy band %f %f\n" % (el, eh))

            urdevt = {}

            for urdn in imgfilters:
                imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([el, eh])
                urdgti[urdn] = urdgti[urdn] & att.get_axis_movement_speed_gti(lambda x: x > 70.)
                imgfilters[urdn]["TIME"] = urdgti[urdn]
                urdevt[urdn] = urdevtt[urdn][imgfilters[urdn].apply(urdevtt[urdn])]
                print("urdn", urdn, urdevt[urdn].size)
                report.write("URDN %d selected events: %d\n" % (urdn, urdevt[urdn].size))



            urdbkg = arttools.background.get_background_lightcurve(bkgtimes, bkggti, bkgfilters, 2000., imgfilters)

            if np.sum([d.size for d in urdevt.values()]) > 0:

                qlist = [arttools.orientation.get_events_quats(urdevt[urdn], urdn, att)*arttools._det_spatial.get_qcorr_for_urddata(urdevt[urdn]) for urdn in arttools.telescope.URDNS if urdn in urdevt and urdevt[urdn].size > 0]
                qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

                i, j = zip(*[arttools.psf.urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in arttools.telescope.URDNS if urdn in urdevt])
                i, j = np.concatenate(i), np.concatenate(j)

                imgdata = arttools.telescope.concat_data_in_order(urdevt)
                eenergy = arttools.telescope.concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

                bkgrates = {urdn: arttools.background.get_local_bkgrates(urdn, urdbkg[urdn], imgfilters[urdn], urdevt[urdn]) for urdn in arttools.telescope.URDNS}
                bkgrates = arttools.telescope.concat_data_in_order(bkgrates)

                photprob = arttools.background.get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
                photprob = arttools.telescope.concat_data_in_order(photprob)

                for urdn in imgfilters:
                    imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([el, eh])

                lgti = tgti
                te, gaps = lgti.arange(1e6)
                pool = Pool(processes=40, initializer=init_ipsf_weight_pool)
                w = pool.map(get_ipsf_weight, zip(i, j, eenergy, repeat(srcvec), qlist))
                pool.close()
            else:
                i, j, w, photprob, bkgrates = np.empty((5, 0), float)
            te, dtn = arttools.expmap.make_exposures(srcvec, te, att, {urdn: g & lgti for urdn, g in urdgti.items()}, imgfilters, app=120., urdweights=urdcrates)
            if dtn.sum() == 0:
                continue
            lcs = arttools.background.get_bkg_lightcurve_for_app(urdbkg, {urdn: g & lgti for urdn, g in urdgti.items()}, imgfilters, att, srcvec, 120., te)
            print("expexted bkg", lcs.sum())
            report.write("pure exposure: %.2f, vignetting corrected exposure %.2f, expected bkg counts %.2f\n" % (np.sum(np.diff(te)[dtn > 0.]), dtn.sum(), lcs.sum()))
            slr = np.ones(i.size + 1, np.double)
            blr = np.ones(i.size + 1, np.double)
            blr[0] = lcs.sum()
            slr[0] = dtn.sum()
            slr[1:] = w*photprob
            blr[1:] = bkgrates
            pickle.dump([slr, blr, dtn, i, j], open("photons_%d.pkl" % k, "wb"))
            mrate = minimize(lambda rate: slr[0]*rate + np.sum(np.log(blr[1:]/(slr[1:]*rate + blr[1:]))), [max(i.size - lcs.sum(), 1)/slr[0],], method="Nelder-Mead")
            report.write("estimated count rate: %.3f \n " % mrate.x[0])
            report.write("crab count rate: %.3f \n" % arttools.spectr.get_crabrate(imgfilters))

            with pymc3.Model() as mo:
                rate = pymc3.Uniform("rate", lower=0., upper=1000) #min(abs(mrate.x[0])*4.5, 30))
                o = np.ones(slr.size, np.int)
                o[0] = 0
                o, blr, slr = o[slr > 0], blr[slr > 0], slr[slr > 0]
                obs = pymc3.Poisson("obs", mu=blr + slr*rate, observed=o)
                trace = pymc3.sample(4096)

            #pickle.dump([trace["rate"], slr, blr, tgti.exposure], open("tracetest.pkl", "wb"))
            #pause

            plt.subplot(141 + k)
            h, e, _ = plt.hist(trace["rate"], 128)
            h = h.astype(int)

            if k > 0:
                rr = []
                ns = []
                pq = []
                for i in range(max(1, int(e[np.argmax(h)]*dtn.sum() - 3)), max(eenergy.size + 2, 3)):
                    ns.append(i)
                    """
                    individual probability:
                        P(mu) = mu^k/k! exp(-mu)
                        integral of probability over mu = int P(mu) dmu = 1/k! int mu^k exp(-mu) dmu  = 1/k! Gamma(k + 1)
                    """
                    scale = i*trace["rate"].size/np.sum(trace["rate"])/tgti.exposure
                    #res2 = minimize(lambda x: -poisson.logpmf(i, trace['rate']*x[0]*tgti.exposure).sum(), [scale,], method="Nelder-Mead")
                    #print("optimization", res2, scale, i)
                    pqual = -poisson.logpmf(h, pfunc((i, scale*tgti.exposure), e)*trace["rate"].size).sum()
                    rr.append(scale)
                    pq.append(pqual)
                    plt.plot(e[1:], pfunc((i, scale*tgti.exposure), e)*trace["rate"].size, label="%d %.1e" % (i, pqual))

                print("events statistics", ns, rr, pq)
                n = ns[np.argmin(pq)]
                eareamod = rr[np.argmin(pq)]
                arf.append(arttools.spectr.get_crabrate(imgfilters)/arttools.spectr.get_crabpflux(el, eh))
                phot.append(n)
                amod.append(eareamod)

        plt.legend(frameon=False)
        plt.savefig("%s.pdf" % fname)

        phot = np.array(phot)
        amod = np.array(amod)
        arf = np.array(arf)

        np.savetxt("spec.dat", np.array([emin[1:], emax[1:], phot, np.sqrt(phot)]).T)

        subprocess.run(["flx2xsp", "spec.dat", "%s.pha" % fname, "%s.rsp" % fname])
        time.sleep(1.)
        pfile = fits.open("%s.pha" % fname)
        d = Table(pfile[1].data).to_pandas()
        d["COUNTS"] = d.RATE.astype(int)
        d.drop(columns=["RATE", "STAT_ERR"], inplace=True)
        hdu = fits.BinTableHDU(header=pfile[1].header, data=Table.from_pandas(d))

        hdu.header["INSTRUM"] = "TEL [1-7]"
        hdu.header["TELESCOP"] = "ART-XC"
        hdu.header["HDUCLAS3"] = "COUNT"
        hdu.header["TUNIT1"] = ""
        hdu.header["TUNIT2"] = "count"
        hdu.header["TUNIT2"] = "count"
        hdu.header["RA-OBJ"] = ra
        hdu.header["DEC-OBJ"] = dec
        hdu.header["OBJECT"] = srcname
        hdu.header["POISSERR"] = True
        hdu.header["EXPOSURE"] = tgti.exposure
        fits.HDUList([pfile[0], hdu, fits.BinTableHDU(data=Table.from_pandas(pandas.DataFrame({"TSTART": gloc.arr[:, 0], "TSTOP":gloc.arr[:, 1]})), name="GTI")]).writeto("%s.pha" % fname, overwrite=True)

        pfile = fits.open("%s.rsp" % fname)
        ee, de, ec = np.loadtxt("/srg/a1/work/andrey/ART-XC/Crab/crabspec.dat").T
        crab = interp1d(ee, ec)

        pfile[2].data["MATRIX"] = arf*amod
        pfile.writeto("%s.rsp" % fname, overwrite=True)
        report.write("spectr stored in %s\n" % ("%s.pha" % fname))

def poltovec(ra, dec):
    shape = np.asarray(ra).shape
    vec = np.empty(shape + (3,), np.double)
    vec[..., 0] = np.cos(dec*pi/180.)*np.cos(ra*pi/180.)
    vec[..., 1] = np.cos(dec*pi/180.)*np.sin(ra*pi/180.)
    vec[..., 2] = np.sin(dec*pi/180.)
    return vec

#skycells = np.copy(fits.getdata("/srg/a1/work/srg/data/eRO_4700_SKYMAPS.fits", 1))
#skycelltree = cKDTree(poltovec(skycells["RA_CEN"], skycells["DE_CEN"]))
SEARCHRAD = 2.*sin(4.9/2.*pi/180.)

urdbkgsc = {28: 1.0269982359153347,
            22: 0.9461951470620872,
            23: 1.029129860773177,
            24: 1.0385034889253482,
            25: 0.9769294100898714,
            26: 1.0047417556512688,
            30: 0.9775021015829128}

import pickle
#bkigti = pickle.load(open("/srg/a1/work/andrey/ART-XC/gc/allbki2.pickle", "rb"))
#allbki = reduce(lambda a, b: a | b, bkigti.values())


def get_neighbours(fpath):
    ids = fpath.split("/")[-1]
    ra, dec = float(ids[:3]), 90. - float(ids[3:])
    print(ra, dec)
    cells = skycelltree.query_ball_point(poltovec(ra, dec), SEARCHRAD)
    print(cells)
    print(list(zip(skycells["RA_CEN"][cells], skycells["DE_CEN"][cells])))
    cells = ["%03d%03d" % (np.ceil(skycells[c]["RA_CEN"]), np.ceil(90. - skycells[c]["DE_CEN"])) for c in cells]
    return cells


def analyze_survey(fpath, pastday=None):
    abspath = fpath #os.path.abspath(".")
    allfiles = os.listdir(os.path.join(abspath, "L0"))

    bokzfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "bokz.fits" in l]
    gyrofiles = [os.path.join(abspath, "L0", l) for l in allfiles if "gyro.fits" in l]
    urdfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "urd.fits" in l]

    date = gyrofiles[-1][-29:-14]


    if not pastday is None:
        allfiles = os.listdir(os.path.join(pastday, "L0"))
        gyrofiles += [os.path.join(pastday, "L0", l) for l in allfiles if "gyro.fits" in l]
        urdfiles += [os.path.join(pastday, "L0", l) for l in allfiles if "urd.fits" in l]

    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    gti = attdata.get_axis_movement_speed_gti(lambda x: (x > 60.) & (x < 120.))
    gti.remove_short_intervals(3.)
    if gti.exposure == 0:
        return None

    print("overall exposure", gti.exposure)

    attdata = attdata.apply_gti(gti + [3, -3])
    gtis = split_survey_mode(attdata)

    for k, sgti in enumerate(gtis):
        #if (sgti & allbki).exposure == 0:
        if os.path.exists("bmap%02d_%s.fits.gz" % (k, date)):
            #print(sgti.exposure, allbki.exposure)
            #print(sgti.arr[[0, -1], [0, 1]], allbki.arr)
            continue
        print("start", k, "exposure", sgti.exposure)
        make_mosaic_for_urdset_by_gti(urdfiles, gyrofiles, sgti + [-30, 30],
                                      "cmap%02d_%s.fits.gz" % (k, date),
                                      "bmap%02d_%s.fits.gz" % (k, date),
                                      "emap%02d_%s.fits.gz" % (k, date),
                                      usedtcorr=False)

def make_img(flist, outputname, usergti=tGTI, emin=4., emax=12., make_detmap=False, ra=None, dec=None):
    tfpat = re.compile(".*T\d{1}_cl.evt")
    allfiles = [l.rstrip() for l in open(flist)]
    attfiles = [l for l in allfiles if "gyro.fits" in l or "att" in l]
    urdfiles = [l for l in allfiles if "urd.fits" == l[-8:] or tfpat.match(l)]

    attdata = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(gf) for gf in attfiles])
    attdata = attdata.apply_gti(usergti + [-3, 3])

    attgti = arttools.time.tGTI
    if not ra is None and not dec is None:
        attgti = attdata.circ_gti(arttools.vector.pol_to_vec(*np.deg2rad([ra, dec]).reshape((2, -1)))[0], 25.*60)


    for urdn in imgfilters:
        imgfilters[urdn]["TIME"] = attdata.gti & usergti & attgti
        imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([float(emin), float(emax)])

    print("check attdata", attdata.gti.exposure)

    urddata, urdhk = arttools.containers.read_urdfiles(urdfiles, {urdn: arttools.filters.IndependentFilters({"TIME": attdata.gti}) for urdn in arttools.telescope.URDNS}) #[f.replace("L0", "L1b") for f in urdfiles])
    for urdn in urddata:
        if not "ENERGY" in urddata[urdn].data.dtype.names:
            urddata[urdn] = arttools.energy.add_energies_and_grades(urddata[urdn], urdhk[urdn], arttools.caldb.get_energycal_by_urd(urdn), arttools.caldb.get_escale_by_urd(urdn))
    bkgdata = {urdn: d.apply_filters(bkgfilters) for urdn, d in urddata.items()}
    bkgtimes = np.sort(np.concatenate([d["TIME"] for d in bkgdata.values()]))
    urdbkg = arttools.background.get_background_lightcurve(bkgtimes, bkgdata, 1000., imgfilters)
    urddtc = {urdn: arttools.time.deadtime_correction(hk) for urdn, hk in urdhk.items()}

    urdevt = {urdn: d.apply_filters(imgfilters[urdn]) for urdn, d in urddata.items()}
    imgf = {urdn: d.filters for urdn, d in urdevt.items()}

    tgti = reduce(lambda a, b: a | b, [d.filters["TIME"] for d in urdevt.values()])

    pixsize = 10./3600.
    lwcs = arttools.planwcs.make_wcs_for_attdata(attdata, gti=tgti, pixsize=pixsize)
    print("initialized lwcs", lwcs)

    bmap = arttools.background.make_bkgmap_for_wcs(lwcs, attdata, urdbkg, imgf, mpnum=30, kind="convolve") #time_corr=urddtc) #, illuminations=illum_filters)

    radec = np.concatenate([np.rad2deg(arttools.orientation.vec_to_pol(arttools.orientation.get_photons_vectors(d, urdn, attdata, randomize=True))).T for urdn, d in urdevt.items() if d.size > 0])
    xy = (lwcs.all_world2pix(radec, 1) - 0.5).astype(int)
    xy = xy[np.all([xy[:, 0] > 0, xy[:, 1] > 0, xy[:, 0] <  bmap.shape[1], xy[:, 1] < bmap.shape[0]], axis=0)]
    u, uc = np.unique(xy, axis=0, return_counts=True)
    img1 = np.zeros(bmap.shape, int)
    img1[u[:, 1], u[:, 0]] = uc

    femap = arttools.expmap.make_expmap_for_wcs(lwcs, attdata, imgf, urdweights=urdcrates, kind="convolve")#, dtcorr=urddtc) #, urdweights=urdcrates) #emin=4., emax=12., phot_index=1.9)
    if make_detmap:
        emap = femap
        """
        bkgrates = {urdn: arttools.background.get_local_bkgrates(urdevt[urdn], urdbkg[urdn]) for urdn in arttools.telescope.URDNS}
        bkgrates = arttools.telescope.concat_data_in_order(bkgrates)
        """

        tit = arttools.source_detection.create_neighbouring_blocks_tasks(lwcs, emap, urdevt, attdata, urdbkg) #, photbkgrate=get_photbkg_rate)
        bs = arttools.source_detection.BlockEstimator(lwcs, mpnum=10)

        ctot, pmap = np.zeros(emap.shape, float), np.zeros(emap.shape, float)

        for x, y, c, th in tqdm.tqdm(bs.get_nphot_and_theta(tit)):
            ctot[x, y] = c
            pmap[x, y] = th

        """

        urdns = arttools.telescope.URDNS
        qlist = [Rotation(np.empty((0, 4), np.double)) if urdevt[urdn].size == 0 else arttools.orientation.get_events_quats(urdevt[urdn], urdn, attdata)*arttools._det_spatial.get_qcorr_for_urddata(urdevt[urdn]) for urdn in arttools.telescope.URDNS if urdn in urdevt]
        qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

        i, j = zip(*[arttools.psf.urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in arttools.telescope.URDNS if urdn in urdevt])
        i, j = np.concatenate(i), np.concatenate(j)

        eenergy = arttools.telescope.concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

        photprob = arttools.background.get_photon_vs_particle_prob(urdevt, urdweights=urdcrates)
        photprob = arttools.telescope.concat_data_in_order(photprob)

        vmap = arttools.psf.get_ipsf_interpolation_func()
        pkoef = photprob/bkgrates


        ije, sidx, ss, sc = arttools.psf.select_psf_groups(i, j, eenergy)
        tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(arttools.psf.unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]

        mask = emap > 1.
        sky = arttools.mosaic2.SkyImage(lwcs, vmap, mpnum=10)
        sky.set_mask(mask)
        ctot = gaussian_filter(img1.astype(float), 3.)*2.*pi*9.
        sky.set_action(arttools.mosaic2.get_source_photon_probability, join=True)
        for _ in range(25):
            sky.set_rmap(np.maximum(ctot, 1.)/np.maximum(emap, 1.), join=True)
            sky.rmap_convolve_multicore(tasks, total=len(tasks))
            mask = ctot < 0.5
            print("zero photons cts: ", mask.sum(), "conv hist", np.histogram(np.abs(ctot[~mask] - sky.img[~mask])/ctot[~mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 2., 10.]))
            ctot = np.copy(sky.img)
            sky.img[:, :] = 0.

        sky.set_action(arttools.mosaic2.get_zerosource_photstat, join=True)
        sky.set_rmap(ctot/np.maximum(emap, 1.), join=True)
        sky.img[:, :] = 0.
        sky.rmap_convolve_multicore(tasks, total=len(tasks))
        """

        prob = (pmap - ctot)/log(10.)
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img1, header=lwcs.to_header(), name='PHOT'),
                                                fits.ImageHDU(ctot/emap, header=lwcs.to_header(), name="rate"),
                                                fits.ImageHDU(prob, header=lwcs.to_header(), name="prob"),
                                                fits.ImageHDU(bmap, header=lwcs.to_header(), name="bmap"),
                                                fits.ImageHDU(femap, header=lwcs.to_header(), name="emap")]).writeto(outputname + ".fits.gz")
    else:
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img1, header=lwcs.to_header(), name='PHOT'),
                                        fits.ImageHDU(bmap, header=lwcs.to_header(), name="bmap"),
                                        fits.ImageHDU(femap, header=lwcs.to_header(), name="emap")]).writeto(outputname + ".fits.gz")



def make_lightcurve(flist, outputname, ra, dec, dt, usergti=tGTI, emin=4., emax=12., app=120., spec=None, join_sep=0.):
    timedel = float(dt)
    allfiles = [l.rstrip() for l in open(flist)]
    attfiles = [l for l in allfiles if "gyro.fits" in l]
    urdfiles = [l for l in allfiles if "urd.fits" == l[-8:]]

    srcvec = arttools.vector.pol_to_vec(*np.deg2rad([float(ra), float(dec)]).reshape((2, -1)))[0]

    attdata = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(gf)  for gf in attfiles])
    gti = attdata.circ_gti(srcvec, 18*60.)
    print(gti.arr)
    attdata = attdata.apply_gti((gti & usergti) + [-3, 3])


    for urdn in imgfilters:
        imgfilters[urdn]["TIME"] = attdata.gti & usergti
        imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([float(emin), float(emax)])

    print("check attdata", attdata.gti.exposure)

    urddata, urdhk = arttools.containers.read_urdfiles(urdfiles, filterslist={urdn:arttools.filters.IndependentFilters({"TIME": gti}) for urdn in URDNS}) #[f.replace("L0", "L1b") for f in urdfiles])
    for urdn in urddata:
        if not "ENERGY" in urddata[urdn].data.dtype.names:
            urddata[urdn] = arttools.energy.add_energies_and_grades(urddata[urdn], urdhk[urdn], arttools.caldb.get_energycal_by_urd(urdn), arttools.caldb.get_escale_by_urd(urdn))
        #urddata[urdn].data["TIME"] = urddata[urdn]["TIME"] + arttools.time.get_global_time(urddata[urdn]["TIME"], arttools.caldb.get_obt_timecorr_calib())
    print(urddata)
    print({urdn: d.filters["TIME"] for urdn, d in urddata.items()})


    #attadta = arttools.orientation.AttDATA(attdata.time + arttools.time.get_global_time(attdata.time, arttools.caldb.get_obt_timecorr_calib()), attdata(attdata.times), attdata.gti)

    bkgevts = {urdn: d.apply_filters(bkgfilters) for urdn, d in urddata.items()}
    if all([d.size == 0 for d in bkgevts.values()]):
        urdbkg = {urdn: arttools.lightcurve.Bkgrate(np.array([-np.inf, np.inf]), np.array([0.,])) for urdn in URDNS}
    else:
        urdbkg = arttools.background.get_background_lightcurve(np.sort(np.concatenate([b['TIME'] for b in bkgevts.values()])),
                                                           bkgevts, 1000., imgfilters)
    urddtc = {urdn: arttools.time.deadtime_correction(hk) for urdn, hk in urdhk.items()}

    urdevt = {urdn: d.apply_filters(imgfilters[urdn]) for urdn, d in urddata.items()}
    imgf = {urdn: d.filters for urdn, d in urdevt.items()}

    tgti = reduce(lambda a, b: a | b, [d.filters["TIME"] for d in urdevt.values()])
    te, gaps = tgti.arange(float(dt))

    if spec is None:
        te, dtn = arttools.expmap.make_exposures(srcvec, te, attdata, imgf, app=app, dtcorr=urddtc, urdweights=urdcrates)
    else:
        spec = interp1d(*np.loadtxt(spec).T, bounds_error=False, fill_value=0.)
        te, dtn = arttools.expmap.make_exposures(srcvec, te, attdata, imgf, app=app, dtcorr=urddtc, urdweights=urdcrates)
    gaps = gaps & (dtn > 0.)
    cs = np.diff(np.searchsorted(np.sort(np.concatenate([d["TIME"][np.sum(srcvec*arttools.orientation.get_photons_vectors(d, urdn, attdata), axis=1) > cos(app*pi/180./3600.)] for d in urdevt.values()])), te))[gaps]
    tc = (te[1:] + te[:-1])[gaps]/2.
    dta = np.diff(te)[gaps]
    lcs = arttools.background.get_bkg_lightcurve_for_app(urdbkg, imgf, attdata, srcvec, app, te, dtcorr=urddtc)[gaps] #{}, illum_filters=None)




    idx = arttools.mask.edges(dtn > 0.)
    tstart = te[idx[:, 0]]
    tstop = te[idx[:, 1]]
    print(tstart, tstop)
    tsobs = (Time(arttools.caldb.MJDREF, format="mjd") + TimeDelta(tgti.arr[0, 0], format="sec")).to_datetime()
    teobs = (Time(arttools.caldb.MJDREF, format="mjd") + TimeDelta(tgti.arr[-1, 1], format="sec")).to_datetime()

    dtn = dtn[gaps]

    idx0 = 0
    ts = te[:-1][gaps]
    tee = te[1:][gaps]
    csn = []
    lcsn = []
    ten = []
    dtnn = []
    tsn = []
    ten = []
    while idx0 < lcs.size:
        idx = np.searchsorted(ts, tee[idx0]+join_sep)
        csn.append(np.sum(cs[idx0:idx]))
        dtnn.append(np.sum(dtn[idx0:idx]))
        lcsn.append(np.sum(lcs[idx0:idx]))
        tsn.append(ts[idx0])
        ten.append(tee[idx-1])
        idx0 = idx

    print("finall check lcs", np.sum(lcs), np.sum(dtn), np.sum(cs), "binned data", np.sum(lcsn), np.sum(dtnn), np.sum(csn))
    print("timedel and time bins", timedel, dtn.min(), dtn.max(), np.min(dtnn), np.max(dtnn))

    cs = np.array(csn)
    dtn = np.array(dtnn)
    lcs = np.array(lcsn)
    ten = np.array(ten)
    tsn = np.array(tsn)

    dt = np.max(ten - tsn)
    tc = (ten + tsn)/2.
    print("tc", tc)
    pickle.dump(tc, open("/srg/a1/work/andrey/ART-XC/LMC/tc.pkl", "wb"))
    print("ten - tsn", ten - tsn)
    pickle.dump(ten - tsn, open("/srg/a1/work/andrey/ART-XC/LMC/tentsn.pkl", "wb"))
    print("cs", cs)
    pickle.dump(cs, open("/srg/a1/work/andrey/ART-XC/LMC/cs.pkl", "wb"))
    print("lcs", lcs)
    pickle.dump(lcs, open("/srg/a1/work/andrey/ART-XC/LMC/lcs.pkl", "wb"))
    print("dtn", dtn, timedel)
    pickle.dump(dtn, open("/srg/a1/work/andrey/ART-XC/LMC/dtn.pkl", "wb"))

    #d = Table.from_pandas(pandas.DataFrame({"TIME": tc, "TIMEDEL":dt, "COUNTS": cs, "BACKV": lcs, "FRACEXP": dtn[gaps]/dt, "RATE": np.maximum(cs - lcs, 0.)/dtn[gaps]}))
    pickle.dump([tc, (ten - tsn), cs, lcs, dtn/timedel, dtn/timedel, np.maximum(cs - lcs, 0.)/dtn, np.sqrt(cs)/dtn], open("/srg/a1/work/andrey/ART-XC/LMC_X-1/tmp.pkl", "wb"))

    d = QTable([tc*au.second, (ten - tsn)*au.second, cs*au.count, lcs*au.count, dtn/timedel, np.maximum(cs - lcs, 0.)/dtn*au.count/au.second, np.sqrt(cs)*au.count/dtn/au.second],
                 names = ["TIME", "TIMEDEL", "COUNT", "BACKV", "FRACEXP", "RATE", "ERROR"])
    phdu = fits.PrimaryHDU(header=fits.Header({"CONTENT":"LIGHT CURVE", "TELESCOP":"ART-XC", "INSTRIME":"T1-7", "TIMEVERS":"OGIP/93-003", "ORIGIN":"IKI", "DATE":datetime.datetime.today().strftime("%y/%m/%d"),
                                               "RA": ra, "DEC": dec, "EQUINOX": 2000.0, "RADECSYS": "FK5", "TIMEVERS": "OGIP/93-003", "AUTHOR": "", "TIMEDEL": timedel,
                                                "DATE-OBS": tsobs.strftime("%y/%m/%d"), "TIME-OBS":tsobs.strftime("%H:%M:%S"),
                                                "DATE-END": teobs.strftime("%y/%m/%d"), "TIME-END":teobs.strftime("%H:%M:%S"),
                                               }))

    header = fits.Header({"TELESCOP": "ART-XC", "OBS-DATE": tsobs.strftime("%y/%m/%d"), "RA": ra, "DEC": dec, "MJDREFI": int(arttools.caldb.MJDREF), "MJDREFF": arttools.caldb.MJDREF%1,
                          "TUNIT1": "s", "TUNIT2":"s", "TUNIT3":"COUNTS", "TUNIT4":"COUNTS", "TUNIT5":"", "TUNIT6":"COUNTS/s", "TUNIT7": "COUNTS/s", "GCOUNT": 1, "TIMESYS": "MJD", "TIMEUNIT": "S", "EMIN":emin, "EMAX": emax,
                          "TSTART": tgti.arr[0, 0], "TSTOP": tgti.arr[-1, 1], "ONTIME": gti.exposure, "TASSIGN": "SATELLITE","TIMEREF": "LOCAL", "CLOCKCOR":"NO", "BACKAPP": True, "DEADAPP": True, "VIGNAPP":True,
                          "TIMEUNIT": "s", "TIMEZERO": 0., "EUNIT": "keV", "TIMEDEL": timedel,
                          "HDUCLASS": "OGIP", "HDUCLAS1":"LIGHTCURVE", "HDUCLAS2":"TOTAL", "HDUCLAS3": "RATE", "TIMEDER": "GEOCENTER", "TIMVERS": "OGIP/93-003",
                          "DATE-OBS": tsobs.strftime("%y/%m/%d"), "TIME-OBS":tsobs.strftime("%H:%M:%S"),
                           "DATE-END": teobs.strftime("%y/%m/%d"), "TIME-END":teobs.strftime("%H:%M:%S"),
                          "DATE": datetime.datetime.today().strftime("%y/%m/%d"),
                          "AUTHOR": "", "OBJECT": "",
                          "ORIGIN":"IKI"})


    dhdu = fits.BinTableHDU(header=header, data=d, name="RATE")
    ghdu = fits.BinTableHDU(header=fits.Header({"MJDREFI": int(arttools.caldb.MJDREF), "MJDREFF": arttools.caldb.MJDREF%1, "TIMEZERO":0.}), data=QTable([tstart*au.second, tstop*au.second], names=["START", "STOP"]), name="GTI")
    fits.HDUList([phdu, dhdu, ghdu]).writeto(outputname + ".fits.gz", overwrite=True)



def make_spec_and_arf(flist, outname, ra, dec, usergti=None):
    allfiles = [l.rstrip() for l in open(flist)]
    attfiles = [l for l in allfiles if "gyro.fits" in l]
    urdfiles = [l for l in allfiles if "urd.fits" == l[-8:]]

    srcvec = arttools.vector.pol_to_vec(*np.deg2rad([float(ra), float(dec)]).reshape((2, -1)))[0]

    attdata = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(gf) for gf in attfiles])
    gti = attdata.circ_gti(srcvec, 20*60.)
    attdata = attdata.apply_gti((gti & usergti) + [-3, 3])

    for urdn in imgfilters:
        imgfilters[urdn]["TIME"] = attdata.gti & usergti
        imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([4., 65.])

    urddata, urdhk = arttools.containers.read_urdfiles(urdfiles, {urdn: arttools.filters.IndependentFilters({"TIME": attdata.gti}) for urdn in arttools.telescope.URDNS}) #[f.replace("L0", "L1b") for f in urdfiles])

    urddtc = {urdn: arttools.time.deadtime_correction(hk) for urdn, hk in urdhk.items()}
    bkgevts = {urdn: d.apply_filters(bkgfilters) for urdn, d in urddata.items()}

    bkgtimes = np.sort(np.concatenate([d["TIME"] for d in urddata.values()]))
    urdbkg = arttools.background.get_background_lightcurve(bkgtimes, {urdn: d.filters for urdn, d in bkgevts.items()}, 1000., imgfilters, dtcorr=urddtc)

    tgti = reduce(lambda a, b: a|b, [f["TIME"] for f in imgfilters.values()])

    arf = arttools.arf.make_correcrted_arf(attdata, srcvec, imgfilters, urddtc)
    arf.writeto(outname + ".arf")

    eeo = np.concatenate([arf[1].data["ENERG_LO"], arf[1].data["ENERG_HI"][-1:]])
    bspec = arttools.background.get_bkg_spec(urdbkg, imgfilters, attdata, srcvec, 120.) #, urddtc)
    bspec = interp1d(bspec[0], np.concatenate([[0.,], bspec[2]]).cumsum(), bounds_error=False, fill_value=(0., bspec[2].sum()))
    bspec = np.diff(bspec(eeo))


    spec = np.zeros(eeo.size - 1, int)
    for urdn in urddata:
        d = urddata[urdn].apply_filters(imgfilters[urdn])
        vecs = arttools.orientation.get_photons_vectors(data, urdn, attdata)
        spec += np.histogram(d["ENERGY"][np.sum(vecs*srcvec, axis=1) > cos(120.*pi/180./3600)], eeo)

    pickle.dump([bspec, spec], open("%s_specdata.pkl" % outname, "wb"))

    """
    hdu.header["INSTRUM"] = "TEL [1-7]"
    hdu.header["TELESCOP"] = "ART-XC"
    hdu.header["HDUCLAS3"] = "COUNT"
    hdu.header["TUNIT1"] = ""
    hdu.header["TUNIT2"] = "count"
    hdu.header["TUNIT2"] = "count"
    hdu.header["RA-OBJ"] = ra
    hdu.header["DEC-OBJ"] = dec
    hdu.header["OBJECT"] = srcname
    hdu.header["POISSERR"] = True
    hdu.header["ARFFILE"] = arf.filename()
    hdu.header["RESPFILE"] = "artxc_rmf_v000.fits"
    hdu.header["EXPOSURE"] = tgti.exposure
    fits.HDUList([pfile[0], hdu, fits.BinTableHDU(data=Table.from_pandas(pandas.DataFrame({"TSTART": gloc.arr[:, 0], "TSTOP":gloc.arr[:, 1]})), name="GTI")]).writeto("%s.pha" % outname, overwrite=True)
    """

def estimate_rate(flist, outname, ra, dec, usergti=None, emin=4., emax=12., app=120.):
    allfiles = [l.rstrip() for l in open(flist)]
    attfiles = [l for l in allfiles if "gyro.fits" in l]
    urdfiles = [l for l in allfiles if "urd.fits" == l[-8:]]

    srcvec = arttools.vector.pol_to_vec(*np.deg2rad([float(ra), float(dec)]).reshape((2, -1)))[0]

    attdata = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(gf) for gf in attfiles])
    gti = attdata.circ_gti(srcvec, 22*60.)
    attdata = attdata.apply_gti((gti & usergti) + [-3, 3])

    report = "ra, dec: %.6f %.6f" % (ra, dec)
    report += "\n" + "files:" + "\n".join(urdfiles)
    report += "\n" + "GTI:" + "\n".join(["%f, %f" % (s, e) for s, e in attdata.gti])


    for urdn in imgfilters:
        imgfilters[urdn]["TIME"] = attdata.gti & usergti
        imgfilters[urdn]["ENERGY"] = arttools.filters.Intervals([float(emin), float(emax)])

    print("check attdata", attdata.gti.exposure)

    urddata, urdhk = arttools.containers.read_urdfiles(urdfiles, filterslist={urdn:arttools.filters.IndependentFilters({"TIME": gti}) for urdn in URDNS}) #[f.replace("L0", "L1b") for f in urdfiles])
    for urdn in urddata:
        if not "ENERGY" in urddata[urdn].data.dtype.names:
            urddata[urdn] = arttools.energy.add_energies_and_grades(urddata[urdn], urdhk[urdn], arttools.caldb.get_energycal_by_urd(urdn), arttools.caldb.get_escale_by_urd(urdn))

    bkgtimes = np.sort(np.concatenate([d["TIME"][bkgfilters.apply(d)] for d in urddata.values()]))
    urdbkg = arttools.background.get_background_lightcurve(bkgtimes, {urdn: d.filters["TIME"] for urdn, d in urddata.items()}, bkgfilters, 1000., imgfilters)
    urddtc = {urdn: arttools.time.deadtime_correction(hk) for urdn, hk in urdhk.items()}

    urdevt = {urdn: d.apply_filters(imgfilters[urdn]) for urdn, d in urddata.items()}
    imgf = {urdn: d.filters for urdn, d in urdevt.items()}

    tgti = reduce(lambda a, b: a | b, [d.filters["TIME"] for d in urdevt.values()])
    te, gaps = tgti.arange(0.1)

    te, dtn = arttools.expmap.make_exposures(srcvec, te, attdata, imgf, app=app, dtcorr=urddtc, urdweights=urdcrates)
    gaps = gaps & (dtn > 0.)
    dtn = dtn[gaps]
    urde = {urdn: d.data[np.sum(srcvec*arttools.orientation.get_photons_vectors(d, urdn, attdata), axis=1) > cos(app*pi/180./3600.)] for urdn, d in urdevt.items()}
    cs = np.diff(np.searchsorted(np.sort(np.concatenate([d["TIME"] for d in urde.values()])), te))[gaps]
    tc = (te[1:] + te[:-1])[gaps]/2.
    dta = np.diff(te)[gaps]
    lcs = arttools.background.get_bkg_lightcurve_for_app(urdbkg, imgf, attdata, srcvec, app, te, dtcorr=urddtc)[gaps] #{}, illum_filters=None)

    report += "\n totevents: %d, exp: %.3f, vexp: %.3f exp bkg: %.3f" % (cs.sum(), dta.sum(), dtn.sum(), lcs.sum())

    mrate = minimize(lambda rate: -poisson.logpmf(cs, rate*dtn + lcs).sum(), [max(cs.sum() - lcs.sum(), 1)/dtn.sum(),], method="Nelder-Mead")

    with pymc3.Model() as mo:
        rate = pymc3.Uniform("rate", lower=0., upper=1000) #min(abs(mrate.x[0])*4.5, 30))
        obs = pymc3.Poisson("obs", mu=rate*dtn + lcs, observed=cs)
        tracelc = pymc3.sample(4096)



    cl, ch = mprob_quantiles(tracelc["rate"], mrate.x[0])[0]
    report += "\nrate estimation from lcs:\n %.2e (%.2e, %.2e)" % (mrate.x[0]*4.21e-11, cl*4.21e-11, ch*4.21e-11)

    if np.sum(cs) > 0:

        qlist = [arttools.orientation.get_events_quats(urde[urdn], urdn, attdata)*arttools._det_spatial.get_qcorr_for_urddata(urde[urdn]) for urdn in arttools.telescope.URDNS if urdn in urde and urde[urdn].size > 0]
        qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

        i, j = zip(*[arttools.psf.urddata_to_opaxoffset(urde[urdn], urdn) for urdn in arttools.telescope.URDNS if urdn in urde])
        i, j = np.concatenate(i), np.concatenate(j)

        imgdata = arttools.telescope.concat_data_in_order(urde)
        eenergy = arttools.telescope.concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urde.items()})

        bkgrates = {urdn: arttools.background.get_local_bkgrates(urdn, urdbkg[urdn], imgfilters[urdn], urde[urdn]) for urdn in arttools.telescope.URDNS}
        bkgrates = arttools.telescope.concat_data_in_order(bkgrates)

        photprob = arttools.background.get_photon_vs_particle_prob(urde, urdweights=urdcrates)
        photprob = arttools.telescope.concat_data_in_order(photprob)

        pool = Pool(processes=10, initializer=init_ipsf_weight_pool)
        w = pool.map(get_ipsf_weight, zip(i, j, eenergy, repeat(srcvec), qlist))
        pool.close()
    else:
        i, j, w, photprob, bkgrates = np.empty((5, 0), float)

    slr = np.ones(i.size + 1, np.double)
    blr = np.ones(i.size + 1, np.double)
    blr[0] = lcs.sum()
    slr[0] = dtn.sum()
    slr[1:] = w*photprob
    blr[1:] = bkgrates
    mrate = minimize(lambda rate: slr[0]*rate + np.sum(np.log(blr[1:]/(slr[1:]*rate + blr[1:]))), [max(i.size - lcs.sum(), 1)/slr[0],], method="Nelder-Mead")

    with pymc3.Model() as mo:
        rate = pymc3.Uniform("rate", lower=0., upper=1000) #min(abs(mrate.x[0])*4.5, 30))
        o = np.ones(slr.size, np.int)
        o[0] = 0
        o, blr, slr = o[slr > 0], blr[slr > 0], slr[slr > 0]
        obs = pymc3.Poisson("obs", mu=blr + slr*rate, observed=o)
        traceph = pymc3.sample(4096)

    cl, ch = mprob_quantiles(traceph["rate"], mrate.x[0])[0]
    report += "\nrate estimation from psf:\n %.2e (%.2e, %.2e)" % (mrate.x[0]*4.21e-11, cl*4.21e-11, ch*4.21e-11)
    open(outname, "w").write(report)


def get_all_orig_data(output, ra, dec):
    att, times, names = pickle.load(open("/srg/work/andrey/ART-XC/att_compressed/allgyro_arch.pkl", "rb"))
    srcvec = arttools.vector.pol_to_vec(*np.deg2rad([float(ra), float(dec)]).reshape((2, -1)))[0]
    gti = att.circ_gti(srcvec, 22.*60) + [-30, 30]
    te, gaps = gti.make_tedges(times)
    idx = np.unique(np.searchsorted(times, (te[1:] + te[:-1])[gaps]/2.)) - 1
    allflist = []
    with open(output, "w") as f:
        for uname in np.unique([names[i] for i in idx]):
            dpath = os.path.dirname(uname)
            f.write("\n".join([os.path.join(dpath, name) for name in os.listdir(dpath)]))


process_help = \
"""
help:
    arttools/processes is a set of simple tools, which can be started through the command line
    this tools are gathered in a single collection due to the similarity of accepted arguments, mainly
    input=file or @flist
    gti=usergtifile
    output=output filename

    folllowing actions are woring now:

    =====================================================================================
        obsgtis: check which nonintersection observation are present in the provided att files
            att files are provided as a list of files in the single text files, entered in input starting with @, or a single att file
            mandatory arguments: ----
            possible arguments: gti

            this action returns a ascii file, containing centers and areas of separate observations and corresponding gti intervals in the form of pair of start and stop times in onboard time in seconds

    =====================================================================================
        image: produce an image for the provided set of data files
            mandatory arguments: ----
            possible arguments: emin, emax, gti

            this action produces a single fits file containg three image extention - pure photon map, consisting of events arived in the specified energy range (having grades 0-10),
            exposure map for this energy  band and expected instrumental background estimated based on the caldb background spectrum

    =====================================================================================
        lightcurve: produces lightcurve from the data for the specified position on sky
            mandatory arguments: ra, dec
            possible arguments: rapp, emin, emax, gti, dt

            this procedure produces fits file, containing lightcurve in the OGIP format.
"""

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=process_help, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("action", help="what u wanna do???? oprions: make_spec make_imgs obsgtis", choices=["spec", "obsgtis", "image", "lightcurve", "spec_std", "estimate_rate", "get_dfiles"])
    parser.add_argument("input", help="name of the input file or file containing list of input files, starting with @")
    parser.add_argument("output", help="root of the output file name, if several output files produced they all will have root in their names")
    parser.add_argument("--ra", help="ra coordinate for the ananlysis purposes", default=None, required=False, type=float)
    parser.add_argument("--dec", help="dec coordinate for the ananlysis purposes", default=None, required=False, type=float)
    parser.add_argument("--gti", help="custom gti for the analysis", default=None, required=False)
    parser.add_argument("--emin", help="lower edge of the working energy band for the product", default=4., required=False)
    parser.add_argument("--emax", help="upper edge of the working energy band for the product", default=12., required=False)
    parser.add_argument("--rapp", help="aperture size for analyzis in arcsec", default=120., required=False)
    parser.add_argument("--dt", help="timebin size for analyzis in seconds", default=1., required=False)
    parser.add_argument("--make_detmap", help="do you need to compute detection probability map", default=False, required=False)
    parser.add_argument("--spec", help="compute vignetting for specific spectral shape", default=None, required=False)
    parser.add_argument("--join_sep", help="if producing lightcurves with gaps, defined the longest time bin in binning separated observations", default=0., required=False, type=float)
    parser.add_argument("--survey", help="performa analysis over survey data for specified surveys (syntax 1 - first survey, 2,3 - second and third survey e.t.c)", default="1,2,3,4", required=False)

    parsed = parser.parse_args(sys.argv[1:])
    gti = tGTI if parsed.gti is None else GTI(np.copy(np.loadtxt(parsed.gti).reshape((-1, 2))))

    if parsed.action == "spec":
        make_spec(parsed.ra, parsed.dec, survey=parsed.survey)
    if parsed.action == "obsgtis":
        fname = parsed.input
        if "@" == fname[0]:
            flist = [l.rstrip() for l in open(fname[1:])]
        else:
            flist = [fname,]
        attdata = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(fname) for fname in flist if "gyro.fits" in fname])
        gtis, areas, centers, chulls = arttools.orientation.get_observations_gti(attdata)
        with open(parsed.output, 'w') as f:
            for g, a, c in zip(gtis, areas, centers):
                f.write("center: %.6f %.6f" % tuple(c) + " area %.2f \n" % a)
                for arr in g.arr:
                    f.write("\t%f %f\n" % tuple(arr))

    if parsed.action == "image":
        make_img(parsed.input[1:], parsed.output, gti, emin=parsed.emin, emax=parsed.emax, make_detmap=parsed.make_detmap, ra=parsed.ra, dec=parsed.dec)

    if parsed.action == "lightcurve":
        make_lightcurve(parsed.input[1:], parsed.output, ra=parsed.ra, dec=parsed.dec, usergti= gti, emin=parsed.emin, emax=parsed.emax, app=parsed.rapp, dt=parsed.dt, spec=parsed.spec, join_sep=parsed.join_sep)

    if parsed.action == "spec_std":
        make_spec_and_arf(parsed.input[1:], parsed.output, ra=parsed.ra, dec=parsed.dec, usergti=gti)

    if parsed.action == "estimate_rate":
        estimate_rate(parsed.input[1:], parsed.output, ra=parsed.ra, dec=parsed.dec, usergti=gti, emin=parsed.emin, emax=parsed.emax, app=parsed.rapp)

    if parsed.action == "get_dfiles":
        get_all_orig_data(parsed.output, ra=parsed.ra, dec=parsed.dec)
