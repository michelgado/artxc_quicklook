from arttools.plot import make_mosaic_for_urdset_by_gti, get_attdata, make_sky_image, make_energies_flags_and_grades
from arttools.orientation import read_bokz_fits, AttDATA
from arttools.planwcs import make_wcs_for_attdata, split_survey_mode
from arttools.time import tGTI, get_gti, GTI, emptyGTI, deadtime_correction
from arttools.expmap import make_expmap_for_wcs
from arttools.background import make_bkgmap_for_wcs
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
from scipy.interpolate import interp1d
from copy import copy
from scipy.spatial.transform import Rotation
from multiprocessing import Pool, Barrier
from scipy.optimize import minimize, root
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.special import gamma, gammainc
import time
import pymc3
from astropy.table import Table
import pandas
from astropy.time import Time, TimeDelta


vmap = None

surveys, surveygtis = None, None
tstart, fnames, atts = None, None, None

urdcrates = arttools.caldb.get_telescope_crabrates()
cr = np.sum([v for v in urdcrates.values()])
urdcrates = {urdn: d/cr for urdn, d in urdcrates.items()}

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
        g1 = att.get_axis_movement_speed_gti(lambda x: (x < pi/180.*110/3600) & (x > pi/180.*75./3600.)) & ~atts.gti & ~gnew
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


def init_ipsf_weight_pool():
    global vmap
    vmap = arttools.psf.get_ipsf_interpolation_func()

def get_ipsf_weight(args):
    i, j, e, ax, q = args
    vmap.values = arttools.psf.unpack_inverse_psf_ayut(i, j, e)
    return vmap(arttools._det_spatial.vec_to_offset(q.apply(ax, inverse=True)))


def make_spec(ra, dec, survey=None):


    imgfilters = {urdn: arttools.filters.IndependentFilters({"ENERGY": arttools.interval.Intervals([4., 30.]),
                                                            "GRADE": arttools.filters.RationalSet(range(10)),
                                                            "RAWXY": arttools.filters.get_shadowmask_filter(urdn)}) for urdn in arttools.telescope.URDNS}

    bkgfilters = arttools.filters.IndependentFilters({"ENERGY": arttools.interval.Intervals([40., 100.]),
                                                    "RAW_X": arttools.filters.InversedRationalSet([0, 47]),
                                                    "RAW_Y": arttools.filters.InversedRationalSet([0, 47]),
                                                    "GRADE": arttools.filters.InversedRationalSet([])})


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



        if os.path.exists("%s_evt.pkl" % fname):
            att, urdgti, bkgtimes, urdevtt, bkggti = pickle.load(open("%s_evt.pkl" % fname, "rb"))
        else:
            dirs = [survfnames[i] for i in np.unique(np.searchsorted(survtstart, gloc.arr.ravel())) - 1]

            print(dirs)

            attfiles = []
            urdfiles = []
            for d in dirs:
                attfiles += [os.path.join(d, "L0", l) for l in os.listdir(os.path.join(d, "L0")) if "gyro.fits" in l]
                urdfiles += [os.path.join(d, "L1b", l) for l in os.listdir(os.path.join(d, "L1b")) if "urd.fits" == l[-8:]]

            att = None
            att = arttools.orientation.AttDATA.concatenate([arttools.orientation.get_attdata(l) for l in attfiles])
            gti = att.circ_gti(ax, 25*60.) & gloc
            print("located files", dirs)
            print("used gti", gti)

            bdata = []
            bkggti = {}
            urdevt = {}

            for urdn in imgfilters:
                imgfilters[urdn]["ENERGY"] = arttools.interval.Intervals([4., 30.])

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
                imgfilters4_12[urdn]["ENERGY"] = arttools.interval.Intervals([4., 12.])

            urdevt = {urdn: urdevtt[urdn][imgfilters4_12[urdn].apply(urdevtt[urdn])] for urdn in urdevtt}
            print("4-12 events", {urdn: d.size for urdn, d in urdevt.items()})

            tgti = reduce(lambda a, b: a | b, urdgti.values())
            locwcs = arttools.planwcs.make_wcs_for_attdata(att, gti=tgti, pixsize=5./3600.)
            locwcs.wcs.crpix = [31, 31]


            qlist = [arttools.orientation.get_events_quats(urdevt[urdn], urdn, att)*arttools._det_spatial.get_qcorr_for_urddata(urdevt[urdn]) for urdn in arttools.telescope.URDNS if urdn in urdevt and urdevt[urdn].size > 0]
            qlist = Rotation.from_quat(np.concatenate([q.as_quat() for q in qlist], axis=0))

            i, j = zip(*[arttools.psf.urddata_to_opaxoffset(urdevt[urdn], urdn) for urdn in arttools.telescope.URDNS if urdn in urdevt])
            i, j = np.concatenate(i), np.concatenate(j)

            imgdata = arttools.telescope.concat_data_in_order(urdevt)
            eenergy = arttools.telescope.concat_data_in_order({urdn: d["ENERGY"] for urdn, d in urdevt.items()})

            urdbkg = arttools.background.get_background_lightcurve(bkgtimes, bkggti, bkgfilters, 2000., imgfilters4_12)

            bkgrates = {urdn: arttools.background.get_local_bkgrates(urdn, urdbkg[urdn], imgfilters4_12[urdn], urdevt[urdn]) for urdn in arttools.telescope.URDNS}
            bkgrates = arttools.telescope.concat_data_in_order(bkgrates)

            photprob = arttools.background.get_photon_vs_particle_prob(imgfilters4_12, urdevt, urdweights=urdcrates)
            photprob = arttools.telescope.concat_data_in_order(photprob)

            vmapl = arttools.psf.get_ipsf_interpolation_func()
            pkoef = photprob/bkgrates
            print("sky initialized")

            ije, sidx, ss, sc = arttools.psf.select_psf_grups(i, j, eenergy)

            sky = arttools.mosaic2.SkyImage(locwcs, vmapl, mpnum=30)
            emap = arttools.expmap.make_expmap_for_wcs(locwcs, att, urdgti, imgfilters4_12, urdweights=urdcrates) #, urdweights=urdcrates) #emin=4., emax=12., phot_index=1.9)
            tasks = [(qlist[sidx[s:s+c]], pkoef[sidx[s:s+c]], np.copy(arttools.psf.unpack_inverse_psf_ayut(ic, jc)[eidx])) for (ic, jc, eidx), s, c in zip(ije.T, ss, sc)]

            mask = emap > 1.
            sky.set_mask(mask)
            ctot = np.ones(mask.shape, np.double)

            sky.set_action(arttools.mosaic2.get_source_photon_probability)
            sky.set_rmap(ctot/np.maximum(emap, 1.))
            for _ in range(21):
                sky.clean_img()
                sky.rmap_convolve_multicore(tasks)
                sky.accumulate_img()
                mask = ctot < 0.5
                print("zero photons cts: ", mask.sum(), "conv hist", np.histogram(np.abs(ctot[~mask] - sky.img[~mask])/ctot[~mask], [0., 1e-3, 1e-2, 5e-2, 0.1, 0.5, 1., 2., 10.]))
                sky.set_rmap(sky.img/np.maximum(emap, 1.))
                ctot = np.copy(sky.img)
                sky.img[:, :] = 0.
            sky.clean_img()
            sky.set_action(arttools.mosaic2.get_zerosource_photstat)
            sky.set_rmap(ctot/np.maximum(emap, 1.))
            sky.img[:, :] = 0.
            sky.rmap_convolve_multicore(tasks)
            sky.accumulate_img()
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
                imgfilters[urdn]["ENERGY"] = arttools.interval.Intervals([el, eh])
                urdgti[urdn] = urdgti[urdn] & att.get_axis_movement_speed_gti(lambda x: x > 70.*pi/180./3600.)
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

                photprob = arttools.background.get_photon_vs_particle_prob(imgfilters, urdevt, urdweights=urdcrates)
                photprob = arttools.telescope.concat_data_in_order(photprob)

                for urdn in imgfilters:
                    imgfilters[urdn]["ENERGY"] = arttools.interval.Intervals([el, eh])

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
    gti = attdata.get_axis_movement_speed_gti(lambda x: (x > pi/180.*60./3600) & (x < pi/180.*120./3600.))
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


"""

def run(fpath):
    os.chdir(fpath)
    abspath = os.path.abspath(".")
    neighbours = get_neighbours(abspath)
    print(neighbours)

    allfiles = os.listdir("L0")
    bokzfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "bokz.fits" in l]
    gyrofiles = [os.path.join(abspath, "L0", l) for l in allfiles if "gyro.fits" in l]
    urdfiles = [os.path.join(abspath, "L0", l) for l in allfiles if "urd.fits" in l]

    attgti = reduce(lambda a, b: a | b, [get_gti(fits.open(urdfile)) for urdfile in urdfiles])
    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    locwcs = make_wcs_for_attdata(attdata, attgti)

    gyrofiles = []
    urdfiles = []
    print("current locations", os.path.abspath("."))
    for neighbour in neighbours:
        abspath = os.path.abspath(os.path.join("../", neighbour, "L0"))
        print("search in cell", abspath)
        if not os.path.exists(abspath):
            print("not filled yes, skip")
            continue
        allfiles = os.listdir(abspath)
        gyrofiles += [os.path.join(abspath, fname) for fname in allfiles if "gyro" in fname]
        urdfiles += [os.path.join(abspath, fname) for fname in allfiles if "urd" in fname]

    try:
        os.mkdir("L3")
    except OSError as exc:
        pass
    os.chdir("L3")

    attdata = AttDATA.concatenate([get_attdata(fname) for fname in set(gyrofiles)])
    print(attdata.gti)

    xsize, ysize = int(locwcs.wcs.crpix[0]*2 + 1), int(locwcs.wcs.crpix[1]*2 + 1)
    imgdata = np.zeros((ysize, xsize), np.double)
    urdgti = {URDN:emptyGTI for URDN in URDNS}
    urdhk = {}
    urdbkg = {}
    urdbkge = {}

    for urdfname in urdfiles:
        urdfile = fits.open(urdfname)
        urdn = urdfile["EVENTS"].header["URDN"]
        print("processing:", urdfname)
        print("overall urd exposure", get_gti(urdfile).exposure)
        locgti = get_gti(urdfile) & attdata.gti & -urdgti.get(urdn, emptyGTI)
        locgti.merge_joint()
        print("exposure in GTI:", locgti.exposure)
        if locgti.exposure == 0.:
            continue
        urdgti[urdn] = urdgti.get(urdn, emptyGTI) | locgti

        urddata = np.copy(urdfile["EVENTS"].data) #hint: do not apply bool mask to a fitsrec - it's a stright way to the memory leak :)
        urddata = urddata[locgti.mask_outofgti_times(urddata["TIME"])]

        hkdata = np.copy(urdfile["HK"].data)
        hkdata = hkdata[(locgti + [-30, 30]).mask_outofgti_times(hkdata["TIME"])]
        urdhk[urdn] = urdhk.get(urdn, []) + [hkdata,]

        energy, grade, flag = make_energies_flags_and_grades(urddata, hkdata, urdn)
        pickimg = np.all([energy > 4., energy < 11.2, grade > -1, grade < 10, flag == 0], axis=0)
        timg = make_sky_image(urddata[pickimg], urdn, attdata, locwcs, 1)
        imgdata += timg

        pickbkg = np.all([energy > 40., energy < 100., grade > -1, grade < 10, flag < 3], axis=0)
        bkgevts = urddata["TIME"][pickbkg]
        urdbkge[urdn] = urdbkge.get(urdn, []) + [bkgevts,]

    for urdn in urdgti:
        print(urdn, urdgti[urdn].exposure)

    img = fits.PrimaryHDU(header=locwcs.to_header(), data=imgdata)
    img.writeto("cmap.fits.gz", overwrite=True)

    urdhk = {urdn:np.unique(np.concatenate(hklist)) for urdn, hklist in urdhk.items()}
    urddtc = {urdn: deadtime_correction(hk) for urdn, hk in urdhk.items()}
    tgti = reduce(lambda a, b: a & b, urdgti.values())
    te = np.concatenate([np.linspace(s, e, int((e-s)//100.) + 2) for s, e in tgti.arr])
    mgaps = np.ones(te.size - 1, np.bool)
    if tgti.arr.size > 2:
        mgaps[np.cumsum([(int((e-s)//100.) + 2) for s, e in tgti.arr[:-1]]) - 1] = False
        mgaps[te[1:] - te[:-1] < 10] = False

    tevts = np.sort(np.concatenate([np.concatenate(e) for e in urdbkge.values()]))
    rate = tevts.searchsorted(te)
    rate = (rate[1:] - rate[:-1])[mgaps]/(te[1:] - te[:-1])[mgaps]
    tc = (te[1:] + te[:-1])[mgaps]/2.
    tm = np.sum(tgti.mask_outofgti_times(tevts))/tgti.exposure

    urdbkg = {urdn: interp1d(tc, rate*urdbkgsc[urdn]/7.61, bounds_error=False, fill_value=tm*urdbkgsc[urdn]/7.62) for urdn in urdbkgsc}

    emap = make_expmap_for_wcs(locwcs, attdata, urdgti, dtcorr=urddtc)
    emap = fits.PrimaryHDU(data=emap, header=locwcs.to_header())
    emap.writeto("emap.fits.gz", overwrite=True)
    bmap = make_bkgmap_for_wcs(locwcs, attdata, urdgti, time_corr=urdbkg)
    bmap = fits.PrimaryHDU(data=bmap, header=locwcs.to_header())
    bmap.writeto("bmap.fits.gz", overwrite=True)

"""

if __name__ == "__main__":
    print("how to use: \n spec ra dec [survey]  # will produce spectrum of the source in the curreny directory")
    if sys.argv[1] == "spec":
        make_spec(*sys.argv[2:])
