import numpy as np
import matplotlib.pyplot as plt
import os
import arttools
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import healpy as hp
import os, shutil, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--list", help="input list of L1b files")
parser.add_argument("--mapname", help="name of map to create, default 'artxc_survmap.fits.gz'", default='artxc_survmap.fits.gz')
args = parser.parse_args()
    

inputlist    = args.list
if inputlist==None or not inputlist:
    print ('>>ERROR>> Please, provide list of input L1b files')
    exit(1)
else:
    print ('>>>>>>>>> input list:'+inputlist)
mapname = args.mapname
if mapname==None or not mapname:
    print ('>>>>>>>>>  will create artxc_survmap.fits.gz')
else:
    print ('>>>>>>>>> will create '+mapname)



inp = open(inputlist, 'r')
fnames = inp.readlines()
inp.close()

NSIDE = 1024*2
NPIX = hp.nside2npix(NSIDE)

sky = np.zeros(NPIX)

for fname in fnames[:]:
    print('adding file: ',fname.strip())
    urdfile = fits.open(fname.strip())
    urddata = urdfile[1].data
    energy  =  urddata['ENERGY']
    grades  =  urddata['GRADE']
    flags   =  urddata['FLAG']
    mask = np.bitwise_and(energy>=4, energy<=11.)
    gmask = np.bitwise_and(np.bitwise_and(grades>=0, grades<=9.), flags==0) 
    mask = np.bitwise_and(mask, gmask)
    ra, dec = urddata['RA'][mask], urddata['DEC'][mask]
    coords = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs', unit='deg')
    l ,b   = coords.galactic.l.degree, coords.galactic.b.degree
    pix = hp.ang2pix(NSIDE, l, b, lonlat=True)
    for p in pix:
        sky[p]+=1
hp.write_map(mapname, sky, overwrite=True) 
