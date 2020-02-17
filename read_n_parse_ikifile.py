#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 14:33:22 2020

@author: hart
"""

from arttools.orientation import quat_to_pol_and_roll
from scipy.spatial.transform import Rotation
from astropy.time import Time
from astropy.io import fits
import numpy as np


def time_to_board(time):
    REFTIME = Time(51543.875, format='mjd')
    return (time - REFTIME).sec - 5


ra,dec = [],[]
ts     = []
planned = ['20200210_220000_11000300300.iki', '20200214_220000_11000400100.iki','20200218_220000_11000400200.iki']

for file in planned:
    infile = open(file, 'r')
    data = infile.readlines()
    for line in data[::10]:
        L = line.split()
        quat = Rotation.from_quat([float(L[5]), float(L[2]), float(L[3]), float(L[4])])
        vec= quat.apply([1, 0, 0])
        ts.append(time_to_board(Time(' '.join(L[0:2]).replace('.','-'), format='iso'))-3600*3)
        dec.append(np.rad2deg(np.arctan(vec[2]/np.sqrt(vec[1]**2 + vec[0]**2))))
        ra.append(np.rad2deg((-np.arctan2(vec[1], vec[0])-np.pi)%(2.*np.pi)))
    infile.close()
#
outf = open('planned_pointing.dat', 'w')

outf.write('#Planned SRG pointings, discrepancy could be high\n')
outf.write('#Up to 22-02-2020\n')
           
for t,r,d in zip(ts, ra, dec):
    outf.write(f'{t} {r} {d}\n')
outf.close()


#
#hdul = fits.open('srg_20200209_154246_000_gyro_att.fits')
#t_g, ra_g, dec_g = hdul[1].data['TIME'], hdul[1].data['RA'], hdul[1].data['Dec']
#import matplotlib.pyplot as plt
#plt.figure()
#plt.ylabel('RA, degree')
#plt.plot(ts, ra , 'rx')
#plt.plot(t_g, ra_g , 'g.')
#plt.show()
#
#plt.figure()
#plt.ylabel('Dec, degree')
#plt.plot(ts, dec , 'rx')
#plt.plot(t_g, dec_g , 'g.')
#plt.show()