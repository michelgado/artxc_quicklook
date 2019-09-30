#!/opt/soft/psoft/bin/python3
# -*- coding: utf8 -*-
'''

                                  _.---"'"""""'`--.._
                             _,.-'                   `-._
                         _,."                            -.
                     .-""   ___...---------.._             `.
                     `---'""                  `-.            `.
                                                 `.            \
                                                   `.           \
                                                     \           \
                                                      .           \
   QUICKLOOK?                                         |            .
                                                      |            |
                                _________             |            |
                          _,.-'"         `"'-.._      :            |
                      _,-'                      `-._.'             |
                   _.'                              `.             '
        _.-.    _,+......__                           `.          .
      .'    `-"'           `"-.,-""--._                 \        /
     /    ,'                  |    __  \                 \      /
    `   ..                       +"  )  \                 \    /
     `.'  \          ,-"`-..    |       |                  \  /
      / " |        .'       \   '.    _.'                   .'
     |,.."--"""--..|    "    |    `""`.                     |
   ,"               `-._     |        |                     |
 .'                     `-._+         |                     |
/                           `.                        /     |
|    `     '                  |                      /      |
`-.....--.__                  |              |      /       |
   `./ "| / `-.........--.-   '              |    ,'        '
     /| ||        `.'  ,'   .'               |_,-+         /
    / ' '.`.        _,'   ,'     `.          |   '   _,.. /
   /   `.  `"'"'""'"   _,^--------"`.        |    `.'_  _/
  /... _.`:.________,.'              `._,.-..|        "'
 `.__.'                                 `._  /
                                           "' mh

picture from: https://www.fiikus.net/asciiart/pokemon/079.txt



artquicklook_l1

General description:
    This task produce several elementary products in order to 
    characterize the condition of ART-XC detectors, their health,
    and also the quality of data
    

    1) it generates lightcurves from non-illuminated parts of ART-XC detectors 
        in order to provide estimate of particle background.

    2) plot histograms of detector count, so operator
        can easily detect 'hot' or 'dead' strips
        and detector spectrums

    3) plot dependence of angular speed vs time


INPUT:
    cleaned event files
    attitude file
    CALDB files
    
OUTPUT:
    pdf file with report
    
AIMS:
    1) 


ALGORITHM:
    for each module:
        select photons with:
            E in 4-11 keV range
            RAWX,RAWY outside illuminated area
            NTOP==NBOT==1
        for dt in observation:
            calculate total livetime
            correct for deadtime
            calculate background rate
    plot lightcurve

    for each module:
        select photons with:
            E in 5-30 keV range
            NTOP==NBOT==1
    plot detector histogram
    



30/09/19 v002
    now instead of 2Mb pdf with ra-dec plot there is slim png
    all hail the hypnotoad

17/09/19 v001
    initial version derived from old artxc_l0_quicklook
    all hail the hypnotoad
'''

import os, shutil, argparse
from astropy.io import fits
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.colors
from arttools.caldb import get_shadowmask
from sys import exit
import os.path
import arttools.quicktools as artql
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--stem", help="ART-XC stem like srg_YYYYMMDD_HHMMSS")
parser.add_argument("--version", help="data version, usually 000", default='000')
args = parser.parse_args()
    

stem    = args.stem
if stem==None or not stem:
    print ('>>ERROR>> Please, provide stem')
    exit(1)
else:
    print ('>>>>>>>>> stem:'+stem)
subvers = args.version
if subvers==None or not subvers:
    print ('>>>>>>>>>  using subversion 000')
else:
    print ('>>>>>>>>> subversion:'+subvers)

wdir      = '/srg/a1/work/oper/data/2019/'
L1        = os.path.join(wdir,stem,'L1')
L1b       = os.path.join(wdir,stem,'L1b')
stem_tail = '_urd.fits'
gyro_file = stem + '_'+subvers+'_gyro_att.fits'
module_names   = ['02','04','08','10','20','40','80']
tel_names = ['T1','T2','T3','T4','T5','T6','T7']
module_color   = ['k','r','g','b','m','c','lime']
pdfname = stem + '.pdf'
pngname = stem + '.png'
with PdfPages(pdfname) as pdffile: 
    for module,teln,modc in zip(module_names[:],tel_names,module_color):
        print ('>>>>>>>>> Working with module '+ module)
        evtfile = stem +'_'+subvers+'.'+ module + '_urd.fits'
        evtpath = os.path.join(L1b,evtfile)
        try:
            evtfits = fits.open(evtpath)
            evtfits.close()
        except:
            print ('>>ERROR>> Cannot open '+evtpath)
        evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti = artql.get_cl_events(evtpath,module,teln)
        fov_hist,bkg_hist,emeans, ewidths = artql.get_spectrum(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, evtpath,module,teln, pdffile)
        cleanmask = np.bitwise_and(np.bitwise_and(evgrade>=0,evgrade<=8),np.bitwise_and(evenergies>=4,evenergies<=11.))
        artql.get_rawmap(evrawx[cleanmask], evrawy[cleanmask], pdffile, teln)    
        artql.get_lcurve(evtimes, evenergies, evgrade, evflag, evrawx, evrawy, gti, evtpath,module,teln, pdffile)
    
    gyropath = os.path.join(L1,gyro_file)
    artql.get_radec(gyropath, gti, pdffile, pngname)
    d = pdffile.infodict()
    d['Title'] = 'Quicklook ART-XC report, v.01'
    d['Author'] = 'oper'
    d['Subject'] = 'ART-XC quicklook data'
    d['CreationDate'] = datetime.datetime.today()
    d['ModDate'] = datetime.datetime.today()
artql.run('convert '+pngname+' '+pngname.replace('.png','_gyro.pdf'))
artql.run('pdfunite '+pdfname+' '+pngname.replace('.png','_gyro.pdf')+' '+pngname.replace('.png','_qreport.pdf'))
artql.run('rm '+pdfname+' '+pngname.replace('.png','_gyro.pdf')+' '+pngname)


