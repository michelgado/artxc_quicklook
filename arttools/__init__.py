from . import plot
from . import energy
from . import mask
from . import expmap
from . import process
from . import aux
from . import datafilters
from . import filters
from . import psf
from . import sensitivity
from . import spectr
from . import illumination
from . import point_source
from . import containers

__version__ = "0.3.1"


"""
This library currently (2020.04.08) consist of several functions, separated in several modules.
Higher level modules, depend on lower.

There is also one module (caldb.py) which is separate and can be called by all others
the overal structure can be described in the following scheme

  LEVEL
    1             |      time.py    _det_spatial.py    energy.py
#==================================================================================================================================
* time.py - contains sevelra functions, which reduces the L0 data time
main functions and calsses are:
    ** GTI - general segment class (applied to the time in this implementation), stores a set of ordered uncrossing time segments.
       has following usefull methods: & - provides intersection of two GTIs, | - provide union of two GTIs,
       single - gives inverse GTI interval

        another helpfull methods of the class are:
            *** mask_outofgti_times - for a set of time points return mask, showing which are in GTI
            *** arange - for selected time resolution return edges of the time bins inside GTI

    ** get_gti - for provided L0 level urd file returns GTI interval

    ** deadtime_correction - based on houskipning extention of the file, return dead time corrections as a function of time

* _det_spatial.py - contains functions, defining the detector spatial properties

    ** raw_xy_to_offset             |
    ** offset_to_raw_xy             |       all these functions switch coordinates system (i.e. pixels to mm offset from the detecto center)
    ** raw_xy_to_vec                |       also, with known focal length, functions produce vectors from offset (based on thangential rule
    ** offset_to_vec                |
    ** vec_to_offset                |
    ** vec_to_offset_pairs          |

    ** get_shadowed_pix_mask - get current shadow pix mask with help of caldb.py and produces events mask based on their coordinates

* energy.py
    ** PHA_to_PI - converts digital amplitude to energy
    ** get_events_energy - for eventlist (with given houskeeping data) produces energies and grades

there is also a small mask.py module (the solo function in it is edges, which returns indexes of edges of True segments)


    2             |    orientation.py      lightcurve.py
#================================================================================================================
* orientation.py - contains a set of funtctinos to work with vectors (produced by _det_spatial) and quaternions (produced by star trackers and SC orientation facilities)

    depends on: time.py _det_spatial.py caldb.py

    ** vec_to_pol & pol_to_vec - converts decartian vector to spherical coordinates and back
    ** AttDATA class
        a class, inheriting from scipy.spatial.transform.Slerp - quaternions interpolation class
        the reason, why we should have a separate class for our purposess are following:
            attitude data may be not actual for specific time moments, for example, part of the att data was lost at the orbit correction,
            in that case, optical axis orientation can not be obtained by interpolating this points,
            this information (as I think) should be stored with interpolated quaternions

        there are also several usefull functions added in this class
        *** concatenate, unify several AttDATA instances, accounting for their GTIs

    ** read_gyro_fits - returns AttDATA from
    ** read_bokz_fits

* lightcurve.py - in many cases we will need estimates of the count rate of particular set of events.
                  during count rate estimation GTIs should be accounted

    depends on: time.py energy.py caldb.py

    ** weigt_time_intervals - since we have 7 independent telescopes we may want to define *"effective area" of the currently working telescopes (since some of the can be not operating)
    ** make_overall_lc - for events, produces by all telescopes, and GTIs from each telescope, produces count rate (*per cm^2)



    3             |    planwcs.py    atthist.py  vignetting.py
#================================================================================================================
* atthist.py - given the att information we frequentyle would need to produce high (spatial) resolution track of our optical axis on sky
               this is required for two important tasks: background map production and exposure map production.
               for the performance reasons there are also subroutines aimed to histogram attitudes, and sum their exposures

    depends on: time.py caldb.py orientation.py _det_spatial.py

    ** hist_orientation - histogram orientation by spherical coordinates and roll angles
    ** make_small_steps_quats - produces a set of quaternions, which are sepatated not more than by a DELTASKY
                                (and DELTAROW in rotation) and corresponding time intervals,

"""
