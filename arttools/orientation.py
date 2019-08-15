from scipy.spatial.transform import Rotation, Slerp
import numpy as np
from math import pi, cos, sin

qrot0 = Rotation([sin(135.*pi/360.), 0., 0., cos(135.*pi/360.)])
#qrot0 = Rotation([0., 0., 0., 1.])
#qrot0 = Rotation([sin(195.*pi/360.), 0., 0., cos(195.*pi/360.)])

"""
ART_det_QUAT = {
        28 : Rotation([-0.0110150413845477,     -0.0013329854433192,     -0.0010840551373762,      0.9999378564878738]),
        22 : Rotation([-0.0097210046441978,     -0.0012050978602830,     -0.0010652583365244,      0.9999514563380221]),
        23 : Rotation([-0.0109821945673236,     -0.0013205569544235,     -0.0010833616131844,      0.9999382350222591]), 
        24 : Rotation([-0.0108749144342713,     -0.0012081051721620,     -0.0009784162714686,      0.9999396578891848]), 
        25 : Rotation([-0.0083598094506972,     -0.0012398390463856,     -0.0011014394837848,      0.9999636809485386]), 
        26 : Rotation([-0.0100908546351636,     -0.0012650094280487,     -0.0011698124374266,      0.9999476015985739]),
        30 : Rotation([-0.0108764670901360,     -0.0012574641047721,     -0.0010592143554341,      0.9999394978260493])
                }
"""

"""
ART_det_QUAT = {
        28 : Rotation([0.0110150413845477,     0.0013329854433192,     0.0010840551373762,      0.9999378564878738]),
        22 : Rotation([0.0097210046441978,     0.0012050978602830,     0.0010652583365244,      0.9999514563380221]),
        23 : Rotation([0.0109821945673236,     0.0013205569544235,     0.0010833616131844,      0.9999382350222591]), 
        24 : Rotation([0.0108749144342713,     0.0012081051721620,     0.0009784162714686,      0.9999396578891848]), 
        25 : Rotation([0.0083598094506972,     0.0012398390463856,     0.0011014394837848,      0.9999636809485386]), 
        26 : Rotation([0.0100908546351636,     0.0012650094280487,     0.0011698124374266,      0.9999476015985739]),
        30 : Rotation([0.0108764670901360,     0.0012574641047721,     0.0010592143554341,      0.9999394978260493])
                }


"""
ART_det_QUAT = {
        28 : Rotation([0., 0., 0., 1.]),
        22 : Rotation([0., 0., 0., 1.]),
        23 : Rotation([0., 0., 0., 1.]), 
        24 : Rotation([0., 0., 0., 1.]), 
        25 : Rotation([0., 0., 0., 1.]), 
        26 : Rotation([0., 0., 0., 1.]),
        30 : Rotation([0., 0., 0., 1.])
                }


def to_2pi_range(val): return val%(2.*pi)


def get_gyro_quat(gyrodata):
    quat = Rotation(np.array([gyrodata["QORT_%d" % i] for i in [1,2,3,0]]).T)
    q0 = Rotation([0, 0, 0, 1]) #gyro axis initial rotattion in J2000 system
    qfin = q0*quat
    return qfin



def extract_raw_gyro(gyrodata, qadd=Rotation([sin(-15.*pi/360.), 0., 0., cos(-15.*pi/360.)])):
    """
    unpacks row gyro fits file in to RA, DEC and roll angle (of the telescope coordinate system) in J2000 coordinates.

    attention!
    currently in gyro fits file quaternion scalar component is stored after the vector component [V, s] (V = {xp, yp, zp}*sin(\alpha/2)) , while most of 
    standard subroutines expect the quaternion in form [s, V] (for example scipy.spatial.transform.Rotation)
    """
    qfin = get_gyro_quat(gyrodata)*qadd

    # telescope optical axis is x axis in this coordinate system
    opticaxis = qfin.apply([1, 0, 0])

    #ra and dec in radians
    dec = np.arctan(opticaxis[:,2]/np.sqrt(opticaxis[:,1]**2 + opticaxis[:,0]**2)) 
    ra = np.arctan2(opticaxis[:,1], opticaxis[:,0])%(2.*pi)

    yzprojection = np.cross(opticaxis, [0., 0., 1.])

    rollangle = np.arctan2(np.sum(yzprojection*qfin.apply([0, 0, 1]), axis=1), np.sum(yzprojection*qfin.apply([0, 1, 0]), axis=1))
    return ra, dec, rollangle

