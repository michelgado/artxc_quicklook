from .caldb import get_shadowmask_by_urd, get_vigneting_by_urd
from .telescope import URDNS
from scipy.interpolate import interp1d
from scipy.integrate import quad


def make_vigneting_for_urd(urdn, energy=6., phot_index=None):
    vignfile = get_vigneting_by_urd(urdn)
    vign = 0.

    eff1 = interp1d(vignfile["Vign_EA"].data["E"],
                    vignfile["Vign_EA"].data["EFFAREA"],
                    axis=0)

   if not plaw is None:
        vign = quad(eff1, vignfile["Vign_EA"].data["E"][0],
                   vignfile["Vign_EA"].data["E"][-1])
    else:
        vign = eff1(energy)
    return vign/vign.max()


def make_overall_vign_map(energy=6., plaw=None):
    v
