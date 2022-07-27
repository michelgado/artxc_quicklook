from collections import OrderedDict
import numpy as np


URDNS = (28, 22, 23, 24, 25, 26, 30)
TELESCOPES = ("T1", "T2", "T3", "T4", "T5", "T6", "T7")

OPAX = (1., 0., 0.)

URDTOTEL = {28: "T1",
            22: "T2",
            23: "T3",
            24: "T4",
            25: "T5",
            26: "T6",
            30: "T7"}

ANYTHINGTOURD = {"T1": 28, "T2": 22, "T3": 23, "T4": 24, "T5": 25, "T6": 26, "T7": 30,
                 "28": 28, "22": 22, "23": 23, "24": 24, "25": 25, "26": 26, "30": 30,
                 "02": 28, "04": 22, "08": 23, "10": 24, "20": 25, "40": 26, "80": 30,
                 2: 28, 4: 22, 8: 23, 10: 24, 20: 25, 40: 26, 80: 30,
                 28: 28, 22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 30: 30}

ANYTHINGTOTELESCOPE = {key: URDTOTEL[val] for key, val in ANYTHINGTOURD.items()}


TELTOURD = {v:k for k, v in URDTOTEL.items()}

def ordered_map(function, *tdicts, **kwargs):
    urdlist = []
    for urdn in URDNS:
        checkin = [urdn in td for td in tdicts]
        if any(checkin) != all(checkin):
            raise ValueError("input data has inconsistent amount of telescope data")
        if all(checkin):
            urdlist.append(urdn)
    """
    to do: put a warn here for the case, when input dicts have different number of keys
    """
    return [function(*[t[urdn] for t in tdicts], **kwargs) for urdn in urdlist]

def do_for_all_telescopes(function):
    def newfunc(*tdicts, **kargs):
        return ordered_map(function, *tdicts, **kwargs)
    return newfunc

def to_ordered(d):
    return OrderedDict(zip(URDNS, [d[urdn] for urdn in URDNS]))

def concat_data_in_order(urddicts):
    return np.concatenate([urddicts[urdn] for urdn in URDNS if urdn in urddicts])
