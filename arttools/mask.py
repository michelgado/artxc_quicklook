import numpy as np

def edges(mask):
    mnew = np.empty(mask.size + 2, np.bool)
    mnew[1:-1] = mask[:]
    mnew[[0, -1]] = False
    return np.where(np.logical_xor(mnew[:-1], mnew[1:]))[0].reshape((-1, 2))

