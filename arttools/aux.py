import numpy as np

def numpy_array_naive_hash(arr):
    return arr.__array_interface__['data'] + arr.strides + (arr.size,) + (arr.dtype,)

def cache_single_result_np(function):
    lastcallsignature = None
    lastcallresult = None
    def newfunc(*args, **kwargs):
        nonlocal lastcallsignature
        nonlocal lastcallresult
        storeresult = kwargs.pop("storeresult", False)
        key = (numpy_array_naive_hash(val) if issubclass(val, np.ndarray) else val for val in args)
        if kwargs:
            for item in kwargs.items():
                key += item
        callsignature = hash(key)
        if lastcallresult == callsignature:
            result = lastcallresult
        else:
            result = function(*args, **kwargs)
            if storeresult:
                lastcallresult = result
                lastcallsignature = callsignature
        return result
    return newfunc

