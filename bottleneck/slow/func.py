
import numpy as np
import scipy.stats as sp

__all__ = ['median', 'nanmean', 'nanvar', 'nanstd', 'nanmin', 'nanmax']

def median(arr, axis=None):
    "Slow median function used for unaccelerated ndim/dtype combinations."
    y = np.median(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

def nanmean(arr, axis=None):
    "Slow nanmean function used for unaccelerated ndim/dtype combinations."
    y = sp.nanmean(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

def nanvar(arr, axis=None, ddof=0):
    "Slow nanvar function used for unaccelerated ndim/dtype combinations."
    y = nanstd(arr, axis=axis, ddof=ddof)
    return y * y

def nanstd(arr, axis=None, ddof=0):
    "Slow nanstd function used for unaccelerated ndim/dtype combinations."
    if ddof == 0:
        bias = True
    elif ddof == 1:
        bias = False
    else:
        raise ValueError("With NaNs ddof must be 0 or 1.")
    if axis != None:
        # Older versions of scipy can't handle negative axis?
        if axis < 0:
            axis += arr.ndim
        if (axis < 0) or (axis >= arr.ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    else:
        # Older versions of scipy choke on axis=None
        arr = arr.ravel()
        axis = 0
    y = sp.nanstd(arr, axis=axis, bias=bias)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

def nanmin(arr, axis=None):
    "Slow nanmin function used for unaccelerated ndim/dtype combinations."
    return np.nanmin(arr, axis=axis)

def nanmax(arr, axis=None):
    "Slow nanmax function used for unaccelerated ndim/dtype combinations."
    return np.nanmax(arr, axis=axis)
