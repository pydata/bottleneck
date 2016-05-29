import warnings
import numpy as np
from numpy import nanmean, nanmedian

__all__ = ['median', 'nanmedian', 'nansum', 'nanmean', 'nanvar', 'nanstd',
           'nanmin', 'nanmax', 'nanargmin', 'nanargmax', 'ss', 'anynan',
           'allnan']


def nansum(arr, axis=None):
    "Slow nansum function used for unaccelerated dtype."
    arr = np.asarray(arr)
    y = np.nansum(arr, axis=axis)
    if y.dtype != arr.dtype:
        y = y.astype(arr.dtype)
    return y


def nanargmin(arr, axis=None):
    "Slow nanargmin function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmin(arr, axis=axis)


def nanargmax(arr, axis=None):
    "Slow nanargmax function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmax(arr, axis=axis)


def nanvar(arr, axis=None, ddof=0):
    "Slow nanvar function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanvar(arr, axis=axis, ddof=ddof)


def nanstd(arr, axis=None, ddof=0):
    "Slow nanstd function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(arr, axis=axis, ddof=ddof)


def nanmin(arr, axis=None):
    "Slow nanmin function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(arr, axis=axis)


def nanmax(arr, axis=None):
    "Slow nanmax function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmax(arr, axis=axis)


def median(arr, axis=None):
    "Slow median function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.median(arr, axis=axis)


def ss(arr, axis=None):
    "Slow sum of squares used for unaccelerated dtypes."
    arr = np.asarray(arr)
    y = np.multiply(arr, arr).sum(axis)
    if y.dtype != arr.dtype:
        y = y.astype(arr.dtype)
    return y


def anynan(arr, axis=None):
    "Slow check for Nans used for unaccelerated dtypes."
    return np.isnan(arr).any(axis)


def allnan(arr, axis=None):
    "Slow check for all Nans used for unaccelerated dtypes."
    return np.isnan(arr).all(axis)
