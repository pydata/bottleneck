import warnings
import numpy as np

__all__ = ['median', 'nanmedian', 'nansum', 'nanmean', 'nanvar', 'nanstd',
           'nanmin', 'nanmax', 'nanargmin', 'nanargmax', 'ss', 'anynan',
           'allnan']

rankdata_func = None

from numpy import (nansum, nanmean, nanstd, nanvar, nanmin, nanmax, median,
                   nanmedian)


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


def ss(arr, axis=None):
    "Slow sum of squares used for unaccelerated dtypes."
    return np.multiply(arr, arr).sum(axis)


def anynan(arr, axis=None):
    "Slow check for Nans used for unaccelerated dtypes."
    return np.isnan(arr).any(axis)


def allnan(arr, axis=None):
    "Slow check for all Nans used for unaccelerated dtypes."
    return np.isnan(arr).all(axis)
