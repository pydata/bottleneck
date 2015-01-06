"""
Alternative methods (non-Cython) of calculating moving window statistics.

These function are slow but useful for unit testing.
"""

import warnings

import numpy as np


__all__ = ['move_sum', 'move_mean', 'move_std', 'move_min', 'move_max',
           'move_median']


def move_sum(arr, window, min_count=None, axis=-1):
    "Slow move_sum for unaccelerated dtype"
    return move_func(np.nansum, arr, window, min_count, axis=axis)


def move_mean(arr, window, min_count=None, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, arr, window, min_count, axis=axis)


def move_std(arr, window, min_count=None, axis=-1, ddof=0):
    "Slow move_std for unaccelerated dtype"
    return move_func(np.nanstd, arr, window, min_count, axis=axis)


def move_min(arr, window, min_count=None, axis=-1):
    "Slow move_min for unaccelerated dtype"
    return move_func(np.nanmin, arr, window, min_count, axis=axis)


def move_max(arr, window, min_count=None, axis=-1):
    "Slow move_max for unaccelerated dtype"
    return move_func(np.nanmax, arr, window, min_count, axis=axis)


def move_median(arr, window, axis=-1):
    "Slow move_median for unaccelerated dtype"
    return move_func(np.nanmedian, arr, window, window, axis=axis)


# magic utility functions ---------------------------------------------------

def move_func(func, arr, window, min_count=None, axis=-1, **kwargs):
    "Generic moving window function implemented with a python loop."
    arr = np.array(arr, copy=False)
    if min_count is None:
        mc = window
    else:
        mc = min_count
        if mc > window:
            msg = "min_count (%d) cannot be greater than window (%d)"
            raise ValueError(msg % (mc, window))
        elif mc <= 0:
            raise ValueError("`min_count` must be greater than zero.")
    if arr.ndim == 0:
        raise ValueError("moving window functions require ndim > 0")
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    if issubclass(arr.dtype.type, np.inexact):
        y = np.empty_like(arr)
    else:
        y = np.empty(arr.shape)
    idx1 = [slice(None)] * arr.ndim
    idx2 = list(idx1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(arr.shape[axis]):
            win = min(window, i + 1)
            idx1[axis] = slice(i + 1 - win, i + 1)
            idx2[axis] = i
            a = arr[idx1]
            y[idx2] = func(a, axis=axis, **kwargs)
    idx = _mask(arr, window, mc, axis)
    y[idx] = np.nan
    return y


def _mask(arr, window, min_count, axis):
    n = (arr == arr).cumsum(axis)
    idx1 = [slice(None)] * arr.ndim
    idx2 = list(idx1)
    idx1[axis] = slice(window, None)
    idx2[axis] = slice(None, -window)
    nidx1 = n[idx1]
    nidx1 = nidx1 - n[idx2]
    idx = n <  min_count
    return idx
