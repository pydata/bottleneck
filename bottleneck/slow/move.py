"""
Alternative methods (non-Cython) of calculating moving window statistics.

These function are slow but useful for unit testing.
"""

import warnings

import numpy as np


__all__ = ['move_sum', 'move_mean', 'move_std', 'move_min', 'move_max',
           'move_median']


def move_sum(arr, window, nmin=-1, axis=-1):
    "Slow move_sum for unaccelerated dtype"
    return move_func(np.nansum, arr, window, nmin, axis=axis)


def move_mean(arr, window, nmin=-1, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, arr, window, nmin, axis=axis)


def move_std(arr, window, nmin=-1, axis=-1, ddof=0):
    "Slow move_std for unaccelerated dtype"
    return move_func(np.nanstd, arr, window, nmin, axis=axis)


def move_min(arr, window, nmin=-1, axis=-1, ddof=0):
    "Slow move_min for unaccelerated dtype"
    return move_func(np.nanmin, arr, window, nmin, axis=axis)


def move_max(arr, window, nmax=-1, axis=-1, ddof=0):
    "Slow move_max for unaccelerated dtype"
    return move_func(np.nanmax, arr, window, nmax, axis=axis)


# MEDIAN --------------------------------------------------------------------


def move_median(arr, window, axis=-1, method='loop'):
    """
    Slow moving window median along the specified axis.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving median. By default the
        moving median is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'loop'      brute force python loop (default)
            'strides'   strides tricks
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving median of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> bn.move_median(arr, window=2)
    array([ NaN,  1.5,  2.5,  3.5,  4.5])

    """
    arr = np.array(arr, copy=False)
    if method == 'strides':
        y = move_func_strides(np.median, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.median, arr, window, axis=axis)
    else:
        msg = "`method` must be 'strides' or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y

# GENERAL --------------------------------------------------------------------


def move_func(func, arr, window, nmin=-1, axis=-1, **kwargs):
    "Generic moving window function implemented with a python loop."
    arr = np.array(arr, copy=False)
    if arr.ndim == 0:
        raise ValueError("moving window functions require ndim > 0")
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    if nmin < 0:
        nmin = window
    elif nmin > window:
        msg = "nmin (%d) cannot be greater than window (%d)"
        raise ValueError(msg % (nmin, window))
    elif nmin == 0:
        raise ValueError("`nmin` cannot be zero")
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
    idx = _mask(arr, window, nmin, axis)
    y[idx] = np.nan
    return y


def _mask(arr, window, nmin, axis):
    n = (arr == arr).cumsum(axis)
    idx1 = [slice(None)] * arr.ndim
    idx2 = list(idx1)
    idx1[axis] = slice(window, None)
    idx2[axis] = slice(None, -window)
    nidx1 = n[idx1]
    nidx1 = nidx1 - n[idx2]
    idx = n <  nmin
    return idx
