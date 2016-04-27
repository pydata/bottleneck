"""
Alternative methods (non-Cython) of calculating moving window statistics.

These function are slow but useful for unit testing.
"""

import warnings

import numpy as np


__all__ = ['move_sum', 'move_mean', 'move_std', 'move_var', 'move_min',
           'move_max', 'move_argmin', 'move_argmax', 'move_median',
           'move_rank']


def move_sum(arr, window, min_count=None, axis=-1):
    "Slow move_sum for unaccelerated dtype"
    return move_func(np.nansum, arr, window, min_count, axis=axis)


def move_mean(arr, window, min_count=None, axis=-1):
    "Slow move_mean for unaccelerated dtype"
    return move_func(np.nanmean, arr, window, min_count, axis=axis)


def move_std(arr, window, min_count=None, axis=-1, ddof=0):
    "Slow move_std for unaccelerated dtype"
    return move_func(np.nanstd, arr, window, min_count, axis=axis)


def move_var(arr, window, min_count=None, axis=-1, ddof=0):
    "Slow move_var for unaccelerated dtype"
    return move_func(np.nanvar, arr, window, min_count, axis=axis)


def move_min(arr, window, min_count=None, axis=-1):
    "Slow move_min for unaccelerated dtype"
    return move_func(np.nanmin, arr, window, min_count, axis=axis)


def move_max(arr, window, min_count=None, axis=-1):
    "Slow move_max for unaccelerated dtype"
    return move_func(np.nanmax, arr, window, min_count, axis=axis)


def move_argmin(arr, window, min_count=None, axis=-1):
    "Slow move_argmin for unaccelerated dtype"
    def argmin(arr, axis):
        arr = np.array(arr, copy=False)
        flip = [slice(None)] * arr.ndim
        flip[axis] = slice(None, None, -1)
        arr = arr[flip]  # if tie, pick index of rightmost tie
        try:
            idx = np.nanargmin(arr, axis=axis)
        except ValueError:
            # an all nan slice encountered
            a = arr.copy()
            mask = np.isnan(a)
            np.copyto(a, np.inf, where=mask)
            idx = np.argmin(a, axis=axis).astype(np.float64)
            if idx.ndim == 0:
                idx = np.nan
            else:
                mask = np.all(mask, axis=axis)
                idx[mask] = np.nan
        return idx
    return move_func(argmin, arr, window, min_count, axis=axis)


def move_argmax(arr, window, min_count=None, axis=-1):
    "Slow move_argmax for unaccelerated dtype"
    def argmax(arr, axis):
        arr = np.array(arr, copy=False)
        flip = [slice(None)] * arr.ndim
        flip[axis] = slice(None, None, -1)
        arr = arr[flip]  # if tie, pick index of rightmost tie
        try:
            idx = np.nanargmax(arr, axis=axis)
        except ValueError:
            # an all nan slice encountered
            a = arr.copy()
            mask = np.isnan(a)
            np.copyto(a, -np.inf, where=mask)
            idx = np.argmax(a, axis=axis).astype(np.float64)
            if idx.ndim == 0:
                idx = np.nan
            else:
                mask = np.all(mask, axis=axis)
                idx[mask] = np.nan
        return idx
    return move_func(argmax, arr, window, min_count, axis=axis)


def move_median(arr, window, min_count=None, axis=-1):
    "Slow move_median for unaccelerated dtype"
    return move_func(np.nanmedian, arr, window, min_count, axis=axis)


def move_rank(arr, window, min_count=None, axis=-1):
    "Slow move_rank for unaccelerated dtype"
    return move_func(lastrank, arr, window, min_count, axis=axis)


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
    idx2 = [slice(None)] * arr.ndim
    idx3 = [slice(None)] * arr.ndim
    idx1[axis] = slice(window, None)
    idx2[axis] = slice(None, -window)
    idx3[axis] = slice(None, window)
    nidx1 = n[idx1]
    nidx1 = nidx1 - n[idx2]
    idx = np.empty(arr.shape, dtype=np.bool)
    idx[idx1] = nidx1 < min_count
    idx[idx3] = n[idx3] < min_count
    return idx


# ---------------------------------------------------------------------------

def lastrank(arr, axis=-1):
    """
    The ranking of the last element along the axis, ignoring NaNs.

    The ranking is normalized to be between -1 and 1 instead of the more
    common 1 and N. The results are adjusted for ties.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : int, optional
        The axis over which to rank. By default (axis=-1) the ranking
        (and reducing) is performed over the last axis.

    Returns
    -------
    d : array
        In the case of, for example, a 2d array of shape (n, m) and
        axis=1, the output will contain the rank (normalized to be between
        -1 and 1 and adjusted for ties) of the the last element of each row.
        The output in this example will have shape (n,).

    Examples
    --------
    Create an array:

    >>> y1 = larry([1, 2, 3])

    What is the rank of the last element (the value 3 in this example)?
    It is the largest element so the rank is 1.0:

    >>> import numpy as np
    >>> from la.afunc import lastrank
    >>> x1 = np.array([1, 2, 3])
    >>> lastrank(x1)
    1.0

    Now let's try an example where the last element has the smallest
    value:

    >>> x2 = np.array([3, 2, 1])
    >>> lastrank(x2)
    -1.0

    Here's an example where the last element is not the minimum or maximum
    value:

    >>> x3 = np.array([1, 3, 4, 5, 2])
    >>> lastrank(x3)
    -0.5

    """
    a = np.array(arr, copy=False)
    ndim = a.ndim
    if a.size == 0:
        # At least one dimension has length 0
        shape = list(a.shape)
        shape.pop(axis)
        r = np.empty(shape, dtype=a.dtype)
        r.fill(np.nan)
        if (r.ndim == 0) and (r.size == 1):
            r = np.nan
        return r
    indlast = [slice(None)] * ndim
    indlast[axis] = slice(-1, None)
    indlast2 = [slice(None)] * ndim
    indlast2[axis] = -1
    n = (~np.isnan(a)).sum(axis)
    a_indlast = a[indlast]
    g = (a_indlast > a).sum(axis)
    e = (a_indlast == a).sum(axis)
    r = (g + g + e - 1.0) / 2.0
    r = r / (n - 1.0)
    r = 2.0 * (r - 0.5)
    if ndim == 1:
        if n == 1:
            r = 0
        if np.isnan(a[indlast2]):  # elif?
            r = np.nan
    else:
        np.putmask(r, n == 1, 0)
        np.putmask(r, np.isnan(a[indlast2]), np.nan)
    return r
