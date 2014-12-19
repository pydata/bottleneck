"""
Alternative methods (non-Cython) of calculating moving window statistics.

These function are slow but useful for unit testing.
"""

import warnings

import numpy as np


__all__ = ['move_sum',
           'move_mean',
           'move_std',
           'move_min',
           'move_max', 'move_nanmax',
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


# MAX -----------------------------------------------------------------------


def move_max(arr, window, axis=-1, method='loop'):
    """
    Slow move_max for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving maximum. By default the
        moving maximum is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =========================================
            'filter'    scipy.ndimage.minimum_filter1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =========================================

    Returns
    -------
    y : ndarray
        The moving maximum of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_max(arr, window=2)
    array([ NaN,   2.,   3.,   4.])

    """
    if method == 'filter':
        y = move_max_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(np.max, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.max, arr, window, axis=axis)
    else:
        raise ValueError("`method` must be 'filter', 'strides', or 'loop'.")
    return y


def move_nanmax(arr, window, axis=-1, method='loop'):
    """
    Slow move_nanmax for unaccelerated ndim/dtype combinations, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving maximum. By default the
        moving maximum is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =========================================
            'filter'    scipy.ndimage.maximum_filter1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =========================================

    Returns
    -------
    y : ndarray
        The moving maximum of the input array along the specified axis,
        ignoring NaNs. (A window with all NaNs returns NaN for the window
        maximum.) The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> bn.slow.move_nanmax(arr, window=2)
    array([ NaN,   2.,   2.,   4.,   5.])

    """
    if method == 'filter':
        y = move_nanmax_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_nanmax_strides(arr, window, axis=axis)
    elif method == 'loop':
        y = move_nanmax_loop(arr, window, axis=axis)
    else:
        raise ValueError("`method` must be 'filter', 'strides', or 'loop'.")
    return y


def move_max_filter(arr, window, axis=-1):
    "Moving window maximium implemented with a filter."
    arr = np.array(arr, copy=False)
    global maximum_filter1d
    if maximum_filter1d is None:
        try:
            from scipy.ndimage import maximum_filter1d
        except ImportError:
            raise ValueError("'filter' method requires SciPy.")
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    y = arr.astype(float)
    x0 = (window - 1) // 2
    maximum_filter1d(y, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=y)
    return y


def move_nanmax_filter(arr, window, axis=-1):
    "Moving window maximium ignoring NaNs, implemented with a filter."
    arr = np.array(arr, copy=False)
    global maximum_filter1d, convolve1d
    if maximum_filter1d is None:
        try:
            from scipy.ndimage import maximum_filter1d
        except ImportError:
            raise ValueError("'filter' method requires SciPy.")
    if convolve1d is None:
        try:
            from scipy.ndimage import convolve1d
        except ImportError:
            raise ValueError("'filter' method requires SciPy.")
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    if issubclass(arr.dtype.type, np.inexact):
        arr = arr.copy()
    else:
        arr = arr.astype(np.float64)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    x0 = (window - 1) // 2
    maximum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=arr)
    w = np.ones(window, dtype=int)
    nrr = nrr.astype(int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr[nrr == window] = np.nan
    return arr


def move_nanmax_loop(arr, window, axis=-1):
    "Moving window maximium ignoring NaNs, implemented with a python loop."
    arr = np.array(arr, copy=False)
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    if issubclass(arr.dtype.type, np.inexact):
        arr = arr.copy()
    else:
        arr = arr.astype(np.float64)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    y = move_func_loop(np.max, arr, window, axis=axis)
    m = move_func_loop(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y


def move_nanmax_strides(arr, window, axis=-1):
    "Moving window maximium ignoring NaNs, implemented with stides tricks."
    arr = np.array(arr, copy=False)
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    if issubclass(arr.dtype.type, np.inexact):
        arr = arr.copy()
    else:
        arr = arr.astype(np.float64)
    nrr = np.isnan(arr)
    arr[nrr] = -np.inf
    y = move_func_strides(np.max, arr, window, axis=axis)
    m = move_func_strides(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

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
