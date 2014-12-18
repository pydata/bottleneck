"""
Alternative methods (non-Cython) of calculating moving window statistics.

These function are slow but useful for unit testing.
"""

import numpy as np
import bottleneck as bn

convolve1d = None
minimum_filter1d = None
maximum_filter1d = None

__all__ = ['move_sum',
           'move_mean', 'move_nanmean',
           'move_std', 'move_nanstd',
           'move_min', 'move_nanmin',
           'move_max', 'move_nanmax',
           'move_median']

# SUM -----------------------------------------------------------------------


def move_sum(arr, window, nmin=-1, axis=-1):
    """
    Slow move_sum for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving sum. By default the moving
        sum is taken over the last axis (-1).

    Returns
    -------
    y : array_like
        The moving sum of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_sum(arr, window=2, axis=0)
       array([ NaN,   3.,   5.,   7.])

    """
    y = move_func(_nansum_default_nan, arr, window, nmin, axis=axis)
    return y


def _nansum_default_nan(arr, axis=None):
    "All nan input returns nan instead of 0. Int input converted to float64"
    a = np.nansum(arr, axis=axis)
    idx = np.isnan(arr).all(axis=axis)
    if a.ndim == 0:
        if idx:
            a = np.float64(np.nan)
    else:
        if issubclass(a.dtype.type, np.inexact):
            a[idx] = np.nan
        else:
            a = a.astype(np.float64)
    return a


# MEAN -------------------------------------------------------------------


def move_mean(arr, window, axis=-1, method='loop'):
    """
    Slow move_mean for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving mean. By default the moving
        mean is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_mean(arr, window=2, axis=0)
    array([ NaN,  1.5,  2.5,  3.5])

    """
    arr = np.array(arr, copy=False)
    if method == 'filter':
        y = move_mean_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(np.mean, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.mean, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_nanmean(arr, window, axis=-1, method='loop'):
    """
    Slow move_nanmean for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving mean. By default the moving
        mean is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis, ignoring
        NaNs. (A window with all NaNs returns NaN for the window mean.) The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4])
    >>> bn.slow.move_nanmean(arr, window=2, axis=0)
    array([ NaN,  1.5,  2. ,  4. ])

    """
    arr = np.array(arr, copy=False)
    if method == 'filter':
        y = move_nanmean_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(bn.slow.nanmean, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(bn.slow.nanmean, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_mean_filter(arr, window, axis=-1):
    "Moving window mean implemented with a filter."
    arr = np.array(arr, copy=False)
    global convolve1d
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
    arr = arr.astype(float)
    w = np.empty(window)
    w.fill(1.0 / window)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    return arr


def move_nanmean_filter(arr, window, axis=-1):
    "Moving window nanmean implemented with a filter."
    arr = np.array(arr, copy=False)
    global convolve1d
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
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = 0
    nrr = nrr.astype(int)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr /= (window - nrr)
    arr[nrr == window] = np.nan
    return arr

# VAR -----------------------------------------------------------------------


def move_var(arr, window, axis=-1, method='loop', ddof=0):
    """
    Slow move_var for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving variance. By default the
        moving variance is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving variance of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_var(arr, window=2, axis=0)
    array([  NaN,  0.25,  0.25,  0.25])

    """
    arr = np.array(arr, copy=False)
    if ddof != 0:
        raise ValueError("`ddof` must be zero for unaccelerated input.")
    if method == 'filter':
        y = move_var_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(np.var, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.var, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_nanvar(arr, window, axis=-1, method='loop', ddof=0):
    """
    Slow move_nanvar for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving variance. By default the
        moving variance is taken over the last axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving variance of the input array along the specified axis,
        ignoring NaNs. (A window with all NaNs returns NaN for the window
        variance.) The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> bn.slow.move_nanvar(arr, window=3, axis=0)
    array([  NaN,   NaN,  0.25,  1.  ,  0.25])

    """
    arr = np.array(arr, copy=False)
    if ddof != 0:
        raise ValueError("`ddof` must be zero for unaccelerated input.")
    if method == 'filter':
        y = move_nanvar_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(bn.slow.nanvar, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(bn.slow.nanvar, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_var_filter(arr, window, axis=-1):
    "Moving window variance implemented with a filter."
    arr = np.array(arr, copy=False)
    global convolve1d
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
    arr = arr.astype(float)
    w = np.empty(window)
    w.fill(1.0 / window)
    x0 = (1 - window) // 2
    y = convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0)
    y *= y
    arr *= arr
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    arr -= y
    return arr


def move_nanvar_filter(arr, window, axis=-1):
    "Moving window variance ignoring NaNs, implemented with a filter."
    arr = np.array(arr, copy=False)
    global convolve1d
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
    arr = arr.astype(float)
    nrr = np.isnan(arr)
    arr[nrr] = 0
    nrr = nrr.astype(int)
    w = np.ones(window, dtype=int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    y = convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0)
    y /= (window - nrr)
    y *= y
    arr *= arr
    convolve1d(arr, w, axis=axis, mode='constant', cval=np.nan, origin=x0,
               output=arr)
    arr /= (window - nrr)
    arr -= y
    arr[nrr == window] = np.nan
    return arr

# STD -----------------------------------------------------------------------


def move_std(arr, window, axis=-1, method='loop', ddof=0):
    """
    Moving window standard deviation along the specified axis.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving standard deviation.
        By default the moving standard deviation is taken over the last
        axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving standard deviation of the input array along the specified
        axis. The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_std(arr, window=2)
    array([ NaN,  0.5,  0.5,  0.5])

    """
    arr = np.array(arr, copy=False)
    if ddof != 0:
        raise ValueError("`ddof` must be zero for unaccelerated input.")
    if method == 'filter':
        y = move_std_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(np.std, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.std, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_nanstd(arr, window, axis=-1, method='loop', ddof=0):
    """
    Moving window standard deviation along the specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving standard deviation.
        By default the moving standard deviation is taken over the last
        axis (-1).
    method : str, optional
        The following moving window methods are available:
            ==========  =====================================
            'filter'    scipy.ndimage.convolve1d
            'strides'   strides tricks
            'loop'      brute force python loop (default)
            ==========  =====================================

    Returns
    -------
    y : ndarray
        The moving standard deviation of the input array along the specified
        axis, ignoring NaNs. (A window with all NaNs returns NaN for the window
        standard deviation.) The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> bn.slow.move_nanstd(arr, window=3)
    array([ NaN,  NaN,  0.5,  1. ,  0.5])

    """
    arr = np.array(arr, copy=False)
    if ddof != 0:
        raise ValueError("`ddof` must be zero for unaccelerated input.")
    if method == 'filter':
        y = move_nanstd_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(bn.slow.nanstd, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(bn.slow.nanstd, arr, window, axis=axis)
    else:
        msg = "`method` must be 'filter', 'strides', or 'loop'."
        raise ValueError(msg)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def move_std_filter(arr, window, axis=-1):
    "Moving window standard deviation implemented with a filter."
    arr = np.array(arr, copy=False)
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    y = move_var_filter(arr, window, axis=axis)
    np.sqrt(y, y)
    return y


def move_nanstd_filter(arr, window, axis=-1):
    "Moving window standard deviation ignoring NaNs, implemented with filter."
    arr = np.array(arr, copy=False)
    if axis is None:
        raise ValueError("An `axis` value of None is not supported.")
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > arr.shape[axis]:
        raise ValueError("`window` is too long.")
    y = move_nanvar_filter(arr, window, axis=axis)
    np.sqrt(y, y)
    return y

# MIN -----------------------------------------------------------------------


def move_min(arr, window, axis=-1, method='loop'):
    """
    Slow move_min for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving minimum. By default the
        moving minimum is taken over the last axis (-1).
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
        The moving minimum of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4])
    >>> bn.slow.move_min(arr, window=2)
    array([ NaN,   1.,   2.,   3.])

    """
    if method == 'filter':
        y = move_min_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_func_strides(np.min, arr, window, axis=axis)
    elif method == 'loop':
        y = move_func_loop(np.min, arr, window, axis=axis)
    else:
        raise ValueError("`method` must be 'filter', 'strides', or 'loop'.")
    return y


def move_nanmin(arr, window, axis=-1, method='loop'):
    """
    Slow move_nanmin for unaccelerated ndim/dtype combinations.

    Parameters
    ----------
    arr : array_like
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving minimum. By default the
        moving minimum is taken over the last axis (-1).
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
        The moving minimum of the input array along the specified axis,
        ignoring NaNs. (A window with all NaNs returns NaN for the window
        minimum.) The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1, 2, np.nan, 4, 5])
    >>> bn.slow.move_nanmin(arr, window=2)
    array([ NaN,   1.,   2.,   4.,   4.])

    """
    if method == 'filter':
        y = move_nanmin_filter(arr, window, axis=axis)
    elif method == 'strides':
        y = move_nanmin_strides(arr, window, axis=axis)
    elif method == 'loop':
        y = move_nanmin_loop(arr, window, axis=axis)
    else:
        raise ValueError("`method` must be 'filter', 'strides', or 'loop'.")
    return y


def move_min_filter(arr, window, axis=-1):
    "Moving window minimium implemented with a filter."
    arr = np.array(arr, copy=False)
    global minimum_filter1d
    if minimum_filter1d is None:
        try:
            from scipy.ndimage import minimum_filter1d
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
    minimum_filter1d(y, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=y)
    return y


def move_nanmin_filter(arr, window, axis=-1):
    "Moving window minimium ignoring NaNs, implemented with a filter."
    global minimum_filter1d, convolve1d
    arr = np.array(arr, copy=False)
    if minimum_filter1d is None:
        try:
            from scipy.ndimage import minimum_filter1d
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
    arr[nrr] = np.inf
    x0 = (window - 1) // 2
    minimum_filter1d(arr, window, axis=axis, mode='constant', cval=np.nan,
                     origin=x0, output=arr)
    w = np.ones(window, dtype=int)
    nrr = nrr.astype(int)
    x0 = (1 - window) // 2
    convolve1d(nrr, w, axis=axis, mode='constant', cval=0, origin=x0,
               output=nrr)
    arr[nrr == window] = np.nan
    return arr


def move_nanmin_loop(arr, window, axis=-1):
    "Moving window minimium ignoring NaNs, implemented with a python loop."
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
    arr[nrr] = np.inf
    y = move_func_loop(np.min, arr, window, axis=axis)
    m = move_func_loop(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y


def move_nanmin_strides(arr, window, axis=-1):
    "Moving window minimium ignoring NaNs, implemented with stides tricks."
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
    arr[nrr] = np.inf
    y = move_func_strides(np.min, arr, window, axis=axis)
    m = move_func_strides(np.sum, nrr.astype(int), window, axis=axis)
    y[m == window] = np.nan
    return y

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
    for i in range(arr.shape[axis]):
        win = min(window, i + 1)
        idx1[axis] = slice(i + 1 - win, i + 1)
        idx2[axis] = i
        a = arr[idx1]
        yi = func(a, axis=axis, **kwargs)
        c = _count(a, axis)
        if yi.ndim == 0:
            if c < nmin:
                yi = np.nan
        else:
            yi[c < nmin] = np.nan
        y[idx2] = yi
    return y


def _count(a, axis=None):
    return np.sum(~np.isnan(a), axis)
