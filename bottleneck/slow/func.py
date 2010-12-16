
import numpy as np

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
    y = scipy_nanmean(arr, axis=axis)
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
    y = scipy_nanstd(arr, axis=axis, bias=bias)
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


# ---------------------------------------------------------------------------
#
# SciPy
#
# Local copy of scipy.stats functions to avoid (by popular demand) a SciPy
# dependency. The SciPy license is included in the Bottleneck license file,
# which is distributed with Bottleneck.
#
# Code taken from scipy trunk on Dec 16, 2010.

def scipy_nanmean(x, axis=0):
    """
    Compute the mean over the given axis ignoring nans.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int, optional
        Axis along which the mean is computed. Default is 0, i.e. the
        first axis.

    Returns
    -------
    m : float
        The mean of `x`, ignoring nans.

    See Also
    --------
    nanstd, nanmedian

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.linspace(0, 4, 3)
    >>> a
    array([ 0.,  2.,  4.])
    >>> a[-1] = np.nan
    >>> stats.nanmean(a)
    1.0

    """
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0-np.sum(np.isnan(x),axis)*1.0/Norig

    x[np.isnan(x)] = 0
    return np.mean(x,axis)/factor

def scipy_nanstd(x, axis=0, bias=False):
    """
    Compute the standard deviation over the given axis, ignoring nans.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int or None, optional
        Axis along which the standard deviation is computed. Default is 0.
        If None, compute over the whole array `x`.
    bias : bool, optional
        If True, the biased (normalized by N) definition is used. If False
        (default), the unbiased definition is used.

    Returns
    -------
    s : float
        The standard deviation.

    See Also
    --------
    nanmean, nanmedian

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.arange(10, dtype=float)
    >>> a[1:3] = np.nan
    >>> np.std(a)
    nan
    >>> stats.nanstd(a)
    2.9154759474226504
    >>> stats.nanstd(a.reshape(2, 5), axis=1)
    array([ 2.0817,  1.5811])
    >>> stats.nanstd(a.reshape(2, 5), axis=None)
    2.9154759474226504

    """
    x, axis = _chk_asarray(x,axis)
    x = x.copy()
    Norig = x.shape[axis]

    Nnan = np.sum(np.isnan(x),axis)*1.0
    n = Norig - Nnan

    x[np.isnan(x)] = 0.
    m1 = np.sum(x,axis)/n

    if axis:
        d = (x - np.expand_dims(m1, axis))**2.0
    else:
        d = (x - m1)**2.0

    m2 = np.sum(d,axis)-(m1*m1)*Nnan
    if bias:
        m2c = m2 / n
    else:
        m2c = m2 / (n - 1.)
    return np.sqrt(m2c)

def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis
