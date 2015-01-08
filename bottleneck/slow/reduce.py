import warnings
import numpy as np

__all__ = ['median', 'nanmedian', 'nansum', 'nanmean', 'nanvar', 'nanstd',
           'nanmin', 'nanmax', 'nanargmin', 'nanargmax', 'ss', 'anynan',
           'allnan']

rankdata_func = None

from numpy import nansum, nanmean, nanstd, nanvar, nanmin, nanmax


def median(arr, axis=None):
    "Slow median function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = np.median(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def nanmedian(arr, axis=None):
    "Slow nanmedian function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = scipy_nanmedian(arr, axis=axis)
    if not hasattr(y, "dtype"):
        if issubclass(arr.dtype.type, np.inexact):
            y = arr.dtype.type(y)
        else:
            y = np.float64(y)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    if (y.size == 1) and (y.ndim == 0):
        y = y[()]
    return y


def nanargmin(arr, axis=None):
    "Slow nanargmin function used for unaccelerated ndim/dtype combinations."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmin(arr, axis=axis)


def nanargmax(arr, axis=None):
    "Slow nanargmax function used for unaccelerated ndim/dtype combinations."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmax(arr, axis=axis)


def ss(arr, axis=None):
    "Slow sum of squares used for unaccelerated ndim/dtype combinations."
    return np.multiply(arr, arr).sum(axis)


def anynan(arr, axis=None):
    "Slow check for Nans used for unaccelerated ndim/dtype combinations."
    return np.isnan(arr).any(axis)


def allnan(arr, axis=None):
    "Slow check for all Nans used for unaccelerated ndim/dtype combinations."
    return np.isnan(arr).all(axis)

# ---------------------------------------------------------------------------
#
# SciPy
#
# Local copy of scipy.stats functions to avoid (by popular demand) a SciPy
# dependency. The SciPy license is included in the Bottleneck license file,
# which is distributed with Bottleneck.
#
# Code taken from scipy trunk on Dec 16, 2010.
# nanmedian taken from scipy trunk on Dec 17, 2010.
# rankdata taken from scipy HEAD on Mar 16, 2011.


def _nanmedian(arr1d):  # This only works on 1d arrays
    """Private function for rank a arrays. Compute the median ignoring Nan.

    Parameters
    ----------
    arr1d : ndarray
        Input array, of rank 1.

    Results
    -------
    m : float
        The median.
    """
    cond = 1-np.isnan(arr1d)
    x = np.sort(np.compress(cond, arr1d, axis=-1))
    if x.size == 0:
        return np.nan
    return np.median(x)


# Feb 2011: patched nanmedian to handle nanmedian(a, 1) with a = np.ones((2,0))
def scipy_nanmedian(x, axis=0):
    """
    Compute the median along the given axis ignoring nan values.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the median is computed. Default is 0, i.e. the
        first axis.

    Returns
    -------
    m : float
        The median of `x` along `axis`.

    See Also
    --------
    nanstd, nanmean

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([0, 3, 1, 5, 5, np.nan])
    >>> stats.nanmedian(a)
    array(3.0)

    >>> b = np.array([0, 3, 1, 5, 5, np.nan, 5])
    >>> stats.nanmedian(b)
    array(4.0)

    Example with axis:

    >>> c = np.arange(30.).reshape(5,6)
    >>> idx = np.array([False, False, False, True, False] * 6).reshape(5,6)
    >>> c[idx] = np.nan
    >>> c
    array([[  0.,   1.,   2.,  nan,   4.,   5.],
           [  6.,   7.,  nan,   9.,  10.,  11.],
           [ 12.,  nan,  14.,  15.,  16.,  17.],
           [ nan,  19.,  20.,  21.,  22.,  nan],
           [ 24.,  25.,  26.,  27.,  nan,  29.]])
    >>> stats.nanmedian(c, axis=1)
    array([  2. ,   9. ,  15. ,  20.5,  26. ])

    """
    x, axis = _chk_asarray(x, axis)
    if x.ndim == 0:
        return float(x.item())
    shape = list(x.shape)
    shape.pop(axis)
    if 0 in shape:
        x = np.empty(shape)
    else:
        x = x.copy()
        x = np.apply_along_axis(_nanmedian, axis, x)
        if x.ndim == 0:
            x = float(x.item())
    return x


def _chk_asarray(a, axis):
    if axis is None:
        a = np.ravel(a)
        outaxis = 0
    else:
        a = np.asarray(a)
        outaxis = axis
    return a, outaxis
