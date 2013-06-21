
import numpy as np

__all__ = ['median', 'nanmedian', 'nansum', 'nanmean', 'nanvar', 'nanstd',
           'nanmin', 'nanmax', 'nanargmin', 'nanargmax', 'rankdata',
           'nanrankdata', 'ss', 'nn', 'partsort', 'argpartsort', 'replace',
           'anynan', 'allnan']

rankdata_func = None


def median(arr, axis=None):
    "Slow median function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = np.median(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def nansum(arr, axis=None):
    "Slow nansum function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = np.nansum(arr, axis=axis)
    if not hasattr(y, "dtype"):
        y = arr.dtype.type(y)
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


def nanmean(arr, axis=None):
    "Slow nanmean function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = scipy_nanmean(arr, axis=axis)
    if y.dtype != arr.dtype:
        if issubclass(arr.dtype.type, np.inexact):
            y = y.astype(arr.dtype)
    return y


def nanvar(arr, axis=None, ddof=0):
    "Slow nanvar function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    y = nanstd(arr, axis=axis, ddof=ddof)
    return y * y


def nanstd(arr, axis=None, ddof=0):
    "Slow nanstd function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    if ddof == 0:
        bias = True
    elif ddof == 1:
        bias = False
    else:
        raise ValueError("With NaNs ddof must be 0 or 1.")
    if axis is not None:
        # Older versions of scipy can't handle negative axis?
        if axis < 0:
            axis += arr.ndim
        if (axis < 0) or (axis >= arr.ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
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
    y = np.nanmin(arr, axis=axis)
    if not hasattr(y, "dtype"):
        # Numpy 1.5.1 doesn't return object with dtype when input is all NaN
        y = arr.dtype.type(y)
    return y


def nanmax(arr, axis=None):
    "Slow nanmax function used for unaccelerated ndim/dtype combinations."
    y = np.nanmax(arr, axis=axis)
    if not hasattr(y, "dtype"):
        # Numpy 1.5.1 doesn't return object with dtype when input is all NaN
        y = arr.dtype.type(y)
    return y


def nanargmin(arr, axis=None):
    "Slow nanargmin function used for unaccelerated ndim/dtype combinations."
    return np.nanargmin(arr, axis=axis)


def nanargmax(arr, axis=None):
    "Slow nanargmax function used for unaccelerated ndim/dtype combinations."
    return np.nanargmax(arr, axis=axis)


def rankdata(arr, axis=None):
    "Slow rankdata function used for unaccelerated ndim/dtype combinations."
    global rankdata_func
    if rankdata_func is None:
        try:
            # Use scipy's rankdata; newer scipy has cython version
            from scipy.stats import rankdata as imported_rankdata
            rankdata_func = imported_rankdata
        except ImportError:
            # Use a local copy of scipy's python (not cython) rankdata
            rankdata_func = scipy_rankdata
    arr = np.asarray(arr)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    elif axis < 0:
        axis = range(arr.ndim)[axis]
    y = np.empty(arr.shape)
    itshape = list(arr.shape)
    itshape.pop(axis)
    for ij in np.ndindex(*itshape):
        ijslice = list(ij[:axis]) + [slice(None)] + list(ij[axis:])
        y[ijslice] = rankdata_func(arr[ijslice].astype('float'))
    return y


def nanrankdata(arr, axis=None):
    "Slow nanrankdata function used for unaccelerated ndim/dtype combinations."
    arr = np.asarray(arr)
    if axis is None:
        arr = arr.ravel()
        axis = 0
    elif axis < 0:
        axis = range(arr.ndim)[axis]
    y = np.empty(arr.shape)
    y.fill(np.nan)
    itshape = list(arr.shape)
    itshape.pop(axis)
    for ij in np.ndindex(*itshape):
        ijslice = list(ij[:axis]) + [slice(None)] + list(ij[axis:])
        x1d = arr[ijslice].astype(float)
        mask1d = ~np.isnan(x1d)
        x1d[mask1d] = scipy_rankdata(x1d[mask1d])
        y[ijslice] = x1d
    return y


def ss(arr, axis=0):
    "Slow sum of squares used for unaccelerated ndim/dtype combinations."
    return scipy_ss(arr, axis)


def nn(arr, arr0, axis=1):
    "Slow nearest neighbor used for unaccelerated ndim/dtype combinations."
    arr = np.array(arr, copy=False)
    arr0 = np.array(arr0, copy=False)
    if arr.ndim != 2:
        raise ValueError("`arr` must be 2d")
    if arr0.ndim != 1:
        raise ValueError("`arr0` must be 1d")
    if axis == 1:
        d = (arr - arr0) ** 2
    elif axis == 0:
        d = (arr - arr0.reshape(-1, 1)) ** 2
    else:
        raise ValueError("`axis` must be 0 or 1.")
    d = d.sum(axis)
    idx = np.argmin(d)
    return np.sqrt(d[idx]), idx


def partsort(arr, n, axis=-1):
    "Slow partial sort used for unaccelerated ndim/dtype combinations."
    return np.sort(arr, axis)


def argpartsort(arr, n, axis=-1):
    "Slow partial argsort used for unaccelerated ndim/dtype combinations."
    return np.argsort(arr, axis)


def replace(arr, old, new):
    "Slow replace (inplace) used for unaccelerated ndim/dtype combinations."
    if type(arr) is not np.ndarray:
        raise TypeError("`arr` must be a numpy array.")
    if not issubclass(arr.dtype.type, np.inexact):
        if old != old:
            # int arrays do not contain NaN
            return
        if int(old) != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if int(new) != new:
            raise ValueError("Cannot safely cast `new` to int.")
    if old != old:
        mask = np.isnan(arr)
    else:
        mask = arr == old
    np.putmask(arr, mask, new)


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
    x, axis = _chk_asarray(x, axis)
    x = x.copy()
    Norig = x.shape[axis]
    factor = 1.0-np.sum(np.isnan(x), axis)*1.0/Norig

    x[np.isnan(x)] = 0
    return np.mean(x, axis)/factor


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
    x, axis = _chk_asarray(x, axis)
    x = x.copy()
    Norig = x.shape[axis]

    Nnan = np.sum(np.isnan(x), axis)*1.0
    n = Norig - Nnan

    x[np.isnan(x)] = 0.
    m1 = np.sum(x, axis)/n

    if axis:
        d = (x - np.expand_dims(m1, axis))**2.0
    else:
        d = (x - m1)**2.0

    m2 = np.sum(d, axis)-(m1*m1)*Nnan
    if bias:
        m2c = m2 / n
    else:
        m2c = m2 / (n - 1.)
    return np.sqrt(m2c)


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


def fastsort(a):
    """
    Sort an array and provide the argsort.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    fastsort : ndarray of type int
        sorted indices into the original array

    """
    # TODO: the wording in the docstring is nonsense.
    it = np.argsort(a)
    as_ = a[it]
    return as_, it


def scipy_rankdata(a):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Parameters
    ----------
    a : array_like
        This array is first flattened.

    Returns
    -------
    rankdata : ndarray
         An array of length equal to the size of `a`, containing rank scores.

    Examples
    --------
    >>> scipy_rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])

    """
    a = np.ravel(a)
    n = len(a)
    svec, ivec = fastsort(a)
    sumranks = 0
    dupcount = 0
    newarray = np.zeros(n, float)
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i == n-1 or svec[i] != svec[i+1]:
            averank = sumranks / float(dupcount) + 1
            for j in range(i-dupcount+1, i+1):
                newarray[ivec[j]] = averank
            sumranks = 0
            dupcount = 0
    return newarray


def scipy_ss(a, axis=0):
    """
    Squares each element of the input array, and returns the square(s) of that.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int or None, optional
        The axis along which to calculate. If None, use whole array.
        Default is 0, i.e. along the first axis.

    Returns
    -------
    ss : ndarray
        The sum along the given axis for (a**2).

    See also
    --------
    square_of_sums : The square(s) of the sum(s) (the opposite of `ss`).

    Examples
    --------
    >>> from scipy import stats
    >>> a = np.array([1., 2., 5.])
    >>> stats.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> stats.ss(b, axis=1)
    array([ 30., 65.])

    """
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)
