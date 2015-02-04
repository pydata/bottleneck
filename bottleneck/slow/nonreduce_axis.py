import numpy as np

__all__ = ['rankdata', 'nanrankdata', 'partsort', 'argpartsort']

rankdata_func = None


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


def partsort(arr, n, axis=-1):
    "Slow partial sort used for unaccelerated ndim/dtype combinations."
    return np.partition(arr, n - 1, axis)


def argpartsort(arr, n, axis=-1):
    "Slow partial argsort used for unaccelerated ndim/dtype combinations."
    if type(arr) is np.ndarray:
        a = arr
    else:
        # bug in numpy 1.9.1: `a` cannot be a list
        a = np.array(arr, copy=False)
    return np.argpartition(a, n - 1, axis)


# ---------------------------------------------------------------------------
#
# SciPy
#
# Local copy of scipy.stats functions to avoid (by popular demand) a SciPy
# dependency. The SciPy license is included in the Bottleneck license file,
# which is distributed with Bottleneck.
#
# Code taken from scipy trunk on Dec 16, 2010.


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
