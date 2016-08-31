import numpy as np

__all__ = ['rankdata', 'nanrankdata', 'partsort', 'argpartsort', 'push']

rankdata_func = None


def rankdata(arr, axis=None):
    "Slow rankdata function used for unaccelerated dtypes."
    global rankdata_func
    if rankdata_func is None:
        try:
            # Use scipy's rankdata
            from scipy.stats import rankdata as imported_rankdata
            rankdata_func = imported_rankdata
        except ImportError:
            # Use a local copy of scipy's rankdata
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
    "Slow nanrankdata function used for unaccelerated dtypes."
    global rankdata_func
    if rankdata_func is None:
        try:
            # Use scipy's rankdata
            from scipy.stats import rankdata as imported_rankdata
            rankdata_func = imported_rankdata
        except ImportError:
            # Use a local copy of scipy's rankdata
            rankdata_func = scipy_rankdata
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
        x1d[mask1d] = rankdata_func(x1d[mask1d])
        y[ijslice] = x1d
    return y


def partsort(arr, n, axis=-1):
    "Slow partial sort used for unaccelerated dtypes."
    return np.partition(arr, n - 1, axis)


def argpartsort(arr, n, axis=-1):
    "Slow partial argsort used for unaccelerated dtypes."
    if type(arr) is np.ndarray:
        a = arr
    else:
        # bug in numpy 1.9.1: `a` cannot be a list
        a = np.array(arr, copy=False)
    return np.argpartition(a, n - 1, axis)


def push(arr, n=np.inf, axis=-1):
    "Slow push used for unaccelerated dtypes."
    if axis is None:
        raise ValueError("`axis` cannot be None")
    y = np.array(arr)
    ndim = y.ndim
    if axis != -1 or axis != ndim - 1:
        y = np.rollaxis(y, axis, ndim)
    if ndim == 1:
        y = y[None, :]
    elif ndim == 0:
        return y
    fidx = ~np.isnan(y)
    recent = np.empty(y.shape[:-1])
    count = np.empty(y.shape[:-1])
    recent.fill(np.nan)
    count.fill(np.nan)
    with np.errstate(invalid='ignore'):
        for i in range(y.shape[-1]):
            idx = (i - count) > n
            recent[idx] = np.nan
            idx = ~fidx[..., i]
            y[idx, i] = recent[idx]
            idx = fidx[..., i]
            count[idx] = i
            recent[idx] = y[idx, i]
    if axis != -1 or axis != ndim - 1:
        y = np.rollaxis(y, ndim - 1, axis)
    if ndim == 1:
        return y[0]
    return y


# ---------------------------------------------------------------------------
#
# SciPy
#
# Local copy of SciPy's rankdata to avoid a SciPy dependency. The SciPy
# license is included in the Bottleneck license file, which is distributed
# with Bottleneck.
#
# Code taken from scipy master branch on Aug 31, 2016.


def scipy_rankdata(a, method='average'):
    """
    rankdata(a, method='average')
    Assign ranks to data, dealing with ties appropriately.
    Ranks begin at 1.  The `method` argument controls how ranks are assigned
    to equal values.  See [1]_ for further discussion of ranking methods.
    Parameters
    ----------
    a : array_like
        The array of values to be ranked.  The array is first flattened.
    method : str, optional
        The method used to assign ranks to tied elements.
        The options are 'average', 'min', 'max', 'dense' and 'ordinal'.
        'average':
            The average of the ranks that would have been assigned to
            all the tied values is assigned to each value.
        'min':
            The minimum of the ranks that would have been assigned to all
            the tied values is assigned to each value.  (This is also
            referred to as "competition" ranking.)
        'max':
            The maximum of the ranks that would have been assigned to all
            the tied values is assigned to each value.
        'dense':
            Like 'min', but the rank of the next highest element is assigned
            the rank immediately after those assigned to the tied elements.
        'ordinal':
            All values are given a distinct rank, corresponding to the order
            that the values occur in `a`.
        The default is 'average'.
    Returns
    -------
    ranks : ndarray
         An array of length equal to the size of `a`, containing rank
         scores.
    References
    ----------
    .. [1] "Ranking", http://en.wikipedia.org/wiki/Ranking
    Examples
    --------
    >>> from scipy.stats import rankdata
    >>> rankdata([0, 2, 3, 2])
    array([ 1. ,  2.5,  4. ,  2.5])
    >>> rankdata([0, 2, 3, 2], method='min')
    array([ 1,  2,  4,  2])
    >>> rankdata([0, 2, 3, 2], method='max')
    array([ 1,  3,  4,  3])
    >>> rankdata([0, 2, 3, 2], method='dense')
    array([ 1,  2,  3,  2])
    >>> rankdata([0, 2, 3, 2], method='ordinal')
    array([ 1,  2,  4,  3])
    """
    if method not in ('average', 'min', 'max', 'dense', 'ordinal'):
        raise ValueError('unknown method "{0}"'.format(method))

    arr = np.ravel(np.asarray(a))
    algo = 'mergesort' if method == 'ordinal' else 'quicksort'
    sorter = np.argsort(arr, kind=algo)

    inv = np.empty(sorter.size, dtype=np.intp)
    inv[sorter] = np.arange(sorter.size, dtype=np.intp)

    if method == 'ordinal':
        return inv + 1

    arr = arr[sorter]
    obs = np.r_[True, arr[1:] != arr[:-1]]
    dense = obs.cumsum()[inv]

    if method == 'dense':
        return dense

    # cumulative counts of each unique value
    count = np.r_[np.nonzero(obs)[0], len(obs)]

    if method == 'max':
        return count[dense]

    if method == 'min':
        return count[dense - 1] + 1

    # average method
    return .5 * (count[dense] + count[dense - 1] + 1)
