"rankdata template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["rankdata"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loops ---------------------------------------------------------------------

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    old = a[ivec[0]]
    for iINDEX0 in xrange(nINDEX0-1):
        sumranks += iINDEX0
        dupcount += 1
        k = iINDEX0 + 1
        new = a[INDEXREPLACE|ivec[k]|]
        if old != new:
            averank = sumranks / dupcount + 1
            for j in xrange(k - dupcount, k):
                y[INDEXREPLACE|ivec[j]|] = averank
            sumranks = 0
            dupcount = 0
        old = new
    sumranks += (nINDEX0 - 1)
    dupcount += 1
    averank = sumranks / dupcount + 1
    for j in xrange(nINDEX0 - dupcount, nINDEX0):
        y[INDEXREPLACE|ivec[j]|] = averank
    return y
"""
loop[2] = """\
    if nINDEX1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for iINDEX0 in xrange(nINDEX0):
        idx = ivec[INDEXREPLACE|0|]
        old = a[INDEXREPLACE|idx|]
        sumranks = 0
        dupcount = 0
        for iINDEX1 in xrange(nINDEX1-1):
            sumranks += iINDEX1
            dupcount += 1
            k = iINDEX1 + 1
            idx = ivec[INDEXREPLACE|k|]
            new = a[INDEXREPLACE|idx|]
            if old != new:
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    idx = ivec[INDEXREPLACE|j|]
                    y[INDEXREPLACE|idx|] = averank
                sumranks = 0
                dupcount = 0
            old = new
        sumranks += (nINDEX1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        for j in xrange(nINDEX1 - dupcount, nINDEX1):
            idx = ivec[INDEXREPLACE|j|]
            y[INDEXREPLACE|idx|] = averank
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for iINDEX0 in xrange(nINDEX0):
        for iINDEX1 in xrange(nINDEX1):
            idx = ivec[INDEXREPLACE|0|]
            old = a[INDEXREPLACE|idx|]
            sumranks = 0
            dupcount = 0
            for iINDEX2 in xrange(nINDEX2-1):
                sumranks += iINDEX2
                dupcount += 1
                k = iINDEX2 + 1
                idx = ivec[INDEXREPLACE|k|]
                new = a[INDEXREPLACE|idx|]
                if old != new:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[INDEXREPLACE|j|]
                        y[INDEXREPLACE|idx|] = averank
                    sumranks = 0
                    dupcount = 0
                old = new
            sumranks += (nINDEX2 - 1)
            dupcount += 1
            averank = sumranks / dupcount + 1
            for j in xrange(nINDEX2 - dupcount, nINDEX2):
                idx = ivec[INDEXREPLACE|j|]
                y[INDEXREPLACE|idx|] = averank
    return y
"""

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'float64'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Ranks nNDIMd array with dtype=DTYPE along axis=AXIS, dealing with ties."
    cdef Py_ssize_t dupcount = 0
    cdef Py_ssize_t j, k, idx
    cdef np.ndarray[np.intp_t, ndim=NDIM] ivec = PyArray_ArgSort(a, AXIS, NPY_QUICKSORT)  # noqa
    cdef np.float64_t old, new, averank, sumranks = 0
"""

floats['loop'] = deepcopy(loop)

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES
ints['force_output_dtype'] = 'float64'
ints['loop'] = deepcopy(loop)

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "rankdata"
slow['signature'] = "arr"
slow['func'] = "bn.slow.rankdata(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

rankdata = {}
rankdata['name'] = 'rankdata'
rankdata['is_reducing_function'] = False
rankdata['cdef_output'] = True
rankdata['slow'] = slow
rankdata['templates'] = {}
rankdata['templates']['float'] = floats
rankdata['templates']['int'] = ints
rankdata['pyx_file'] = 'func/rankdata.pyx'

rankdata['main'] = '''"rankdata auto-generated from template"

def rankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the elements of the array are ranked. The default
        (axis=None) is to rank the elements of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`. The dtype is 'float64'.

    See also
    --------
    bottleneck.nanrankdata: Ranks the data dealing with ties and NaNs.

    Examples
    --------
    >>> bn.rankdata([0, 2, 2, 3])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]])
    array([ 1. ,  2.5,  2.5,  4. ])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=0)
    array([[ 1.,  1.],
           [ 2.,  2.]])
    >>> bn.rankdata([[0, 2], [2, 3]], axis=1)
    array([[ 1.,  2.],
           [ 1.,  2.]])

    """
    func, arr = rankdata_selector(arr, axis)
    return func(arr)

def rankdata_selector(arr, axis):
    """
    Return rankdata function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.rankdata() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to rank the elements.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which to rank the elements of the array.

    Returns
    -------
    func : function
        The rankdata function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to rank.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([0, 2, 2, 3])

    Obtain the function needed to rank the elements of `arr` along axis=0:

    >>> func, a = bn.func.rankdata_selector(arr, axis=0)
    >>> func
    <function rankdata_1d_int64_axis0>

    Use the returned function and array:

    >>> func(a)
    array([ 1. ,  2.5,  2.5,  4. ])

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef tuple key
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    try:
        func = rankdata_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = rankdata_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
