"nansum template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nansum"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]


# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = False
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Sum of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int allnan = 1
    cdef np.DTYPE_t asum = 0, ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        asum = 0
        allnan = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                allnan = 0
        if allnan == 0:
            y[INDEXPOP] = asum
        else:
            y[INDEXPOP] = NAN
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            allnan = 1
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:
                y[INDEXPOP] = asum
            else:
                y[INDEXPOP] = NAN
    return y
"""
floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

returns = """\
    if allnan == 0:
        return np.DTYPE(asum)
    else:
        return np.DTYPE(NAN)
"""

loop = {}
loop[1] = """\
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            allnan = 0
""" + returns
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                allnan = 0
""" + returns
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    allnan = 0
""" + returns
floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

ints['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Sum of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef np.DTYPE_t asum = 0, ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        asum = 0
        for iINDEX1 in range(nINDEX1):
            asum += a[INDEXALL]
        y[INDEXPOP] = asum
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            for iINDEX2 in range(nINDEX2):
                asum += a[INDEXALL]
            y[INDEXPOP] = asum
    return y
"""
ints['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(ints)
ints_None['top'] = ints['top'] + "    cdef Py_ssize_t size\n"
ints_None['axisNone'] = True

loop = {}
loop[1] = """\
    size = nINDEX0
    for iINDEX0 in range(nINDEX0):
        asum += a[INDEXALL]
    return np.DTYPE(asum)
"""
loop[2] = """\
    size = nINDEX0 * nINDEX1
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum += a[INDEXALL]
    return np.DTYPE(asum)
"""
loop[3] = """\
    size = nINDEX0 * nINDEX1 * nINDEX2
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                asum += a[INDEXALL]
    return np.DTYPE(asum)
"""
ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nansum"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nansum(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nansum = {}
nansum['name'] = 'nansum'
nansum['is_reducing_function'] = True
nansum['cdef_output'] = True
nansum['slow'] = slow
nansum['templates'] = {}
nansum['templates']['float'] = floats
nansum['templates']['float_None'] = floats_None
nansum['templates']['int'] = ints
nansum['templates']['int_None'] = ints_None
nansum['pyx_file'] = 'func/nansum.pyx'

nansum['main'] = '''"nansum auto-generated from template"

def nansum(arr, axis=None):
    """
    Sum of array elements along given axis ignoring NaNs.

    When the input has an integer type with less precision than the default
    platform integer, the default platform integer is used for the
    accumulator and return values.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose sum is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is computed. The default (axis=None) is to
        compute the sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    Notes
    -----
    No error is raised on overflow.

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> bn.nansum(1)
    1
    >>> bn.nansum([1])
    1
    >>> bn.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> bn.nansum(a)
    3.0
    >>> bn.nansum(a, axis=0)
    array([ 2.,  1.])

    When positive infinity and negative infinity are present:

    >>> bn.nansum([1, np.nan, np.inf])
    inf
    >>> bn.nansum([1, np.nan, np.NINF])
    -inf
    >>> bn.nansum([1, np.nan, np.inf, np.NINF])
    nan

    """
    func, arr = nansum_selector(arr, axis)
    return func(arr)

def nansum_selector(arr, axis):
    """
    Return nansum function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nansum() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the sum is to be computed.

    Returns
    -------
    func : function
        The nansum function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the sum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, np.nan, 3.0])

    Obtain the function needed to determine the nansum of `arr` along axis=0:

    >>> func, a = bn.func.nansum_selector(arr, axis=0)
    >>> func
    <function nansum_1d_float64_axis0>

    Use the returned function and array to determine the sum:

    >>> func(a)
    4.0

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if dtype < NPY_int_:
        a = a.astype(np.int_)
        dtype = PyArray_TYPE(a)
    if (axis is not None) and (axis < 0):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nansum_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nansum_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
