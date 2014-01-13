"nanargmax template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanargmax"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'intp'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Index of max of NDIMd, DTYPE array along axis=AXIS ignoring NaNs."
    cdef int allnan = 1
    cdef np.DTYPE_t amax, ai
    cdef Py_ssize_t idx = 0
"""

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        ai = a[INDEXALL]
        if ai >= amax:
            amax = ai
            allnan = 0
            idx = iINDEX0
    if allnan == 0:
        return np.intp(idx)
    else:
        raise ValueError("All-NaN slice encountered")
"""
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        amax = MINDTYPE
        allnan = 1
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = iINDEX1
        if allnan == 0:
            y[INDEXPOP] = idx
        else:
            raise ValueError("All-NaN slice encountered")
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            amax = MINDTYPE
            allnan = 1
            for iINDEX2 in range(nINDEX2 - 1, -1, -1):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = iINDEX2
            if allnan == 0:
                y[INDEXPOP] = idx
            else:
                raise ValueError("All-NaN slice encountered")
    return y
"""

floats['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai > amax:
            amax = ai
            idx = iINDEX0
    return np.intp(idx)
"""
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        amax = MINDTYPE
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
                idx = iINDEX1
        y[INDEXPOP] = idx
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            amax = MINDTYPE
            for iINDEX2 in range(nINDEX2 - 1, - 1, -1):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
                    idx = iINDEX2
            y[INDEXPOP] = idx
    return y
"""

ints['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanargmax"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanargmax(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanargmax = {}
nanargmax['name'] = 'nanargmax'
nanargmax['is_reducing_function'] = True
nanargmax['cdef_output'] = True
nanargmax['slow'] = slow
nanargmax['templates'] = {}
nanargmax['templates']['float'] = floats
nanargmax['templates']['int'] = ints
nanargmax['pyx_file'] = 'func/nanargmax.pyx'

nanargmax['main'] = '''"nanargmax auto-generated from template"

def nanargmax(arr, axis=None):
    """
    Indices of the maximum values along an axis, ignoring NaNs.

    For all-NaN slices ``ValueError`` is raised. Unlike NumPy, the results
    can be trusted if a slice contains only NaNs and Infs.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : {int, None}, optional
        Axis along which to operate. By default (axis=None) flattened input
        is used.

    See also
    --------
    bottleneck.nanargmin: Indices of the minimum values along an axis.
    bottleneck.nanmax: Maximum values along specified axis, ignoring NaNs.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> bn.nanargmax(a)
    1
    >>> a.flat[1]
    4.0
    >>> bn.nanargmax(a, axis=0)
    array([1, 0])
    >>> bn.nanargmax(a, axis=1)
    array([1, 1])

    """
    func, arr = nanargmax_selector(arr, axis)
    return func(arr)

def nanargmax_selector(arr, axis):
    """
    Return nanargmax function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanargmax() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to find the indices of the maximum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the indices are found.

    Returns
    -------
    func : function
        The nanargmax function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])

    Obtain the function needed to determine the nanargmax of `arr` along
    axis=0:

    >>> func, a = bn.func.nanargmax_selector(arr, axis=0)
    >>> func
    <function nanargmax_1d_float64_axis0>

    Use the returned function and array to determine the maximum:

    >>> func(a)
    2

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis = 0
        ndim = 1
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanargmax_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nanargmax_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
