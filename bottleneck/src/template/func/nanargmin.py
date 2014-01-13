"nanargmin template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanargmin"]

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
    cdef np.DTYPE_t amin, ai
    cdef Py_ssize_t idx = 0
"""

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        ai = a[INDEXALL]
        if ai <= amin:
            amin = ai
            allnan = 0
            idx = iINDEX0
    if allnan == 0:
        return np.intp(idx)
    else:
        raise ValueError("All-NaN slice encountered")
"""
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        amin = MAXDTYPE
        allnan = 1
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
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
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            amin = MAXDTYPE
            allnan = 1
            for iINDEX2 in range(nINDEX2 - 1, -1, -1):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
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
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai < amin:
            amin = ai
            idx = iINDEX0
    return np.intp(idx)
"""
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        amin = MAXDTYPE
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
                idx = iINDEX1
        y[INDEXPOP] = idx
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0 - 1, -1, -1):
        for iINDEX1 in range(nINDEX1 - 1, -1, -1):
            amin = MAXDTYPE
            for iINDEX2 in range(nINDEX2 - 1, - 1, -1):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
                    idx = iINDEX2
            y[INDEXPOP] = idx
    return y
"""

ints['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanargmin"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanargmin(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanargmin = {}
nanargmin['name'] = 'nanargmin'
nanargmin['is_reducing_function'] = True
nanargmin['cdef_output'] = True
nanargmin['slow'] = slow
nanargmin['templates'] = {}
nanargmin['templates']['float'] = floats
nanargmin['templates']['int'] = ints
nanargmin['pyx_file'] = 'func/nanargmin.pyx'

nanargmin['main'] = '''"nanargmin auto-generated from template"

def nanargmin(arr, axis=None):
    """
    Indices of the minimum values along an axis, ignoring NaNs.

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
    bottleneck.nanargmax: Indices of the maximum values along an axis.
    bottleneck.nanmin: Minimum values along specified axis, ignoring NaNs.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.

    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> bn.nanargmin(a)
    2
    >>> a.flat[1]
    2.0
    >>> bn.nanargmax(a, axis=0)
    array([1, 1])
    >>> bn.nanargmax(a, axis=1)
    array([1, 0])

    """
    func, arr = nanargmin_selector(arr, axis)
    return func(arr)

def nanargmin_selector(arr, axis):
    """
    Return nanargmin function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanargmin() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to find the indices of the minimum.

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
        The nanargmin function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])

    Obtain the function needed to determine the nanargmin of `arr` along
    axis=0:

    >>> func, a = bn.func.nanargmin_selector(arr, axis=0)
    >>> func
    <function nanargmin_1d_float64_axis0>

    Use the returned function and array to determine the maximum:

    >>> func(a)
    0

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
        func = nanargmin_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nanargmin_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
