"nanmin template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanmin"]

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
    "Minimum of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int allnan = 1
    cdef np.DTYPE_t amin, ai
"""

loop = {}
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        amin = MAXDTYPE
        allnan = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[INDEXPOP] = amin
        else:
            y[INDEXPOP] = NAN
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does." 
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            amin = MAXDTYPE
            allnan = 1
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[INDEXPOP] = amin
            else:
                y[INDEXPOP] = NAN
    return y
"""

floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan == 0:       
        return np.DTYPE(amin)
    else:
        return NAN
"""
loop[2] = """\
    if nINDEX0 * nINDEX1 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
                allnan = 0
    if allnan == 0:       
        return np.DTYPE(amin)
    else:
        return NAN
"""
loop[3] = """\
    if nINDEX0 * nINDEX1 * nINDEX2 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
                    allnan = 0
    if allnan == 0:       
        return np.DTYPE(amin)
    else:
        return NAN
"""

floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 

loop = {}
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        amin = MAXDTYPE
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
        y[INDEXPOP] = amin
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            amin = MAXDTYPE
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
            y[INDEXPOP] = amin
    return y
"""

ints['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(ints) 
ints_None['axisNone'] = True

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai <= amin:
            amin = ai
    return np.DTYPE(amin)
"""
loop[2] = """\
    if nINDEX0 * nINDEX1 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
    return np.DTYPE(amin)
"""
loop[3] = """\
    if nINDEX0 * nINDEX1 * nINDEX2 == 0:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
    return np.DTYPE(amin)
"""

ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanmin"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanmin(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanmin = {}
nanmin['name'] = 'nanmin'
nanmin['is_reducing_function'] = True
nanmin['cdef_output'] = True
nanmin['slow'] = slow
nanmin['templates'] = {}
nanmin['templates']['float'] = floats
nanmin['templates']['float_None'] = floats_None
nanmin['templates']['int'] = ints
nanmin['templates']['int_None'] = ints_None
nanmin['pyx_file'] = 'func/nanmin.pyx'

nanmin['main'] = '''"nanmin auto-generated from template"

def nanmin(arr, axis=None):
    """
    Minimum values along specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is computed. The default (axis=None) is
        to compute the minimum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    See also
    --------
    bottleneck.nanmax: Maximum along specified axis, ignoring NaNs.
    bottleneck.nanargmin: Indices of minimum values along axis, ignoring NaNs. 

    Examples
    --------
    >>> bn.nanmin(1)
    1
    >>> bn.nanmin([1])
    1
    >>> bn.nanmin([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmin(a)
    1.0
    >>> bn.nanmin(a, axis=0)
    array([ 1.,  4.])
    
    """
    func, arr = nanmin_selector(arr, axis)
    return func(arr)

def nanmin_selector(arr, axis):
    """
    Return nanmin function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanmin()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use
    to calculate the minimum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is to be computed. The default
        (axis=None) is to compute the minimum of the flattened array.
    
    Returns
    -------
    func : function
        The nanmin function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to find
        the minimum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the minimum of `arr` along
    axis=0:

    >>> func, a = bn.func.nanmin_selector(arr, axis=0)
    >>> func
    <built-in function nanmin_1d_float64_axis0> 
    
    Use the returned function and array to determine the minimum:
    
    >>> func(a)
    1.0

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if (axis < 0) and (axis is not None):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmin_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmin_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
