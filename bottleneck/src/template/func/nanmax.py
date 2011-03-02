"nanmax template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanmax"]

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
    "Maximum of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int allnan = 1
    cdef np.DTYPE_t amax, ai
"""

loop = {}
loop[2] = """\
    if nINDEX1 == 0:
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does." 
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        amax = MINDTYPE
        allnan = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[INDEXPOP] = amax
        else:
            y[INDEXPOP] = NAN
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does." 
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            amax = MINDTYPE
            allnan = 1
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[INDEXPOP] = amax
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
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan == 0:       
        return np.DTYPE(amax)
    else:
        return NAN
"""
loop[2] = """\
    if nINDEX0 * nINDEX1 == 0:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
                allnan = 0
    if allnan == 0:       
        return np.DTYPE(amax)
    else:
        return NAN
"""
loop[3] = """\
    if nINDEX0 * nINDEX1 * nINDEX2 == 0:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too." 
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
                    allnan = 0
    if allnan == 0:       
        return np.DTYPE(amax)
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
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        amax = MINDTYPE
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
        y[INDEXPOP] = amax
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            amax = MINDTYPE
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
            y[INDEXPOP] = amax
    return y
"""

ints['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(ints) 
ints_None['axisNone'] = True

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai >= amax:
            amax = ai
    return np.DTYPE(amax)
"""
loop[2] = """\
    if nINDEX0 * nINDEX1 == 0:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai >= amax:
                amax = ai
    return np.DTYPE(amax)
"""
loop[3] = """\
    if nINDEX0 * nINDEX1 * nINDEX2 == 0:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    amax = MINDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai >= amax:
                    amax = ai
    return np.DTYPE(amax)
"""

ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanmax"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanmax(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanmax = {}
nanmax['name'] = 'nanmax'
nanmax['is_reducing_function'] = True
nanmax['cdef_output'] = True
nanmax['slow'] = slow
nanmax['templates'] = {}
nanmax['templates']['float'] = floats
nanmax['templates']['float_None'] = floats_None
nanmax['templates']['int'] = ints
nanmax['templates']['int_None'] = ints_None
nanmax['pyx_file'] = 'func/nanmax.pyx'

nanmax['main'] = '''"nanmax auto-generated from template"

def nanmax(arr, axis=None):
    """
    Maximum values along specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the maximum is computed. The default (axis=None) is
        to compute the maximum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    See also
    --------
    bottleneck.nanmin: Minimum along specified axis, ignoring NaNs.
    bottleneck.nanargmax: Indices of maximum values along axis, ignoring NaNs. 
    
    Examples
    --------
    >>> bn.nanmax(1)
    1
    >>> bn.nanmax([1])
    1
    >>> bn.nanmax([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmax(a)
    4.0
    >>> bn.nanmax(a, axis=0)
    array([ 1.,  4.])
    
    """
    func, arr = nanmax_selector(arr, axis)
    return func(arr)

def nanmax_selector(arr, axis):
    """
    Return nanmax function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanmax()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use
    to calculate the maximum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the maximum is to be computed. The default
        (axis=None) is to compute the maximum of the flattened array.
    
    Returns
    -------
    func : function
        The nanamx function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to find
        the maximum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the maximum of `arr` along
    axis=0:

    >>> func, a = bn.func.nanmax_selector(arr, axis=0)
    >>> func
    <built-in function nanmax_1d_float64_axis0> 
    
    Use the returned function and array to determine the maximum:
    
    >>> func(a)
    3.0

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
        func = nanmax_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmax_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''
