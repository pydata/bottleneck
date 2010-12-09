"nanmin template"

from copy import deepcopy

__all__ = ["nanmin"]

FLOAT_DTYPES = ['float64']
INT_DTYPES = ['int32', 'int64']

# Float dtypes (not axis=None) ----------------------------------------------

nanmin_float = {}
nanmin_float['dtypes'] = FLOAT_DTYPES
nanmin_float['axisNone'] = False
nanmin_float['force_output_dtype'] = False

nanmin_float['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "NAME of NDIMd numpy array with dtype=DTYPE along axis=AXIS."
    cdef int allnan = 1
    cdef np.DTYPE_t amin, ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        amin = np.inf
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
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            amin = np.inf
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

nanmin_float['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

nanmin_float_None = deepcopy(nanmin_float)
nanmin_float_None['axisNone'] = True

loop = {}
loop[1] = """\
    amin = np.inf
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
    amin = np.inf
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
    amin = np.inf
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

nanmin_float_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

nanmin_int = deepcopy(nanmin_float)
nanmin_int['dtypes'] = INT_DTYPES 

loop = {}
loop[2] = """\
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

nanmin_int['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

nanmin_int_None = deepcopy(nanmin_int) 
nanmin_int_None['axisNone'] = True

loop = {}
loop[1] = """\
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai <= amin:
            amin = ai
    return np.DTYPE(amin)
"""
loop[2] = """\
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai <= amin:
                amin = ai
    return np.DTYPE(amin)
"""
loop[3] = """\
    amin = MAXDTYPE
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai <= amin:
                    amin = ai
    return np.DTYPE(amin)
"""

nanmin_int_None['loop'] = loop

# Template ------------------------------------------------------------------

nanmin = {}
nanmin['name'] = 'nanmin'
nanmin['templates'] = {}
nanmin['templates']['float'] = nanmin_float
nanmin['templates']['float_None'] = nanmin_float_None
nanmin['templates']['int'] = nanmin_int
nanmin['templates']['int_None'] = nanmin_int_None
nanmin['pyx_file'] = '../func/nanmin.pyx'

nanmin['main'] = """"nanmin auto-generated from template"

def nanmin(arr, axis=None):
    '''
    Minimum along the specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is computed. The default is to compute
        the minimum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    
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
    
    '''
    func, arr = nanmin_selector(arr, axis)
    return func(arr)

def nanmin_selector(arr, axis):
    '''
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

    '''
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    cdef int size = a.size
    if size == 0:
        msg = "numpy.nanmin() raises on size=0 input; so Bottleneck does too." 
        raise ValueError, msg
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmin_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype), str(axis))
        raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
"""    
