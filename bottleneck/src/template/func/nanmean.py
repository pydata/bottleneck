"nanmean template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanmean"]

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
    "Mean of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int count = 0
    cdef np.DTYPE_t asum = 0, ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        asum = 0
        count = 0
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:       
            y[INDEXPOP] = asum / count
        else:
            y[INDEXPOP] = NAN
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            count = 0
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[INDEXPOP] = asum / count
            else:
                y[INDEXPOP] = NAN
    return y
"""
floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

returns = """\
    if count > 0:
        return np.DTYPE(asum / count)
    else:
        return np.DTYPE(NAN)
"""        

loop = {}
loop[1] = """\
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            count += 1
""" + returns
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
""" + returns
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    count += 1
""" + returns
floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 
ints['force_output_dtype'] = 'float64'

ints['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Mean of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef np.float64_t asum = 0, ai
"""

loop = {}
loop[2] = """\
    if nINDEX1 == 0:
        PyArray_FillWithScalar(y, NAN)
    else:
        for iINDEX0 in range(nINDEX0):
            asum = 0
            for iINDEX1 in range(nINDEX1):
                asum += a[INDEXALL]
            y[INDEXPOP] = asum / nINDEX1
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        PyArray_FillWithScalar(y, NAN)
    else:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                asum = 0
                for iINDEX2 in range(nINDEX2):
                    asum += a[INDEXALL]
                y[INDEXPOP] = asum / nINDEX2
    return y
"""
ints['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(ints) 
ints_None['top'] = ints['top'] + "    cdef int size\n"
ints_None['axisNone'] = True

loop = {}
loop[1] = """\
    size = nINDEX0
    for iINDEX0 in range(nINDEX0):
        asum += a[INDEXALL]
    if size > 0:    
        return np.float64(asum / size)
    else:
        return np.float64(NAN)
"""
loop[2] = """\
    size = nINDEX0 * nINDEX1    
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum += a[INDEXALL]
    if size > 0:    
        return np.float64(asum / size)
    else:
        return np.float64(NAN)
"""
loop[3] = """\
    size = nINDEX0 * nINDEX1 * nINDEX2 
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                asum += a[INDEXALL]
    if size > 0:    
        return np.float64(asum / size)
    else:
        return np.float64(NAN)
"""
ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanmean"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanmean(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanmean = {}
nanmean['name'] = 'nanmean'
nanmean['is_reducing_function'] = True
nanmean['cdef_output'] = True
nanmean['slow'] = slow
nanmean['templates'] = {}
nanmean['templates']['float'] = floats
nanmean['templates']['float_None'] = floats_None
nanmean['templates']['int'] = ints
nanmean['templates']['int_None'] = ints_None
nanmean['pyx_file'] = 'func/nanmean.pyx'

nanmean['main'] = '''"nanmean auto-generated from template"

def nanmean(arr, axis=None):
    """
    Mean of array elements along given axis ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose mean is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the mean is computed. The default (axis=None) is to
        compute the mean of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs. 

    See also
    --------
    bottleneck.nanmedian: Median along specified axis, ignoring NaNs.
    
    Notes
    -----
    No error is raised on overflow. (The sum is computed and then the result
    is divided by the number of non-NaN elements.)

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> bn.nanmean(1)
    1.0
    >>> bn.nanmean([1])
    1.0
    >>> bn.nanmean([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmean(a)
    2.0
    >>> bn.nanmean(a, axis=0)
    array([ 1.,  4.])

    When positive infinity and negative infinity are present:

    >>> bn.nanmean([1, np.nan, np.inf])
    inf
    >>> bn.nanmean([1, np.nan, np.NINF])
    -inf
    >>> bn.nanmean([1, np.nan, np.inf, np.NINF])
    nan
    
    """
    func, arr = nanmean_selector(arr, axis)
    return func(arr)

def nanmean_selector(arr, axis):
    """
    Return nanmean function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanmean() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the mean.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the mean is to be computed. The default (axis=None)
        is to compute the mean of the flattened array.
    
    Returns
    -------
    func : function
        The nanmean function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the mean.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the nanmean of `arr` along axis=0:

    >>> func, a = bn.func.nanmean_selector(arr, axis=0)
    >>> func
    <built-in function nanmean_1d_float64_axis0> 
    
    Use the returned function and array to determine the mean:

    >>> func(a)
    2.0

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
        func = nanmean_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmean_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
