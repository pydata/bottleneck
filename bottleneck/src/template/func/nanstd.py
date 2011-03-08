"nanstd template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanstd"]

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
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a, int ddof):
    "Variance of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int count = 0
    cdef np.DTYPE_t asum = 0, amean, ai
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
            amean = asum / count
            asum = 0
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[INDEXPOP] = sqrt(asum / (count - ddof))
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
                amean = asum / count
                asum = 0
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[INDEXPOP] = sqrt(asum / (count - ddof))
            else:
                y[INDEXPOP] = NAN
    return y  
"""
floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

returns = """\
        return np.DTYPE(sqrt(asum / (count - ddof)))
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
    if count > 0:
        amean = asum / count
        asum = 0
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai == ai:
                ai -= amean
                asum += (ai * ai)
""" + returns
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
""" + returns
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
""" + returns
floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 
ints['force_output_dtype'] = 'float64'

ints['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a, int ddof):
    "Valriance of NDIMd array with dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / nINDEX1
            asum = 0
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                ai -= amean
                asum += (ai * ai)
            y[INDEXPOP] = sqrt(asum / (nINDEX1 - ddof))
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
                amean = asum / nINDEX2
                asum = 0
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    ai -= amean
                    asum += (ai * ai)
                y[INDEXPOP] = sqrt(asum / (nINDEX2 - ddof))
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
    if size == 0:
        return np.float64(NAN)
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
    amean = asum / size
    asum = 0
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            ai -= amean
            asum += (ai * ai)
    if size > ddof:        
        return np.float64(sqrt(asum / (size - ddof)))
    else:
        return np.float64(NAN)
"""
loop[2] = """\
    size = nINDEX0 * nINDEX1
    if size == 0:
        return np.float64(NAN)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum += a[INDEXALL]
    amean = asum / size
    asum = 0
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            ai -= amean
            asum += (ai * ai)
    if size > ddof:        
        return np.float64(sqrt(asum / (size - ddof)))
    else:
        return np.float64(NAN)
"""
loop[3] = """\
    size = nINDEX0 * nINDEX1 * nINDEX2
    if size == 0:
        return np.float64(NAN)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                asum += a[INDEXALL]
    amean = asum / size
    asum = 0
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                ai -= amean
                asum += (ai * ai)
    if size > ddof:        
        return np.float64(sqrt(asum / (size - ddof)))
    else:
        return np.float64(NAN)
"""
ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanstd"
slow['signature'] = "arr, ddof"
slow['func'] = "bn.slow.nanstd(arr, axis=AXIS, ddof=ddof)"

# Template ------------------------------------------------------------------

nanstd = {}
nanstd['name'] = 'nanstd'
nanstd['is_reducing_function'] = True
nanstd['cdef_output'] = True
nanstd['slow'] = slow
nanstd['templates'] = {}
nanstd['templates']['float'] = floats
nanstd['templates']['float_None'] = floats_None
nanstd['templates']['int'] = ints
nanstd['templates']['int_None'] = ints_None
nanstd['pyx_file'] = 'func/nanstd.pyx'

nanstd['main'] = '''"nanstd auto-generated from template"

def nanstd(arr, axis=None, int ddof=0):
    """
    Standard deviation along the specified axis, ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Instead of a faster one-pass algorithm, a more stable two-pass algorithm
    is used.

    An example of a one-pass algorithm:

        >>> np.sqrt((arr*arr).mean() - arr.mean()**2)
    
    An example of a two-pass algorithm:    
    
        >>> np.sqrt(((arr - arr.mean())**2).mean())

    Note in the two-pass algorithm the mean must be found (first pass) before
    the squared deviation (second pass) can be found.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the standard deviation is computed. The default
        (axis=None) is to compute the standard deviation of the flattened
        array.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs. 

    See also
    --------
    bottleneck.nanvar: Variance along specified axis ignoring NaNs

    Notes
    -----
    If positive or negative infinity are present the result is Not A Number
    (NaN).

    Examples
    --------
    >>> bn.nanstd(1)
    0.0
    >>> bn.nanstd([1])
    0.0
    >>> bn.nanstd([1, np.nan])
    0.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanstd(a)
    1.4142135623730951
    >>> bn.nanstd(a, axis=0)
    array([ 0.,  0.])

    When positive infinity or negative infinity are present NaN is returned:

    >>> bn.nanstd([1, np.nan, np.inf])
    nan
    
    """
    func, arr = nanstd_selector(arr, axis)
    return func(arr, ddof)

def nanstd_selector(arr, axis):
    """
    Return std function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanstd()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use
    to calculate the standard deviation.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the standard deviation is to be computed. The
        default (axis=None) is to compute the standard deviation of the
        flattened array.
    
    Returns
    -------
    func : function
        The standard deviation function that matches the number of dimensions
        and dtype of the input array and the axis along which you wish to
        find the standard deviation.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the standard deviation of `arr`
    along axis=0:

    >>> func, a = bn.func.nanstd_selector(arr, axis=0)
    >>> func
    <built-in function nanstd_1d_float64_axis0> 
    
    Use the returned function and array to determine the standard deviation:
    
    >>> func(a)
    0.81649658092772603

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
        func = nanstd_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanstd_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
