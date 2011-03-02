"nanmedian template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanmedian"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loops ---------------------------------------------------------------------

loop = {}
loop[1] = """\
    if nINDEX0 == 0:
        return np.FLOAT(NAN)
    k = nAXIS 
    l = 0
    r = k - 1
    while l < r:
        i = l
        j = r
        while a[i] == a[i]:
            i += 1
            if i == nAXIS:
                break
        while a[j] != a[j]:
            j -= 1
        if i <= j:
            tmp = a[i]
            a[i] = a[j]
            a[j] = tmp
            i += 1
            j -= 1
        if i > j: break
        l = i
        r = j
    n = j + 1 
    k = n >> 1
    l = 0
    r = n - 1 
    with nogil:       
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n % 2 == 0:        
        amax = MINDTYPE
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.FLOAT(0.5 * (a[k] + amax))
    else:
        return np.FLOAT(a[k])
"""        
loop[2] = """\
    if nINDEX1 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for iINDEX0 in range(nINDEX0): 
        k = nAXIS 
        l = 0
        r = k - 1
        while l < r:
            i = l
            j = r
            while a[INDEXREPLACE|i|] == a[INDEXREPLACE|i|]:
                i += 1
                if i == nAXIS:
                    break
            while a[INDEXREPLACE|j|] != a[INDEXREPLACE|j|]:
                j -= 1
            if i <= j:
                tmp = a[INDEXREPLACE|i|]
                a[INDEXREPLACE|i|] = a[INDEXREPLACE|j|]
                a[INDEXREPLACE|j|] = tmp
                i += 1
                j -= 1
            if i > j: break
            l = i
            r = j
        n = j + 1 
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = a[INDEXREPLACE|k|]
            i = l
            j = r
            while 1:
                while a[INDEXREPLACE|i|] < x: i += 1
                while x < a[INDEXREPLACE|j|]: j -= 1
                if i <= j:
                    tmp = a[INDEXREPLACE|i|]
                    a[INDEXREPLACE|i|] = a[INDEXREPLACE|j|]
                    a[INDEXREPLACE|j|] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n % 2 == 0:        
            amax = MINDTYPE
            for i in range(k):
                ai = a[INDEXREPLACE|i|]
                if ai >= amax:
                    amax = ai
            y[INDEXPOP] = 0.5 * (a[INDEXREPLACE|k|] + amax)
        else:
            y[INDEXPOP] = CASTa[INDEXREPLACE|k|]         
    return y
"""
loop[3] = """\
    if nINDEX2 == 0:
        PyArray_FillWithScalar(y, NAN)
        return y
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            k = nAXIS 
            l = 0
            r = k - 1
            while l < r:
                i = l
                j = r
                while a[INDEXREPLACE|i|] == a[INDEXREPLACE|i|]:
                    i += 1
                    if i == nAXIS:
                        break
                while a[INDEXREPLACE|j|] != a[INDEXREPLACE|j|]:
                    j -= 1
                if i <= j:
                    tmp = a[INDEXREPLACE|i|]
                    a[INDEXREPLACE|i|] = a[INDEXREPLACE|j|]
                    a[INDEXREPLACE|j|] = tmp
                    i += 1
                    j -= 1
                if i > j: break
                l = i
                r = j
            n = j + 1 
            k = n >> 1
            l = 0
            r = n - 1
            while l < r:
                x = a[INDEXREPLACE|k|]
                i = l
                j = r
                while 1:
                    while a[INDEXREPLACE|i|] < x: i += 1
                    while x < a[INDEXREPLACE|j|]: j -= 1
                    if i <= j:
                        tmp = a[INDEXREPLACE|i|]
                        a[INDEXREPLACE|i|] = a[INDEXREPLACE|j|]
                        a[INDEXREPLACE|j|] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n % 2 == 0:        
                amax = MINDTYPE
                for i in range(k):
                    ai = a[INDEXREPLACE|i|]
                    if ai >= amax:
                        amax = ai
                y[INDEXPOP] = 0.5 * (a[INDEXREPLACE|k|] + amax)
            else:
                y[INDEXPOP] = CASTa[INDEXREPLACE|k|]         
    return y
"""

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
    "Median of NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef np.npy_intp i, j = 0, l, r, k, n 
    cdef np.DTYPE_t x, tmp, amax, ai
"""

floats['loop'] = {}
floats['loop'][1] = loop[1].replace('FLOAT', 'DTYPE')
floats['loop'][2] = loop[2].replace('CAST', '')
floats['loop'][3] = loop[3].replace('CAST', '')

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 
ints['reuse_non_nan_func'] = True

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanmedian"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanmedian(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanmedian = {}
nanmedian['name'] = 'nanmedian'
nanmedian['is_reducing_function'] = True
nanmedian['cdef_output'] = True
nanmedian['slow'] = slow
nanmedian['templates'] = {}
nanmedian['templates']['float'] = floats
nanmedian['templates']['int'] = ints
nanmedian['pyx_file'] = 'func/nanmedian.pyx'

nanmedian['main'] = '''"nanmedian auto-generated from template"
# Select smallest k elements code used for inner loop of median method:
# http://projects.scipy.org/numpy/attachment/ticket/1213/quickselect.pyx
# (C) 2009 Sturla Molden
# SciPy license 
#
# From the original C function (code in public domain) in:
#   Fast median search: an ANSI C implementation
#   Nicolas Devillard - ndevilla AT free DOT fr
#   July 1998
# which, in turn, took the algorithm from
#   Wirth, Niklaus
#   Algorithms + data structures = programs, p. 366
#   Englewood Cliffs: Prentice-Hall, 1976
#
# Adapted and expanded for Bottleneck:
# (C) 2010 Keith Goodman

def nanmedian(arr, axis=None):
    """
    Median of array elements along given axis ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is computed. The default (axis=None) is to
        compute the median of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, except that the specified axis
        has been removed. If `arr` is a 0d array, or if axis is None, a scalar
        is returned. `float64` return values are used for integer inputs. 
    
    See also
    --------
    bottleneck.median: Median along specified axis. 

    Examples
    --------
    >>> a = np.array([[np.nan, 7, 4], [3, 2, 1]])
    >>> a 
    array([[ nan,   7.,   4.],
           [  3.,   2.,   1.]])
    >>> bn.nanmedian(a)
    3.0
    >> bn.nanmedian(a, axis=0)
    array([ 3. ,  4.5,  2.5])
    >> bn.nanmedian(a, axis=1)
    array([ 5.5,  2. ])
    
    """
    func, arr = nanmedian_selector(arr, axis)
    return func(arr)

def nanmedian_selector(arr, axis):
    """
    Return nanmedian function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanmedian() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the mean.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is to be computed. The default (axis=None)
        is to compute the mean of the flattened array.
    
    Returns
    -------
    func : function
        The nanmedian function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the
        median.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the median of `arr` along axis=0:

    >>> func, a = bn.func.nanmedian_selector(arr, axis=0)
    >>> func
    <built-in function nanmedian_1d_float64_axis0> 
    
    Use the returned function and array to determine the median:

    >>> func(a)
    2.0

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
        func = nanmedian_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmedian_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
