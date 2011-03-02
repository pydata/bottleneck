"nanrankdata template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nanrankdata"]

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
            if old == old:
                averank = sumranks / dupcount + 1
                for j in xrange(k - dupcount, k):
                    y[INDEXREPLACE|ivec[j]|] = averank
            else:
                y[INDEXREPLACE|ivec[iINDEX0]|] = NAN
            sumranks = 0
            dupcount = 0
        old = new    
    sumranks += (nINDEX0 - 1)
    dupcount += 1
    if old == old:
        averank = sumranks / dupcount + 1
        for j in xrange(nINDEX0 - dupcount, nINDEX0):
                y[INDEXREPLACE|ivec[j]|] = averank
    else:
        y[INDEXREPLACE|ivec[nINDEX0 - 1]|] = NAN
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
                if old == old:
                    averank = sumranks / dupcount + 1
                    for j in xrange(k - dupcount, k):
                        idx = ivec[INDEXREPLACE|j|]
                        y[INDEXREPLACE|idx|] = averank
                else:
                    idx = ivec[INDEXALL]
                    y[INDEXREPLACE|idx|] = NAN
                sumranks = 0
                dupcount = 0
            old = new    
        sumranks += (nINDEX1 - 1)
        dupcount += 1
        averank = sumranks / dupcount + 1
        if old == old:
            for j in xrange(nINDEX1 - dupcount, nINDEX1):
                idx = ivec[INDEXREPLACE|j|]
                y[INDEXREPLACE|idx|] = averank
        else:
            idx = ivec[INDEXREPLACE|nINDEX1 - 1|]
            y[INDEXREPLACE|idx|] = NAN
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
                    if old == old:
                        averank = sumranks / dupcount + 1
                        for j in xrange(k - dupcount, k):
                            idx = ivec[INDEXREPLACE|j|]
                            y[INDEXREPLACE|idx|] = averank
                    else:
                        idx = ivec[INDEXREPLACE|iINDEX2|]
                        y[INDEXREPLACE|idx|] = NAN
                    sumranks = 0
                    dupcount = 0
                old = new    
            sumranks += (nINDEX2 - 1)
            dupcount += 1
            averank = sumranks / dupcount + 1
            if old == old:
                for j in xrange(nINDEX2 - dupcount, nINDEX2):
                    idx = ivec[INDEXREPLACE|j|]
                    y[INDEXREPLACE|idx|] = averank
            else:
                idx = ivec[INDEXREPLACE|nINDEX2 - 1|]
                y[INDEXREPLACE|idx|] = NAN
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
    cdef dupcount = 0
    cdef Py_ssize_t j=0, k, idx
    cdef np.ndarray[np.NPINT_t, ndim=NDIM] ivec = PyArray_ArgSort(a, AXIS, NPY_QUICKSORT)
    cdef np.float64_t old, new, averank, sumranks = 0
"""

floats['loop'] = deepcopy(loop)

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES 
ints['reuse_non_nan_func'] = True
ints['loop'] = deepcopy(loop)

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nanrankdata"
slow['signature'] = "arr"
slow['func'] = "bn.slow.nanrankdata(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

nanrankdata = {}
nanrankdata['name'] = 'nanrankdata'
nanrankdata['is_reducing_function'] = False
nanrankdata['cdef_output'] = True
nanrankdata['slow'] = slow
nanrankdata['templates'] = {}
nanrankdata['templates']['float'] = floats
nanrankdata['templates']['int'] = ints
nanrankdata['pyx_file'] = 'func/nanrankdata.pyx'

nanrankdata['main'] = '''"nanrankdata auto-generated from template"

def nanrankdata(arr, axis=None):
    """
    Ranks the data, dealing with ties and NaNs appropriately.

    Equal values are assigned a rank that is the average of the ranks that
    would have been otherwise assigned to all of the values within that set.
    Ranks begin at 1, not 0.

    NaNs in the input array are returned as NaNs.

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
    bottleneck.rankdata: Ranks the data, dealing with ties and appropriately.
    
    Examples
    --------
    >>> bn.nanrankdata([np.nan, 2, 2, 3])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]])
    array([ nan,  1.5,  1.5,  3. ])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=0)
    array([[ nan,   1.],
           [  1.,   2.]])
    >>> bn.nanrankdata([[np.nan, 2], [2, 3]], axis=1)
    array([[ nan,   1.],
           [  1.,   2.]])
    
    """
    func, arr = nanrankdata_selector(arr, axis)
    return func(arr)

def nanrankdata_selector(arr, axis):
    """
    Return nanrankdata function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanrankdata() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to rank the elements.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which to rank the elements of the array. The default
        (axis=None) is to rank the elements of the flattened array.
    
    Returns
    -------
    func : function
        The nanrankdata function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to rank.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([np.nan, 2, 2, 3])
    
    Obtain the function needed to rank the elements of `arr` along axis=0:

    >>> func, a = bn.func.nanrankdata_selector(arr, axis=0)
    >>> func
    <built-in function nanrankdata_1d_float64_axis0> 
    
    Use the returned function and array:

    >>> func(a)
    array([ nan,  1.5,  1.5,  3. ])

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
        func = nanrankdata_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanrankdata_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
