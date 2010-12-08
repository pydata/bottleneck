"nanmin template"

__all__ = ["nanmin"]

# Float dtypes (not axis=None) ----------------------------------------------

nanmin_float = {}

nanmin_float['dtype'] = ['float64']
nanmin_float['ndims'] = [2, 3]
nanmin_float['axisNone'] = False
nanmin_float['name'] = 'nanmin'
nanmin_float['inarr'] = 'a'
nanmin_float['outarr'] = 'y'

nanmin_float['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "NAME of NDIMd numpy array with dtype=DTYPE along axis=AXIS."
    cdef int allnan
    cdef np.DTYPE_t amin, ai
"""

nanmin_float['init'] = """
amin = np.inf
allnan = 1
"""

nanmin_float['inner'] = """
ai = a[INDEX]
if ai <= amin:
    amin = ai
    allnan = 0
""" 

nanmin_float['result'] = """
if allnan == 0:       
    y[INDEX] = amin
else:
    y[INDEX] = NAN
"""

nanmin_float['returns'] = "return y"

# Float dtypes (axis=None) --------------------------------------------------

nanmin_float_None = {}

nanmin_float_None['dtype'] = ['float64']
nanmin_float_None['ndims'] = [1, 2, 3]
nanmin_float_None['axisNone'] = True
nanmin_float_None['name'] = nanmin_float['name']
nanmin_float_None['inarr'] = nanmin_float['inarr']
nanmin_float_None['outarr'] = nanmin_float['outarr']

nanmin_float_None['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "NAME of NDIMd numpy array with dtype=DTYPE along axis=AXIS."
    cdef int allnan
    cdef np.DTYPE_t amin = np.inf, ai
"""

nanmin_float_None['init'] = None

nanmin_float_None['inner'] = nanmin_float['inner']

nanmin_float_None['result'] = """
if allnan == 0:       
    return np.DTYPE(amin)
else:
    return NAN
"""

nanmin_float_None['returns'] = None

# Int dtypes ----------------------------------------------------------------

nanmin_int = {}

nanmin_int['dtype'] = ['int32', 'int64'] 
nanmin_int['ndims'] = [2, 3]
nanmin_int['axisNone'] = False
nanmin_int['name'] = nanmin_float['name']
nanmin_int['inarr'] = nanmin_float['inarr']
nanmin_int['outarr'] = nanmin_float['outarr']

nanmin_int['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "NAME of NDIMd numpy array with dtype=DTYPE along axis=AXIS."
    cdef np.DTYPE_t amin, ai
"""

nanmin_int['init'] = """
amin = MAXDTYPE
"""

nanmin_int['inner'] = """
ai = a[INDEX]
if ai <= amin:
    amin = ai
""" 

nanmin_int['result'] = """
y[INDEX] = amin
"""

nanmin_int['returns'] = "return y"

# Int dtypes (axis=None)-----------------------------------------------------

nanmin_int_None = {}

nanmin_int_None['dtype'] = ['int32', 'int64'] 
nanmin_int_None['ndims'] = [1, 2, 3]
nanmin_int_None['axisNone'] = True
nanmin_int_None['name'] = nanmin_float['name']
nanmin_int_None['inarr'] = nanmin_float['inarr']
nanmin_int_None['outarr'] = nanmin_float['outarr']

nanmin_int_None['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "NAME of NDIMd numpy array with dtype=DTYPE along axis=AXIS."
    cdef np.DTYPE_t amin = MAXDTYPE, ai
"""

nanmin_int_None['init'] = None

nanmin_int_None['inner'] = nanmin_int['inner']

nanmin_int_None['result'] = """
return np.DTYPE(amin)
"""

nanmin_int_None['returns'] = None

# ---------------------------------------------------------------------------

nanmin = {}
nanmin['templates'] = {}
nanmin['templates']['float'] = nanmin_float
nanmin['templates']['float_None'] = nanmin_float_None
nanmin['templates']['int'] = nanmin_int
nanmin['templates']['int'] = nanmin_int_None
nanmin['pyx_file'] = '../func/nanmin2.pyx'

nanmin['main'] = """
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
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a
"""    
