"nansum auto-generated from template"

def nansum(arr, axis=None):
    """
    Sum of array elements along given axis ignoring NaNs.

    When the input has an integer type with less precision than the default
    platform integer, the default platform integer is used for the
    accumulator and return values.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose sum is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is computed. The default (axis=None) is to
        compute the sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned. 
    
    Notes
    -----
    No error is raised on overflow.

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> bn.nansum(1)
    1
    >>> bn.nansum([1])
    1
    >>> bn.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> bn.nansum(a)
    3.0
    >>> bn.nansum(a, axis=0)
    array([ 2.,  1.])

    When positive infinity and negative infinity are present:

    >>> bn.nansum([1, np.nan, np.inf])
    inf
    >>> bn.nansum([1, np.nan, np.NINF])
    -inf
    >>> bn.nansum([1, np.nan, np.inf, np.NINF])
    nan
    
    """
    func, arr = nansum_selector(arr, axis)
    return func(arr)

def nansum_selector(arr, axis):
    """
    Return nansum function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nansum() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is to be computed. The default (axis=None)
        is to compute the sum of the flattened array.
    
    Returns
    -------
    func : function
        The nansum function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the sum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, np.nan, 3.0])
    
    Obtain the function needed to determine the nansum of `arr` along axis=0:

    >>> func, a = bn.func.nansum_selector(arr, axis=0)
    >>> func
    <built-in function nansum_1d_float64_axis0> 
    
    Use the returned function and array to determine the sum:

    >>> func(a)
    4.0

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if dtype < NPY_int_:
        a = a.astype(np.int_)
        dtype = PyArray_TYPE(a)
    if (axis < 0) and (axis is not None):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nansum_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nansum_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        asum = 0
        for i0 in range(n0):
            asum += a[i0, i1]
        y[i1] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        asum = 0
        for i1 in range(n1):
            asum += a[i0, i1]
        y[i0] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        asum = 0
        for i0 in range(n0):
            asum += a[i0, i1]
        y[i1] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        asum = 0
        for i1 in range(n1):
            asum += a[i0, i1]
        y[i0] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            for i0 in range(n0):
                asum += a[i0, i1, i2]
            y[i1, i2] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            for i1 in range(n1):
                asum += a[i0, i1, i2]
            y[i0, i2] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            for i2 in range(n2):
                asum += a[i0, i1, i2]
            y[i0, i1] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            for i0 in range(n0):
                asum += a[i0, i1, i2]
            y[i1, i2] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            for i1 in range(n1):
                asum += a[i0, i1, i2]
            y[i0, i2] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            for i2 in range(n2):
                asum += a[i0, i1, i2]
            y[i0, i1] = asum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Mean of 1d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            allnan = 0
    if allnan == 0:
        return np.float32(asum)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Mean of 1d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            allnan = 0
    if allnan == 0:
        return np.float64(asum)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
    if allnan == 0:
        return np.float32(asum)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
    if allnan == 0:
        return np.float64(asum)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
    if allnan == 0:
        return np.float32(asum)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
    if allnan == 0:
        return np.float64(asum)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        asum = 0
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
        if allnan == 0:       
            y[i1] = asum
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        asum = 0
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
        if allnan == 0:       
            y[i0] = asum
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        asum = 0
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
        if allnan == 0:       
            y[i1] = asum
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        asum = 0
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                allnan = 0
        if allnan == 0:       
            y[i0] = asum
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i1, i2] = asum
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i0, i2] = asum
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i0, i1] = asum
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i1, i2] = asum
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i0, i2] = asum
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i0, i1] = asum
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Mean of 1d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    size = n0
    for i0 in range(n0):
        asum += a[i0]
    return np.int32(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Mean of 1d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    size = n0
    for i0 in range(n0):
        asum += a[i0]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    size = n0 * n1    
    for i0 in range(n0):
        for i1 in range(n1):
            asum += a[i0, i1]
    return np.int32(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    size = n0 * n1    
    for i0 in range(n0):
        for i1 in range(n1):
            asum += a[i0, i1]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.int32_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    size = n0 * n1 * n2 
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                asum += a[i0, i1, i2]
    return np.int32(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.int64_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    size = n0 * n1 * n2 
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                asum += a[i0, i1, i2]
    return np.int64(asum)

cdef dict nansum_dict = {}
nansum_dict[(2, NPY_int32, 0)] = nansum_2d_int32_axis0
nansum_dict[(2, NPY_int32, 1)] = nansum_2d_int32_axis1
nansum_dict[(2, NPY_int64, 0)] = nansum_2d_int64_axis0
nansum_dict[(2, NPY_int64, 1)] = nansum_2d_int64_axis1
nansum_dict[(3, NPY_int32, 0)] = nansum_3d_int32_axis0
nansum_dict[(3, NPY_int32, 1)] = nansum_3d_int32_axis1
nansum_dict[(3, NPY_int32, 2)] = nansum_3d_int32_axis2
nansum_dict[(3, NPY_int64, 0)] = nansum_3d_int64_axis0
nansum_dict[(3, NPY_int64, 1)] = nansum_3d_int64_axis1
nansum_dict[(3, NPY_int64, 2)] = nansum_3d_int64_axis2
nansum_dict[(1, NPY_float32, 0)] = nansum_1d_float32_axisNone
nansum_dict[(1, NPY_float32, None)] = nansum_1d_float32_axisNone
nansum_dict[(1, NPY_float64, 0)] = nansum_1d_float64_axisNone
nansum_dict[(1, NPY_float64, None)] = nansum_1d_float64_axisNone
nansum_dict[(2, NPY_float32, None)] = nansum_2d_float32_axisNone
nansum_dict[(2, NPY_float64, None)] = nansum_2d_float64_axisNone
nansum_dict[(3, NPY_float32, None)] = nansum_3d_float32_axisNone
nansum_dict[(3, NPY_float64, None)] = nansum_3d_float64_axisNone
nansum_dict[(2, NPY_float32, 0)] = nansum_2d_float32_axis0
nansum_dict[(2, NPY_float32, 1)] = nansum_2d_float32_axis1
nansum_dict[(2, NPY_float64, 0)] = nansum_2d_float64_axis0
nansum_dict[(2, NPY_float64, 1)] = nansum_2d_float64_axis1
nansum_dict[(3, NPY_float32, 0)] = nansum_3d_float32_axis0
nansum_dict[(3, NPY_float32, 1)] = nansum_3d_float32_axis1
nansum_dict[(3, NPY_float32, 2)] = nansum_3d_float32_axis2
nansum_dict[(3, NPY_float64, 0)] = nansum_3d_float64_axis0
nansum_dict[(3, NPY_float64, 1)] = nansum_3d_float64_axis1
nansum_dict[(3, NPY_float64, 2)] = nansum_3d_float64_axis2
nansum_dict[(1, NPY_int32, 0)] = nansum_1d_int32_axisNone
nansum_dict[(1, NPY_int32, None)] = nansum_1d_int32_axisNone
nansum_dict[(1, NPY_int64, 0)] = nansum_1d_int64_axisNone
nansum_dict[(1, NPY_int64, None)] = nansum_1d_int64_axisNone
nansum_dict[(2, NPY_int32, None)] = nansum_2d_int32_axisNone
nansum_dict[(2, NPY_int64, None)] = nansum_2d_int64_axisNone
nansum_dict[(3, NPY_int32, None)] = nansum_3d_int32_axisNone
nansum_dict[(3, NPY_int64, None)] = nansum_3d_int64_axisNone

cdef dict nansum_slow_dict = {}
nansum_slow_dict[0] = nansum_slow_axis0
nansum_slow_dict[1] = nansum_slow_axis1
nansum_slow_dict[2] = nansum_slow_axis2
nansum_slow_dict[3] = nansum_slow_axis3
nansum_slow_dict[4] = nansum_slow_axis4
nansum_slow_dict[5] = nansum_slow_axis5
nansum_slow_dict[6] = nansum_slow_axis6
nansum_slow_dict[7] = nansum_slow_axis7
nansum_slow_dict[8] = nansum_slow_axis8
nansum_slow_dict[9] = nansum_slow_axis9
nansum_slow_dict[10] = nansum_slow_axis10
nansum_slow_dict[11] = nansum_slow_axis11
nansum_slow_dict[12] = nansum_slow_axis12
nansum_slow_dict[13] = nansum_slow_axis13
nansum_slow_dict[14] = nansum_slow_axis14
nansum_slow_dict[15] = nansum_slow_axis15
nansum_slow_dict[16] = nansum_slow_axis16
nansum_slow_dict[17] = nansum_slow_axis17
nansum_slow_dict[18] = nansum_slow_axis18
nansum_slow_dict[19] = nansum_slow_axis19
nansum_slow_dict[20] = nansum_slow_axis20
nansum_slow_dict[21] = nansum_slow_axis21
nansum_slow_dict[22] = nansum_slow_axis22
nansum_slow_dict[23] = nansum_slow_axis23
nansum_slow_dict[24] = nansum_slow_axis24
nansum_slow_dict[25] = nansum_slow_axis25
nansum_slow_dict[26] = nansum_slow_axis26
nansum_slow_dict[27] = nansum_slow_axis27
nansum_slow_dict[28] = nansum_slow_axis28
nansum_slow_dict[29] = nansum_slow_axis29
nansum_slow_dict[30] = nansum_slow_axis30
nansum_slow_dict[31] = nansum_slow_axis31
nansum_slow_dict[32] = nansum_slow_axis32
nansum_slow_dict[None] = nansum_slow_axisNone

def nansum_slow_axis0(arr):
    "Unaccelerated (slow) nansum along axis 0."
    return bn.slow.nansum(arr, axis=0)

def nansum_slow_axis1(arr):
    "Unaccelerated (slow) nansum along axis 1."
    return bn.slow.nansum(arr, axis=1)

def nansum_slow_axis2(arr):
    "Unaccelerated (slow) nansum along axis 2."
    return bn.slow.nansum(arr, axis=2)

def nansum_slow_axis3(arr):
    "Unaccelerated (slow) nansum along axis 3."
    return bn.slow.nansum(arr, axis=3)

def nansum_slow_axis4(arr):
    "Unaccelerated (slow) nansum along axis 4."
    return bn.slow.nansum(arr, axis=4)

def nansum_slow_axis5(arr):
    "Unaccelerated (slow) nansum along axis 5."
    return bn.slow.nansum(arr, axis=5)

def nansum_slow_axis6(arr):
    "Unaccelerated (slow) nansum along axis 6."
    return bn.slow.nansum(arr, axis=6)

def nansum_slow_axis7(arr):
    "Unaccelerated (slow) nansum along axis 7."
    return bn.slow.nansum(arr, axis=7)

def nansum_slow_axis8(arr):
    "Unaccelerated (slow) nansum along axis 8."
    return bn.slow.nansum(arr, axis=8)

def nansum_slow_axis9(arr):
    "Unaccelerated (slow) nansum along axis 9."
    return bn.slow.nansum(arr, axis=9)

def nansum_slow_axis10(arr):
    "Unaccelerated (slow) nansum along axis 10."
    return bn.slow.nansum(arr, axis=10)

def nansum_slow_axis11(arr):
    "Unaccelerated (slow) nansum along axis 11."
    return bn.slow.nansum(arr, axis=11)

def nansum_slow_axis12(arr):
    "Unaccelerated (slow) nansum along axis 12."
    return bn.slow.nansum(arr, axis=12)

def nansum_slow_axis13(arr):
    "Unaccelerated (slow) nansum along axis 13."
    return bn.slow.nansum(arr, axis=13)

def nansum_slow_axis14(arr):
    "Unaccelerated (slow) nansum along axis 14."
    return bn.slow.nansum(arr, axis=14)

def nansum_slow_axis15(arr):
    "Unaccelerated (slow) nansum along axis 15."
    return bn.slow.nansum(arr, axis=15)

def nansum_slow_axis16(arr):
    "Unaccelerated (slow) nansum along axis 16."
    return bn.slow.nansum(arr, axis=16)

def nansum_slow_axis17(arr):
    "Unaccelerated (slow) nansum along axis 17."
    return bn.slow.nansum(arr, axis=17)

def nansum_slow_axis18(arr):
    "Unaccelerated (slow) nansum along axis 18."
    return bn.slow.nansum(arr, axis=18)

def nansum_slow_axis19(arr):
    "Unaccelerated (slow) nansum along axis 19."
    return bn.slow.nansum(arr, axis=19)

def nansum_slow_axis20(arr):
    "Unaccelerated (slow) nansum along axis 20."
    return bn.slow.nansum(arr, axis=20)

def nansum_slow_axis21(arr):
    "Unaccelerated (slow) nansum along axis 21."
    return bn.slow.nansum(arr, axis=21)

def nansum_slow_axis22(arr):
    "Unaccelerated (slow) nansum along axis 22."
    return bn.slow.nansum(arr, axis=22)

def nansum_slow_axis23(arr):
    "Unaccelerated (slow) nansum along axis 23."
    return bn.slow.nansum(arr, axis=23)

def nansum_slow_axis24(arr):
    "Unaccelerated (slow) nansum along axis 24."
    return bn.slow.nansum(arr, axis=24)

def nansum_slow_axis25(arr):
    "Unaccelerated (slow) nansum along axis 25."
    return bn.slow.nansum(arr, axis=25)

def nansum_slow_axis26(arr):
    "Unaccelerated (slow) nansum along axis 26."
    return bn.slow.nansum(arr, axis=26)

def nansum_slow_axis27(arr):
    "Unaccelerated (slow) nansum along axis 27."
    return bn.slow.nansum(arr, axis=27)

def nansum_slow_axis28(arr):
    "Unaccelerated (slow) nansum along axis 28."
    return bn.slow.nansum(arr, axis=28)

def nansum_slow_axis29(arr):
    "Unaccelerated (slow) nansum along axis 29."
    return bn.slow.nansum(arr, axis=29)

def nansum_slow_axis30(arr):
    "Unaccelerated (slow) nansum along axis 30."
    return bn.slow.nansum(arr, axis=30)

def nansum_slow_axis31(arr):
    "Unaccelerated (slow) nansum along axis 31."
    return bn.slow.nansum(arr, axis=31)

def nansum_slow_axis32(arr):
    "Unaccelerated (slow) nansum along axis 32."
    return bn.slow.nansum(arr, axis=32)

def nansum_slow_axisNone(arr):
    "Unaccelerated (slow) nansum along axis None."
    return bn.slow.nansum(arr, axis=None)
