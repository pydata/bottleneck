"nanmean auto-generated from template"

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

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
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
        for i0 in range(n0):
            asum += a[i0, i1]
        y[i1] = asum / n0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
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
        for i1 in range(n1):
            asum += a[i0, i1]
        y[i0] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
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
        for i0 in range(n0):
            asum += a[i0, i1]
        y[i1] = asum / n0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
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
        for i1 in range(n1):
            asum += a[i0, i1]
        y[i0] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
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
            for i0 in range(n0):
                asum += a[i0, i1, i2]
            y[i1, i2] = asum / n0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
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
            for i1 in range(n1):
                asum += a[i0, i1, i2]
            y[i0, i2] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
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
            for i2 in range(n2):
                asum += a[i0, i1, i2]
            y[i0, i1] = asum / n2
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
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
            for i0 in range(n0):
                asum += a[i0, i1, i2]
            y[i1, i2] = asum / n0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
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
            for i1 in range(n1):
                asum += a[i0, i1, i2]
            y[i0, i2] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
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
            for i2 in range(n2):
                asum += a[i0, i1, i2]
            y[i0, i1] = asum / n2
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Mean of 1d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        return np.float32(asum / count)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Mean of 1d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
                count += 1
    if count > 0:
        return np.float32(asum / count)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
                count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
                    count += 1
    if count > 0:
        return np.float32(asum / count)
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
                    count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
        count = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:       
            y[i1] = asum / count
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Mean of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
        count = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:       
            y[i0] = asum / count
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
        count = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:       
            y[i1] = asum / count
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
        count = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:       
            y[i0] = asum / count
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i1, i2] = asum / count
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i0, i2] = asum / count
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Mean of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i0, i1] = asum / count
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i1, i2] = asum / count
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i0, i2] = asum / count
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
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
            count = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:       
                y[i0, i1] = asum / count
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Mean of 1d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    size = n0
    for i0 in range(n0):
        asum += a[i0]
    return np.float64(asum / size)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Mean of 1d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    size = n0
    for i0 in range(n0):
        asum += a[i0]
    return np.float64(asum / size)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
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
    return np.float64(asum / size)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Mean of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
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
    return np.float64(asum / size)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "Mean of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
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
    return np.float64(asum / size)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "Mean of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
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
    return np.float64(asum / size)

cdef dict nanmean_dict = {}
nanmean_dict[(2, NPY_int32, 0)] = nanmean_2d_int32_axis0
nanmean_dict[(2, NPY_int32, 1)] = nanmean_2d_int32_axis1
nanmean_dict[(2, NPY_int64, 0)] = nanmean_2d_int64_axis0
nanmean_dict[(2, NPY_int64, 1)] = nanmean_2d_int64_axis1
nanmean_dict[(3, NPY_int32, 0)] = nanmean_3d_int32_axis0
nanmean_dict[(3, NPY_int32, 1)] = nanmean_3d_int32_axis1
nanmean_dict[(3, NPY_int32, 2)] = nanmean_3d_int32_axis2
nanmean_dict[(3, NPY_int64, 0)] = nanmean_3d_int64_axis0
nanmean_dict[(3, NPY_int64, 1)] = nanmean_3d_int64_axis1
nanmean_dict[(3, NPY_int64, 2)] = nanmean_3d_int64_axis2
nanmean_dict[(1, NPY_float32, 0)] = nanmean_1d_float32_axisNone
nanmean_dict[(1, NPY_float32, None)] = nanmean_1d_float32_axisNone
nanmean_dict[(1, NPY_float64, 0)] = nanmean_1d_float64_axisNone
nanmean_dict[(1, NPY_float64, None)] = nanmean_1d_float64_axisNone
nanmean_dict[(2, NPY_float32, None)] = nanmean_2d_float32_axisNone
nanmean_dict[(2, NPY_float64, None)] = nanmean_2d_float64_axisNone
nanmean_dict[(3, NPY_float32, None)] = nanmean_3d_float32_axisNone
nanmean_dict[(3, NPY_float64, None)] = nanmean_3d_float64_axisNone
nanmean_dict[(2, NPY_float32, 0)] = nanmean_2d_float32_axis0
nanmean_dict[(2, NPY_float32, 1)] = nanmean_2d_float32_axis1
nanmean_dict[(2, NPY_float64, 0)] = nanmean_2d_float64_axis0
nanmean_dict[(2, NPY_float64, 1)] = nanmean_2d_float64_axis1
nanmean_dict[(3, NPY_float32, 0)] = nanmean_3d_float32_axis0
nanmean_dict[(3, NPY_float32, 1)] = nanmean_3d_float32_axis1
nanmean_dict[(3, NPY_float32, 2)] = nanmean_3d_float32_axis2
nanmean_dict[(3, NPY_float64, 0)] = nanmean_3d_float64_axis0
nanmean_dict[(3, NPY_float64, 1)] = nanmean_3d_float64_axis1
nanmean_dict[(3, NPY_float64, 2)] = nanmean_3d_float64_axis2
nanmean_dict[(1, NPY_int32, 0)] = nanmean_1d_int32_axisNone
nanmean_dict[(1, NPY_int32, None)] = nanmean_1d_int32_axisNone
nanmean_dict[(1, NPY_int64, 0)] = nanmean_1d_int64_axisNone
nanmean_dict[(1, NPY_int64, None)] = nanmean_1d_int64_axisNone
nanmean_dict[(2, NPY_int32, None)] = nanmean_2d_int32_axisNone
nanmean_dict[(2, NPY_int64, None)] = nanmean_2d_int64_axisNone
nanmean_dict[(3, NPY_int32, None)] = nanmean_3d_int32_axisNone
nanmean_dict[(3, NPY_int64, None)] = nanmean_3d_int64_axisNone

cdef dict nanmean_slow_dict = {}
nanmean_slow_dict[0] = nanmean_slow_axis0
nanmean_slow_dict[1] = nanmean_slow_axis1
nanmean_slow_dict[2] = nanmean_slow_axis2
nanmean_slow_dict[3] = nanmean_slow_axis3
nanmean_slow_dict[4] = nanmean_slow_axis4
nanmean_slow_dict[5] = nanmean_slow_axis5
nanmean_slow_dict[6] = nanmean_slow_axis6
nanmean_slow_dict[7] = nanmean_slow_axis7
nanmean_slow_dict[8] = nanmean_slow_axis8
nanmean_slow_dict[9] = nanmean_slow_axis9
nanmean_slow_dict[10] = nanmean_slow_axis10
nanmean_slow_dict[11] = nanmean_slow_axis11
nanmean_slow_dict[12] = nanmean_slow_axis12
nanmean_slow_dict[13] = nanmean_slow_axis13
nanmean_slow_dict[14] = nanmean_slow_axis14
nanmean_slow_dict[15] = nanmean_slow_axis15
nanmean_slow_dict[16] = nanmean_slow_axis16
nanmean_slow_dict[17] = nanmean_slow_axis17
nanmean_slow_dict[18] = nanmean_slow_axis18
nanmean_slow_dict[19] = nanmean_slow_axis19
nanmean_slow_dict[20] = nanmean_slow_axis20
nanmean_slow_dict[21] = nanmean_slow_axis21
nanmean_slow_dict[22] = nanmean_slow_axis22
nanmean_slow_dict[23] = nanmean_slow_axis23
nanmean_slow_dict[24] = nanmean_slow_axis24
nanmean_slow_dict[25] = nanmean_slow_axis25
nanmean_slow_dict[26] = nanmean_slow_axis26
nanmean_slow_dict[27] = nanmean_slow_axis27
nanmean_slow_dict[28] = nanmean_slow_axis28
nanmean_slow_dict[29] = nanmean_slow_axis29
nanmean_slow_dict[30] = nanmean_slow_axis30
nanmean_slow_dict[31] = nanmean_slow_axis31
nanmean_slow_dict[32] = nanmean_slow_axis32
nanmean_slow_dict[None] = nanmean_slow_axisNone

def nanmean_slow_axis0(arr):
    "Unaccelerated (slow) nanmean along axis 0."
    return bn.slow.nanmean(arr, axis=0)

def nanmean_slow_axis1(arr):
    "Unaccelerated (slow) nanmean along axis 1."
    return bn.slow.nanmean(arr, axis=1)

def nanmean_slow_axis2(arr):
    "Unaccelerated (slow) nanmean along axis 2."
    return bn.slow.nanmean(arr, axis=2)

def nanmean_slow_axis3(arr):
    "Unaccelerated (slow) nanmean along axis 3."
    return bn.slow.nanmean(arr, axis=3)

def nanmean_slow_axis4(arr):
    "Unaccelerated (slow) nanmean along axis 4."
    return bn.slow.nanmean(arr, axis=4)

def nanmean_slow_axis5(arr):
    "Unaccelerated (slow) nanmean along axis 5."
    return bn.slow.nanmean(arr, axis=5)

def nanmean_slow_axis6(arr):
    "Unaccelerated (slow) nanmean along axis 6."
    return bn.slow.nanmean(arr, axis=6)

def nanmean_slow_axis7(arr):
    "Unaccelerated (slow) nanmean along axis 7."
    return bn.slow.nanmean(arr, axis=7)

def nanmean_slow_axis8(arr):
    "Unaccelerated (slow) nanmean along axis 8."
    return bn.slow.nanmean(arr, axis=8)

def nanmean_slow_axis9(arr):
    "Unaccelerated (slow) nanmean along axis 9."
    return bn.slow.nanmean(arr, axis=9)

def nanmean_slow_axis10(arr):
    "Unaccelerated (slow) nanmean along axis 10."
    return bn.slow.nanmean(arr, axis=10)

def nanmean_slow_axis11(arr):
    "Unaccelerated (slow) nanmean along axis 11."
    return bn.slow.nanmean(arr, axis=11)

def nanmean_slow_axis12(arr):
    "Unaccelerated (slow) nanmean along axis 12."
    return bn.slow.nanmean(arr, axis=12)

def nanmean_slow_axis13(arr):
    "Unaccelerated (slow) nanmean along axis 13."
    return bn.slow.nanmean(arr, axis=13)

def nanmean_slow_axis14(arr):
    "Unaccelerated (slow) nanmean along axis 14."
    return bn.slow.nanmean(arr, axis=14)

def nanmean_slow_axis15(arr):
    "Unaccelerated (slow) nanmean along axis 15."
    return bn.slow.nanmean(arr, axis=15)

def nanmean_slow_axis16(arr):
    "Unaccelerated (slow) nanmean along axis 16."
    return bn.slow.nanmean(arr, axis=16)

def nanmean_slow_axis17(arr):
    "Unaccelerated (slow) nanmean along axis 17."
    return bn.slow.nanmean(arr, axis=17)

def nanmean_slow_axis18(arr):
    "Unaccelerated (slow) nanmean along axis 18."
    return bn.slow.nanmean(arr, axis=18)

def nanmean_slow_axis19(arr):
    "Unaccelerated (slow) nanmean along axis 19."
    return bn.slow.nanmean(arr, axis=19)

def nanmean_slow_axis20(arr):
    "Unaccelerated (slow) nanmean along axis 20."
    return bn.slow.nanmean(arr, axis=20)

def nanmean_slow_axis21(arr):
    "Unaccelerated (slow) nanmean along axis 21."
    return bn.slow.nanmean(arr, axis=21)

def nanmean_slow_axis22(arr):
    "Unaccelerated (slow) nanmean along axis 22."
    return bn.slow.nanmean(arr, axis=22)

def nanmean_slow_axis23(arr):
    "Unaccelerated (slow) nanmean along axis 23."
    return bn.slow.nanmean(arr, axis=23)

def nanmean_slow_axis24(arr):
    "Unaccelerated (slow) nanmean along axis 24."
    return bn.slow.nanmean(arr, axis=24)

def nanmean_slow_axis25(arr):
    "Unaccelerated (slow) nanmean along axis 25."
    return bn.slow.nanmean(arr, axis=25)

def nanmean_slow_axis26(arr):
    "Unaccelerated (slow) nanmean along axis 26."
    return bn.slow.nanmean(arr, axis=26)

def nanmean_slow_axis27(arr):
    "Unaccelerated (slow) nanmean along axis 27."
    return bn.slow.nanmean(arr, axis=27)

def nanmean_slow_axis28(arr):
    "Unaccelerated (slow) nanmean along axis 28."
    return bn.slow.nanmean(arr, axis=28)

def nanmean_slow_axis29(arr):
    "Unaccelerated (slow) nanmean along axis 29."
    return bn.slow.nanmean(arr, axis=29)

def nanmean_slow_axis30(arr):
    "Unaccelerated (slow) nanmean along axis 30."
    return bn.slow.nanmean(arr, axis=30)

def nanmean_slow_axis31(arr):
    "Unaccelerated (slow) nanmean along axis 31."
    return bn.slow.nanmean(arr, axis=31)

def nanmean_slow_axis32(arr):
    "Unaccelerated (slow) nanmean along axis 32."
    return bn.slow.nanmean(arr, axis=32)

def nanmean_slow_axisNone(arr):
    "Unaccelerated (slow) nanmean along axis None."
    return bn.slow.nanmean(arr, axis=None)
