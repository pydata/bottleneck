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
        Axis along which the mean is computed. The default is to compute the
        mean of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs. 
    
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
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmean_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Mean of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
def nanmean_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Mean of 1d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
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
def nanmean_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
def nanmean_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
def nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Mean of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
def nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Mean of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
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
    cdef int n0 = a.shape[0]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    size = n0 * n1 * n2 
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                asum += a[i0, i1, i2]
    return np.float64(asum / size)

cdef dict nanmean_dict = {}
nanmean_dict[(2, int32, 0)] = nanmean_2d_int32_axis0
nanmean_dict[(2, int32, 1)] = nanmean_2d_int32_axis1
nanmean_dict[(2, int64, 0)] = nanmean_2d_int64_axis0
nanmean_dict[(2, int64, 1)] = nanmean_2d_int64_axis1
nanmean_dict[(3, int32, 0)] = nanmean_3d_int32_axis0
nanmean_dict[(3, int32, 1)] = nanmean_3d_int32_axis1
nanmean_dict[(3, int32, 2)] = nanmean_3d_int32_axis2
nanmean_dict[(3, int64, 0)] = nanmean_3d_int64_axis0
nanmean_dict[(3, int64, 1)] = nanmean_3d_int64_axis1
nanmean_dict[(3, int64, 2)] = nanmean_3d_int64_axis2
nanmean_dict[(1, float64, 0)] = nanmean_1d_float64_axisNone
nanmean_dict[(1, float64, None)] = nanmean_1d_float64_axisNone
nanmean_dict[(2, float64, None)] = nanmean_2d_float64_axisNone
nanmean_dict[(3, float64, None)] = nanmean_3d_float64_axisNone
nanmean_dict[(2, float64, 0)] = nanmean_2d_float64_axis0
nanmean_dict[(2, float64, 1)] = nanmean_2d_float64_axis1
nanmean_dict[(3, float64, 0)] = nanmean_3d_float64_axis0
nanmean_dict[(3, float64, 1)] = nanmean_3d_float64_axis1
nanmean_dict[(3, float64, 2)] = nanmean_3d_float64_axis2
nanmean_dict[(1, int32, 0)] = nanmean_1d_int32_axisNone
nanmean_dict[(1, int32, None)] = nanmean_1d_int32_axisNone
nanmean_dict[(1, int64, 0)] = nanmean_1d_int64_axisNone
nanmean_dict[(1, int64, None)] = nanmean_1d_int64_axisNone
nanmean_dict[(2, int32, None)] = nanmean_2d_int32_axisNone
nanmean_dict[(2, int64, None)] = nanmean_2d_int64_axisNone
nanmean_dict[(3, int32, None)] = nanmean_3d_int32_axisNone
nanmean_dict[(3, int64, None)] = nanmean_3d_int64_axisNone