"nanstd auto-generated from template"

def nanstd(arr, axis=None, int ddof=0):
    """
    Standard deviation along the specified axis, ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the standard deviation is computed. The default is
        to compute the standard deviation of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs. 
    
    Notes
    -----
    No error is raised on overflow.

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
        func = nanstd_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / n0
        asum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
        y[i1] = sqrt(asum / (n0 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / n1
        asum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
        y[i0] = sqrt(asum / (n1 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / n0
        asum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
        y[i1] = sqrt(asum / (n0 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / n1
        asum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
        y[i0] = sqrt(asum / (n1 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n0
            asum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i1, i2] = sqrt(asum / (n0 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n1
            asum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i0, i2] = sqrt(asum / (n1 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n2
            asum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i0, i1] = sqrt(asum / (n2 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n0
            asum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i1, i2] = sqrt(asum / (n0 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n1
            asum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i0, i2] = sqrt(asum / (n1 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / n2
            asum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
            y[i0, i1] = sqrt(asum / (n2 - ddof))
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a, int ddof):
    "Variance of 1d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            ai = a[i0]
            if ai == ai:
                ai -= amean
                asum += (ai * ai)
        return np.float32(sqrt(asum / (count - ddof)))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a, int ddof):
    "Variance of 1d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            ai = a[i0]
            if ai == ai:
                ai -= amean
                asum += (ai * ai)
        return np.float64(sqrt(asum / (count - ddof)))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
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
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
        return np.float32(sqrt(asum / (count - ddof)))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
        return np.float64(sqrt(asum / (count - ddof)))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
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
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
        return np.float32(sqrt(asum / (count - ddof)))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
        amean = asum / count
        asum = 0
        for i0 in range(n0):
            for i1 in range(n1):
                for i2 in range(n2):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
        return np.float64(sqrt(asum / (count - ddof)))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
            amean = asum / count
            asum = 0
            for i0 in range(n0):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[i1] = sqrt(asum / (count - ddof))
        else:
            y[i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
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
            amean = asum / count
            asum = 0
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[i0] = sqrt(asum / (count - ddof))
        else:
            y[i0] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / count
            asum = 0
            for i0 in range(n0):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[i1] = sqrt(asum / (count - ddof))
        else:
            y[i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a, int ddof):
    "Variance of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
            amean = asum / count
            asum = 0
            for i1 in range(n1):
                ai = a[i0, i1]
                if ai == ai:
                    ai -= amean
                    asum += (ai * ai)
            y[i0] = sqrt(asum / (count - ddof))
        else:
            y[i0] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
                amean = asum / count
                asum = 0
                for i0 in range(n0):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i1, i2] = sqrt(asum / (count - ddof))
            else:
                y[i1, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
                amean = asum / count
                asum = 0
                for i1 in range(n1):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i0, i2] = sqrt(asum / (count - ddof))
            else:
                y[i0, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float32_t asum = 0, amean, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
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
                amean = asum / count
                asum = 0
                for i2 in range(n2):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i0, i1] = sqrt(asum / (count - ddof))
            else:
                y[i0, i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
                amean = asum / count
                asum = 0
                for i0 in range(n0):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i1, i2] = sqrt(asum / (count - ddof))
            else:
                y[i1, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
                amean = asum / count
                asum = 0
                for i1 in range(n1):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i0, i2] = sqrt(asum / (count - ddof))
            else:
                y[i0, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a, int ddof):
    "Variance of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef np.float64_t asum = 0, amean, ai
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
                amean = asum / count
                asum = 0
                for i2 in range(n2):
                    ai = a[i0, i1, i2]
                    if ai == ai:
                        ai -= amean
                        asum += (ai * ai)
                y[i0, i1] = sqrt(asum / (count - ddof))
            else:
                y[i0, i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a, int ddof):
    "Valriance of 1d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    size = n0    
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            ai -= amean
            asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof)))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a, int ddof):
    "Valriance of 1d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
    cdef int size
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    size = n0    
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        ai = a[i0]
        if ai == ai:
            ai -= amean
            asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof)))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
    cdef int size
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    size = n0 * n1
    for i0 in range(n0):
        for i1 in range(n1):
            asum += a[i0, i1]
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof)))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a, int ddof):
    "Valriance of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
    cdef int size
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    size = n0 * n1
    for i0 in range(n0):
        for i1 in range(n1):
            asum += a[i0, i1]
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ai -= amean
            asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof)))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof))) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanstd_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a, int ddof):
    "Valriance of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef np.float64_t asum = 0, amean, ai
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
    amean = asum / size
    asum = 0
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ai -= amean
                asum += (ai * ai)
    return np.float64(sqrt(asum / (size - ddof))) 

cdef dict nanstd_dict = {}
nanstd_dict[(2, int32, 0)] = nanstd_2d_int32_axis0
nanstd_dict[(2, int32, 1)] = nanstd_2d_int32_axis1
nanstd_dict[(2, int64, 0)] = nanstd_2d_int64_axis0
nanstd_dict[(2, int64, 1)] = nanstd_2d_int64_axis1
nanstd_dict[(3, int32, 0)] = nanstd_3d_int32_axis0
nanstd_dict[(3, int32, 1)] = nanstd_3d_int32_axis1
nanstd_dict[(3, int32, 2)] = nanstd_3d_int32_axis2
nanstd_dict[(3, int64, 0)] = nanstd_3d_int64_axis0
nanstd_dict[(3, int64, 1)] = nanstd_3d_int64_axis1
nanstd_dict[(3, int64, 2)] = nanstd_3d_int64_axis2
nanstd_dict[(1, float32, 0)] = nanstd_1d_float32_axisNone
nanstd_dict[(1, float32, None)] = nanstd_1d_float32_axisNone
nanstd_dict[(1, float64, 0)] = nanstd_1d_float64_axisNone
nanstd_dict[(1, float64, None)] = nanstd_1d_float64_axisNone
nanstd_dict[(2, float32, None)] = nanstd_2d_float32_axisNone
nanstd_dict[(2, float64, None)] = nanstd_2d_float64_axisNone
nanstd_dict[(3, float32, None)] = nanstd_3d_float32_axisNone
nanstd_dict[(3, float64, None)] = nanstd_3d_float64_axisNone
nanstd_dict[(2, float32, 0)] = nanstd_2d_float32_axis0
nanstd_dict[(2, float32, 1)] = nanstd_2d_float32_axis1
nanstd_dict[(2, float64, 0)] = nanstd_2d_float64_axis0
nanstd_dict[(2, float64, 1)] = nanstd_2d_float64_axis1
nanstd_dict[(3, float32, 0)] = nanstd_3d_float32_axis0
nanstd_dict[(3, float32, 1)] = nanstd_3d_float32_axis1
nanstd_dict[(3, float32, 2)] = nanstd_3d_float32_axis2
nanstd_dict[(3, float64, 0)] = nanstd_3d_float64_axis0
nanstd_dict[(3, float64, 1)] = nanstd_3d_float64_axis1
nanstd_dict[(3, float64, 2)] = nanstd_3d_float64_axis2
nanstd_dict[(1, int32, 0)] = nanstd_1d_int32_axisNone
nanstd_dict[(1, int32, None)] = nanstd_1d_int32_axisNone
nanstd_dict[(1, int64, 0)] = nanstd_1d_int64_axisNone
nanstd_dict[(1, int64, None)] = nanstd_1d_int64_axisNone
nanstd_dict[(2, int32, None)] = nanstd_2d_int32_axisNone
nanstd_dict[(2, int64, None)] = nanstd_2d_int64_axisNone
nanstd_dict[(3, int32, None)] = nanstd_3d_int32_axisNone
nanstd_dict[(3, int64, None)] = nanstd_3d_int64_axisNone