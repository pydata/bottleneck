"nanvar auto-generated from template"

def nanvar(arr, axis=None, int ddof=0):
    """
    Variance along the specified axis, ignoring NaNs.

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
        Axis along which the variance is computed. The default (axis=None)is
        to compute the variance of the flattened array.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis
        removed. If `arr` is a 0-d array, or if axis is None, a scalar is
        returned. `float64` intermediate and return values are used for
        integer inputs. 
    
    See also
    --------
    bottleneck.nanstd: Standard deviation along specified axis ignoring NaNs.
    
    Notes
    -----
    If positive or negative infinity are present the result is Not A Number
    (NaN).

    Examples
    --------
    >>> bn.nanvar(1)
    0.0
    >>> bn.nanvar([1])
    0.0
    >>> bn.nanvar([1, np.nan])
    0.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanvar(a)
    2.0
    >>> bn.nanvar(a, axis=0)
    array([ 0.,  0.])

    When positive infinity or negative infinity are present NaN is returned:

    >>> bn.nanvar([1, np.nan, np.inf])
    nan
    
    """
    func, arr = nanvar_selector(arr, axis)
    return func(arr, ddof)

def nanvar_selector(arr, axis):
    """
    Return nanvar function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanvar()
    is in checking that `axis` is within range, converting `arr` into an array
    (if it is not already an array), and selecting the function to use to
    calculate the variance.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the variance is to be computed. The default
        (axis=None) is to compute the variance of the flattened array.
    
    Returns
    -------
    func : function
        The nanvar function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the
        variance.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the variance of `arr` along
    axis=0:

    >>> func, a = ds.func.nanvar_selector(arr, axis=0)
    >>> func
    <built-in function nanvar_1d_float64_axis0> 
    
    Use the returned function and array to determine the variance:
    
    >>> func(a)
    0.66666666666666663

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanvar_dict[key]
    except KeyError:
        try:
            func = nanvar_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a, int ddof):
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
        y[i1] = asum / (n0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a, int ddof):
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
        y[i0] = asum / (n1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a, int ddof):
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
        y[i1] = asum / (n0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a, int ddof):
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
        y[i0] = asum / (n1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a, int ddof):
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
            y[i1, i2] = asum / (n0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a, int ddof):
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
            y[i0, i2] = asum / (n1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a, int ddof):
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
            y[i0, i1] = asum / (n2 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a, int ddof):
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
            y[i1, i2] = asum / (n0 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a, int ddof):
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
            y[i0, i2] = asum / (n1 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a, int ddof):
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
            y[i0, i1] = asum / (n2 - ddof)
    return y 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a, int ddof):
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
        return np.float32(asum / (count - ddof))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a, int ddof):
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
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a, int ddof):
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
        return np.float32(asum / (count - ddof))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a, int ddof):
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
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a, int ddof):
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
        return np.float32(asum / (count - ddof))
    else:
        return np.float32(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a, int ddof):
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
        return np.float64(asum / (count - ddof))
    else:
        return np.float64(NAN)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a, int ddof):
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
            y[i1] = asum / (count - ddof)
        else:
            y[i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a, int ddof):
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
            y[i0] = asum / (count - ddof)
        else:
            y[i0] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a, int ddof):
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
            y[i1] = asum / (count - ddof)
        else:
            y[i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a, int ddof):
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
            y[i0] = asum / (count - ddof)
        else:
            y[i0] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a, int ddof):
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
                y[i1, i2] = asum / (count - ddof)
            else:
                y[i1, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a, int ddof):
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
                y[i0, i2] = asum / (count - ddof)
            else:
                y[i0, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a, int ddof):
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
                y[i0, i1] = asum / (count - ddof)
            else:
                y[i0, i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a, int ddof):
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
                y[i1, i2] = asum / (count - ddof)
            else:
                y[i1, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a, int ddof):
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
                y[i0, i2] = asum / (count - ddof)
            else:
                y[i0, i2] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a, int ddof):
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
                y[i0, i1] = asum / (count - ddof)
            else:
                y[i0, i1] = NAN
    return y  

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a, int ddof):
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
    return np.float64(asum / (size - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a, int ddof):
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
    return np.float64(asum / (size - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a, int ddof):
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
    return np.float64(asum / (size - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a, int ddof):
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
    return np.float64(asum / (size - ddof))

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a, int ddof):
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
    return np.float64(asum / (size - ddof)) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanvar_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a, int ddof):
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
    return np.float64(asum / (size - ddof)) 

cdef dict nanvar_dict = {}
nanvar_dict[(2, int32, 0)] = nanvar_2d_int32_axis0
nanvar_dict[(2, int32, 1)] = nanvar_2d_int32_axis1
nanvar_dict[(2, int64, 0)] = nanvar_2d_int64_axis0
nanvar_dict[(2, int64, 1)] = nanvar_2d_int64_axis1
nanvar_dict[(3, int32, 0)] = nanvar_3d_int32_axis0
nanvar_dict[(3, int32, 1)] = nanvar_3d_int32_axis1
nanvar_dict[(3, int32, 2)] = nanvar_3d_int32_axis2
nanvar_dict[(3, int64, 0)] = nanvar_3d_int64_axis0
nanvar_dict[(3, int64, 1)] = nanvar_3d_int64_axis1
nanvar_dict[(3, int64, 2)] = nanvar_3d_int64_axis2
nanvar_dict[(1, float32, 0)] = nanvar_1d_float32_axisNone
nanvar_dict[(1, float32, None)] = nanvar_1d_float32_axisNone
nanvar_dict[(1, float64, 0)] = nanvar_1d_float64_axisNone
nanvar_dict[(1, float64, None)] = nanvar_1d_float64_axisNone
nanvar_dict[(2, float32, None)] = nanvar_2d_float32_axisNone
nanvar_dict[(2, float64, None)] = nanvar_2d_float64_axisNone
nanvar_dict[(3, float32, None)] = nanvar_3d_float32_axisNone
nanvar_dict[(3, float64, None)] = nanvar_3d_float64_axisNone
nanvar_dict[(2, float32, 0)] = nanvar_2d_float32_axis0
nanvar_dict[(2, float32, 1)] = nanvar_2d_float32_axis1
nanvar_dict[(2, float64, 0)] = nanvar_2d_float64_axis0
nanvar_dict[(2, float64, 1)] = nanvar_2d_float64_axis1
nanvar_dict[(3, float32, 0)] = nanvar_3d_float32_axis0
nanvar_dict[(3, float32, 1)] = nanvar_3d_float32_axis1
nanvar_dict[(3, float32, 2)] = nanvar_3d_float32_axis2
nanvar_dict[(3, float64, 0)] = nanvar_3d_float64_axis0
nanvar_dict[(3, float64, 1)] = nanvar_3d_float64_axis1
nanvar_dict[(3, float64, 2)] = nanvar_3d_float64_axis2
nanvar_dict[(1, int32, 0)] = nanvar_1d_int32_axisNone
nanvar_dict[(1, int32, None)] = nanvar_1d_int32_axisNone
nanvar_dict[(1, int64, 0)] = nanvar_1d_int64_axisNone
nanvar_dict[(1, int64, None)] = nanvar_1d_int64_axisNone
nanvar_dict[(2, int32, None)] = nanvar_2d_int32_axisNone
nanvar_dict[(2, int64, None)] = nanvar_2d_int64_axisNone
nanvar_dict[(3, int32, None)] = nanvar_3d_int32_axisNone
nanvar_dict[(3, int64, None)] = nanvar_3d_int64_axisNone

cdef dict nanvar_slow_dict = {}
nanvar_slow_dict[0] = nanvar_slow_axis0
nanvar_slow_dict[1] = nanvar_slow_axis1
nanvar_slow_dict[2] = nanvar_slow_axis2
nanvar_slow_dict[3] = nanvar_slow_axis3
nanvar_slow_dict[4] = nanvar_slow_axis4
nanvar_slow_dict[5] = nanvar_slow_axis5
nanvar_slow_dict[6] = nanvar_slow_axis6
nanvar_slow_dict[7] = nanvar_slow_axis7
nanvar_slow_dict[8] = nanvar_slow_axis8
nanvar_slow_dict[9] = nanvar_slow_axis9
nanvar_slow_dict[10] = nanvar_slow_axis10
nanvar_slow_dict[11] = nanvar_slow_axis11
nanvar_slow_dict[12] = nanvar_slow_axis12
nanvar_slow_dict[13] = nanvar_slow_axis13
nanvar_slow_dict[14] = nanvar_slow_axis14
nanvar_slow_dict[15] = nanvar_slow_axis15
nanvar_slow_dict[16] = nanvar_slow_axis16
nanvar_slow_dict[17] = nanvar_slow_axis17
nanvar_slow_dict[18] = nanvar_slow_axis18
nanvar_slow_dict[19] = nanvar_slow_axis19
nanvar_slow_dict[20] = nanvar_slow_axis20
nanvar_slow_dict[21] = nanvar_slow_axis21
nanvar_slow_dict[22] = nanvar_slow_axis22
nanvar_slow_dict[23] = nanvar_slow_axis23
nanvar_slow_dict[24] = nanvar_slow_axis24
nanvar_slow_dict[25] = nanvar_slow_axis25
nanvar_slow_dict[26] = nanvar_slow_axis26
nanvar_slow_dict[27] = nanvar_slow_axis27
nanvar_slow_dict[28] = nanvar_slow_axis28
nanvar_slow_dict[29] = nanvar_slow_axis29
nanvar_slow_dict[30] = nanvar_slow_axis30
nanvar_slow_dict[31] = nanvar_slow_axis31
nanvar_slow_dict[32] = nanvar_slow_axis32
nanvar_slow_dict[None] = nanvar_slow_axisNone

def nanvar_slow_axis0(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 0."
    return bn.slow.nanvar(arr, axis=0, ddof=ddof)

def nanvar_slow_axis1(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 1."
    return bn.slow.nanvar(arr, axis=1, ddof=ddof)

def nanvar_slow_axis2(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 2."
    return bn.slow.nanvar(arr, axis=2, ddof=ddof)

def nanvar_slow_axis3(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 3."
    return bn.slow.nanvar(arr, axis=3, ddof=ddof)

def nanvar_slow_axis4(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 4."
    return bn.slow.nanvar(arr, axis=4, ddof=ddof)

def nanvar_slow_axis5(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 5."
    return bn.slow.nanvar(arr, axis=5, ddof=ddof)

def nanvar_slow_axis6(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 6."
    return bn.slow.nanvar(arr, axis=6, ddof=ddof)

def nanvar_slow_axis7(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 7."
    return bn.slow.nanvar(arr, axis=7, ddof=ddof)

def nanvar_slow_axis8(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 8."
    return bn.slow.nanvar(arr, axis=8, ddof=ddof)

def nanvar_slow_axis9(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 9."
    return bn.slow.nanvar(arr, axis=9, ddof=ddof)

def nanvar_slow_axis10(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 10."
    return bn.slow.nanvar(arr, axis=10, ddof=ddof)

def nanvar_slow_axis11(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 11."
    return bn.slow.nanvar(arr, axis=11, ddof=ddof)

def nanvar_slow_axis12(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 12."
    return bn.slow.nanvar(arr, axis=12, ddof=ddof)

def nanvar_slow_axis13(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 13."
    return bn.slow.nanvar(arr, axis=13, ddof=ddof)

def nanvar_slow_axis14(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 14."
    return bn.slow.nanvar(arr, axis=14, ddof=ddof)

def nanvar_slow_axis15(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 15."
    return bn.slow.nanvar(arr, axis=15, ddof=ddof)

def nanvar_slow_axis16(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 16."
    return bn.slow.nanvar(arr, axis=16, ddof=ddof)

def nanvar_slow_axis17(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 17."
    return bn.slow.nanvar(arr, axis=17, ddof=ddof)

def nanvar_slow_axis18(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 18."
    return bn.slow.nanvar(arr, axis=18, ddof=ddof)

def nanvar_slow_axis19(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 19."
    return bn.slow.nanvar(arr, axis=19, ddof=ddof)

def nanvar_slow_axis20(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 20."
    return bn.slow.nanvar(arr, axis=20, ddof=ddof)

def nanvar_slow_axis21(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 21."
    return bn.slow.nanvar(arr, axis=21, ddof=ddof)

def nanvar_slow_axis22(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 22."
    return bn.slow.nanvar(arr, axis=22, ddof=ddof)

def nanvar_slow_axis23(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 23."
    return bn.slow.nanvar(arr, axis=23, ddof=ddof)

def nanvar_slow_axis24(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 24."
    return bn.slow.nanvar(arr, axis=24, ddof=ddof)

def nanvar_slow_axis25(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 25."
    return bn.slow.nanvar(arr, axis=25, ddof=ddof)

def nanvar_slow_axis26(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 26."
    return bn.slow.nanvar(arr, axis=26, ddof=ddof)

def nanvar_slow_axis27(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 27."
    return bn.slow.nanvar(arr, axis=27, ddof=ddof)

def nanvar_slow_axis28(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 28."
    return bn.slow.nanvar(arr, axis=28, ddof=ddof)

def nanvar_slow_axis29(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 29."
    return bn.slow.nanvar(arr, axis=29, ddof=ddof)

def nanvar_slow_axis30(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 30."
    return bn.slow.nanvar(arr, axis=30, ddof=ddof)

def nanvar_slow_axis31(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 31."
    return bn.slow.nanvar(arr, axis=31, ddof=ddof)

def nanvar_slow_axis32(arr, ddof):
    "Unaccelerated (slow) nanvar along axis 32."
    return bn.slow.nanvar(arr, axis=32, ddof=ddof)

def nanvar_slow_axisNone(arr, ddof):
    "Unaccelerated (slow) nanvar along axis None."
    return bn.slow.nanvar(arr, axis=None, ddof=ddof)
