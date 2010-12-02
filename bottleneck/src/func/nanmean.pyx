"nanmean"

cdef dict nanmean_dict = {}

#     Dim dtype axis
nanmean_dict[(1, f64, 0)] = nanmean_1d_float64_axis0
nanmean_dict[(1, f64, N)] = nanmean_1d_float64_axis0
nanmean_dict[(2, f64, 0)] = nanmean_2d_float64_axis0
nanmean_dict[(2, f64, 1)] = nanmean_2d_float64_axis1
nanmean_dict[(2, f64, N)] = nanmean_2d_float64_axisNone
nanmean_dict[(3, f64, 0)] = nanmean_3d_float64_axis0
nanmean_dict[(3, f64, 1)] = nanmean_3d_float64_axis1
nanmean_dict[(3, f64, 2)] = nanmean_3d_float64_axis2
nanmean_dict[(3, f64, N)] = nanmean_3d_float64_axisNone

nanmean_dict[(1, i32, 0)] = nanmean_1d_int32_axis0
nanmean_dict[(1, i32, N)] = nanmean_1d_int32_axis0
nanmean_dict[(2, i32, 0)] = nanmean_2d_int32_axis0
nanmean_dict[(2, i32, 1)] = nanmean_2d_int32_axis1
nanmean_dict[(2, i32, N)] = nanmean_2d_int32_axisNone
nanmean_dict[(3, i32, 0)] = nanmean_3d_int32_axis0
nanmean_dict[(3, i32, 1)] = nanmean_3d_int32_axis1
nanmean_dict[(3, i32, 2)] = nanmean_3d_int32_axis2
nanmean_dict[(3, i32, N)] = nanmean_3d_int32_axisNone

nanmean_dict[(1, i64, 0)] = nanmean_1d_int64_axis0
nanmean_dict[(1, i64, N)] = nanmean_1d_int64_axis0
nanmean_dict[(2, i64, 0)] = nanmean_2d_int64_axis0
nanmean_dict[(2, i64, 1)] = nanmean_2d_int64_axis1
nanmean_dict[(2, i64, N)] = nanmean_2d_int64_axisNone
nanmean_dict[(3, i64, 0)] = nanmean_3d_int64_axis0
nanmean_dict[(3, i64, 1)] = nanmean_3d_int64_axis1
nanmean_dict[(3, i64, 2)] = nanmean_3d_int64_axis2
nanmean_dict[(3, i64, N)] = nanmean_3d_int64_axisNone


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
    cdef int size = a.size
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

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "nanmean of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0]
    cdef np.float64_t asum = 0
    for i in range(a0):
        asum += a[i]
    return np.float64(asum / a0)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "nanmean of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0]
    cdef np.float64_t asum = 0
    for i in range(a0):
        asum += a[i]
    return np.float64(asum / a0)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "nanmean of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef np.float64_t asum = 0, ai
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        for i in range(a0):
            asum += a[i,j]
        y[j] = asum / a0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for j in range(a0):
        asum = 0
        for i in range(a1):
            asum += a[j,i]
        y[j] = asum / a1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], aa = a0 * a1
    cdef np.float64_t asum = 0
    for j in range(a1):
        for i in range(a0):
            asum += a[i,j]
    return np.float64(asum / aa)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        for i in range(a0):
            asum += a[i,j]
        y[j] = asum / a0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for j in range(a0):
        asum = 0
        for i in range(a1):
            asum += a[j,i]
        y[j] = asum / a1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], aa = a0*a1
    cdef np.float64_t asum = 0
    for j in range(a1):
        for i in range(a0):
            asum += a[i,j]
    return np.float64(asum / aa) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count 
    cdef np.float64_t asum = 0, aij 
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a1, dtype=np.float64)
    for j in range(a1):
        asum = 0
        count = 0
        for i in range(a0):
            aij = a[i,j]
            if aij == aij:
                asum += aij
                count += 1
        if count > 0:       
            y[j] = asum / count
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef np.float64_t asum = 0, aji  
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(a0, dtype=np.float64)
    for j in range(a0):
        asum = 0
        count = 0
        for i in range(a1):
            aji = a[j,i]
            if aji == aji:
                asum += aji
                count += 1
        if count > 0:       
            y[j] = asum / count
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "nanmean of 2d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count = 0
    cdef np.float64_t asum = 0, aij
    for i in range(a0):
        for j in range(a1):
            aij = a[i,j]
            if aij == aij:
                asum += aij
                count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            for i in range(n0):
                asum += a[i,j,k]
            y[j, k] = asum / n0
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            for j in range(n1):
                asum += a[i,j,k]
            y[i, k] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int32 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            for k in range(n2):
                asum += a[i,j,k]
            y[i, j] = asum / n2
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], nn = n0*n1*n2
    cdef np.float64_t asum = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                asum += a[i,j,k]
    return np.float64(asum / nn) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            for i in range(n0):
                asum += a[i,j,k]
            y[j, k] = asum / n0    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            for j in range(n1):
                asum += a[i,j,k]
            y[i, k] = asum / n1
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int64 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.float64_t asum = 0   
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            for k in range(n2):
                asum += a[i,j,k]
            y[i, j] = asum / n2 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], nn = n0*n1*n2
    cdef np.float64_t asum = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                asum += a[i,j,k]
    return np.float64(asum / nn) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], count
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            count = 0
            for i in range(n0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:   
                y[j, k] = asum / count
            else:
                y[j, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], count
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            count = 0
            for j in range(n1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:   
                y[i, k] = asum / count
            else:
                y[i, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.float64 along axis=2."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], count
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            count = 0
            for k in range(n2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:   
                y[i, j] = asum / count
            else:
                y[i, j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "nanmean of 3d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], count = 0
    cdef np.float64_t asum = 0, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
    if count > 0:                
        return np.float64(asum / count)
    else:
        return np.float64(NAN)
