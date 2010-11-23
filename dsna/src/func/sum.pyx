"sum"

cdef dict sum_dict = {}

#     Dim dtype axis
sum_dict[(1, f64, 0)] = sum_1d_float64_axis0
sum_dict[(1, f64, N)] = sum_1d_float64_axis0
sum_dict[(2, f64, 0)] = sum_2d_float64_axis0
sum_dict[(2, f64, 1)] = sum_2d_float64_axis1
sum_dict[(2, f64, N)] = sum_2d_float64_axisNone
sum_dict[(3, f64, 0)] = sum_3d_float64_axis0
sum_dict[(3, f64, 1)] = sum_3d_float64_axis1
sum_dict[(3, f64, 2)] = sum_3d_float64_axis2
sum_dict[(3, f64, N)] = sum_3d_float64_axisNone

sum_dict[(1, i32, 0)] = sum_1d_int32_axis0
sum_dict[(1, i32, N)] = sum_1d_int32_axis0
sum_dict[(2, i32, 0)] = sum_2d_int32_axis0
sum_dict[(2, i32, 1)] = sum_2d_int32_axis1
sum_dict[(2, i32, N)] = sum_2d_int32_axisNone
sum_dict[(3, i32, 0)] = sum_3d_int32_axis0
sum_dict[(3, i32, 1)] = sum_3d_int32_axis1
sum_dict[(3, i32, 2)] = sum_3d_int32_axis2
sum_dict[(3, i32, N)] = sum_3d_int32_axisNone

sum_dict[(1, i64, 0)] = sum_1d_int64_axis0
sum_dict[(1, i64, N)] = sum_1d_int64_axis0
sum_dict[(2, i64, 0)] = sum_2d_int64_axis0
sum_dict[(2, i64, 1)] = sum_2d_int64_axis1
sum_dict[(2, i64, N)] = sum_2d_int64_axisNone
sum_dict[(3, i64, 0)] = sum_3d_int64_axis0
sum_dict[(3, i64, 1)] = sum_3d_int64_axis1
sum_dict[(3, i64, 2)] = sum_3d_int64_axis2
sum_dict[(3, i64, N)] = sum_3d_int64_axisNone


def sum(arr, axis=None):
    """
    Sum of array elements along given axis treating NaNs as zero.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose sum is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is computed. The default is to compute the
        sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned. If
        the input array is of integer type that has less precision than the
        default platform integer, the default platform integer is used instead
        for accumulation of the sum and as the return values.

    Notes
    -----
    No error is raised on overflow.

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> ds.sum(1)
    1
    >>> ds.sum([1])
    1
    >>> ds.sum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> ds.sum(a)
    3.0
    >>> ds.sum(a, axis=0)
    array([ 2.,  1.])

    When positive infinity and negative infinity are present:

    >>> ds.sum([1, np.nan, np.inf])
    inf
    >>> ds.sum([1, np.nan, np.NINF])
    -inf
    >>> ds.sum([1, np.nan, np.inf, np.NINF])
    nan
    
    """
    func, arr = sum_selector(arr, axis)
    return func(arr)

def sum_selector(arr, axis=None):
    """
    Return sum function and array that matches `arr` and `axis`.

    Under the hood dsna uses a separate Cython function for each combination
    of ndim, dtype, and axis. A lot of the overhead in ds.sum() is in checking
    that `axis` is within range, converting `arr` into an array (if it is not
    already an array), and selecting the function to use to calculate the sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is to be computed. The default (axis=None) is
        to compute the sum of the flattened array.
    
    Returns
    -------
    func : function
        The sum function that matches the number of dimensions and dtype of
        the input array and the axis along which you wish to sum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to sum `arr` along axis=0:

    >>> func, a = ds.func.sum_selector(arr, axis=0)
    >>> func
    <built-in function sum_1d_float64_axis0> 
    
    Use the returned function and array to determine the sum:

    >>> func(a)
    3.0

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
        func = sum_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "sum of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int alen = a.shape[0]
    cdef np.int64_t asum = 0
    for i in range(alen):
        asum += a[i]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "sum of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int alen = a.shape[0]
    cdef np.int64_t asum = 0
    for i in range(alen):
        asum += a[i]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "sum of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int alen = a.shape[0], allnan = 1
    cdef np.float64_t asum = 0, ai
    for i in range(alen):
        ai = a[i]
        if ai == ai:
            asum += ai
            allnan = 0
    if allnan == 0:
        return np.float64(asum)
    else:
        return NAN

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(acol, dtype=np.int64)
    for j in range(acol):
        asum = 0
        for i in range(arow):
            asum += a[i,j]
        y[j] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(arow, dtype=np.int64)
    for j in range(arow):
        asum = 0
        for i in range(acol):
            asum += a[j,i]
        y[j] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0
    for j in range(acol):
        for i in range(arow):
            asum += a[i,j]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(acol, dtype=np.int64)
    for j in range(acol):
        asum = 0
        for i in range(arow):
            asum += a[i,j]
        y[j] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(arow, dtype=np.int64)
    for j in range(arow):
        asum = 0
        for i in range(acol):
            asum += a[j,i]
        y[j] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0
    for j in range(acol):
        for i in range(arow):
            asum += a[i,j]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1], allnan 
    cdef np.float64_t asum = 0, aij 
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(acol, dtype=np.float64)
    for j in range(acol):
        asum = 0
        allnan = 1
        for i in range(arow):
            aij = a[i,j]
            if aij == aij:
                asum += aij
                allnan = 0
        if allnan == 0:       
            y[j] = asum
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1], allnan
    cdef np.float64_t asum = 0, aji  
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(arow, dtype=np.float64)
    for j in range(arow):
        asum = 0
        allnan = 1
        for i in range(acol):
            aji = a[j,i]
            if aji == aji:
                asum += aji
                allnan = 0
        if allnan == 0:       
            y[j] = asum
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "sum of 2d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1], allnan = 1
    cdef np.float64_t asum = 0, aij
    for i in range(arow):
        for j in range(acol):
            aij = a[i,j]
            if aij == aij:
                asum += aij
                allnan = 0
    if allnan == 0:
        return np.float64(asum)
    else:
        return NAN

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n1, n2), dtype=np.int64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            for i in range(n0):
                asum += a[i,j,k]
            y[j, k] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n2), dtype=np.int64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            for j in range(n1):
                asum += a[i,j,k]
            y[i, k] = asum 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int32 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n1), dtype=np.int64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            for k in range(n2):
                asum += a[i,j,k]
            y[i, j] = asum 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                asum += a[i,j,k]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n1, n2), dtype=np.int64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            for i in range(n0):
                asum += a[i,j,k]
            y[j, k] = asum    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n2), dtype=np.int64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            for j in range(n1):
                asum += a[i,j,k]
            y[i, k] = asum 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int64 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0   
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n1), dtype=np.int64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            for k in range(n2):
                asum += a[i,j,k]
            y[i, j] = asum 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t asum = 0
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                asum += a[i,j,k]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            asum = 0
            allnan = 1
            for i in range(n0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[j, k] = asum
            else:
                y[j, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            asum = 0
            allnan = 1
            for j in range(n1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i, k] = asum
            else:
                y[i, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.float64 along axis=2."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            asum = 0
            allnan = 1
            for k in range(n2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    allnan = 0
            if allnan == 0:   
                y[i, j] = asum
            else:
                y[i, j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def sum_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "sum of 3d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan = 1
    cdef np.float64_t asum = 0, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    allnan = 0
    if allnan == 0:                
        return np.float64(asum)
    else:
        return NAN
