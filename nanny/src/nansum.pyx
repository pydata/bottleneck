"nansum"

cdef dict nansum_dict = {}

#     Dim dtype axis
nansum_dict[(1, f64, 0)] = nansum_1d_float64_axis0
nansum_dict[(1, f64, N)] = nansum_1d_float64_axis0
nansum_dict[(2, f64, 0)] = nansum_2d_float64_axis0
nansum_dict[(2, f64, 1)] = nansum_2d_float64_axis1
nansum_dict[(2, f64, N)] = nansum_2d_float64_axisNone
nansum_dict[(3, f64, 0)] = nansum_3d_float64_axis0
nansum_dict[(3, f64, 1)] = nansum_3d_float64_axis1
nansum_dict[(3, f64, 2)] = nansum_3d_float64_axis2
nansum_dict[(3, f64, N)] = nansum_3d_float64_axisNone

nansum_dict[(1, i32, 0)] = nansum_1d_int32_axis0
nansum_dict[(1, i32, N)] = nansum_1d_int32_axis0
nansum_dict[(2, i32, 0)] = nansum_2d_int32_axis0
nansum_dict[(2, i32, 1)] = nansum_2d_int32_axis1
nansum_dict[(2, i32, N)] = nansum_2d_int32_axisNone
nansum_dict[(3, i32, 0)] = nansum_3d_int32_axis0
nansum_dict[(3, i32, 1)] = nansum_3d_int32_axis1
nansum_dict[(3, i32, 2)] = nansum_3d_int32_axis2
nansum_dict[(3, i32, N)] = nansum_3d_int32_axisNone

nansum_dict[(1, i64, 0)] = nansum_1d_int64_axis0
nansum_dict[(1, i64, N)] = nansum_1d_int64_axis0
nansum_dict[(2, i64, 0)] = nansum_2d_int64_axis0
nansum_dict[(2, i64, 1)] = nansum_2d_int64_axis1
nansum_dict[(2, i64, N)] = nansum_2d_int64_axisNone
nansum_dict[(3, i64, 0)] = nansum_3d_int64_axis0
nansum_dict[(3, i64, 1)] = nansum_3d_int64_axis1
nansum_dict[(3, i64, 2)] = nansum_3d_int64_axis2
nansum_dict[(3, i64, N)] = nansum_3d_int64_axisNone


def nansum(arr, axis=None):
    """
    Return the sum of array elements over a given axis treating
    Not a Numbers (NaNs) as zero.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose sum is desired. If `a` is not an
        array, a conversion is attempted.
    axis : int, optional
        Axis along which the sum is computed. The default is to compute
        the sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as a, with the specified axis removed.
        If a is a 0-d array, or if axis is None, a scalar is returned with
        the same dtype as `a`.

    See Also
    --------
    numpy.sum : Sum across array including Not a Numbers.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite: Shows which elements are not: Not a Number, positive and
             negative infinity

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Arithmetic is modular when using integer types (all elements of `a` must
    be finite i.e. no elements that are NaNs, positive infinity and negative
    infinity because NaNs are floating point types), and no error is raised
    on overflow.
    
    Examples
    --------
    >>> ny.nansum(1)
    1
    >>> ny.nansum([1])
    1
    >>> ny.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> ny.nansum(a)
    3.0
    >>> ny.nansum(a, axis=0)
    array([ 2.,  1.])

    When positive infinity and negative infinity are present

    >>> ny.nansum([1, np.nan, np.inf])
    inf
    >>> ny.nansum([1, np.nan, np.NINF])
    -inf
    >>> ny.nansum([1, np.nan, np.inf, np.NINF])
    nan
    
    """
    func, arr = nansum_selector(arr, axis)
    return func(arr)

def nansum_selector(arr, axis):
    "Return nansum function that matches `arr` and `axis` and return `arr`."
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
        func = nansum_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "nansum of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int alen = a.shape[0]
    cdef np.int64_t asum = 0
    for i in range(alen):
        asum += a[i]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "nansum of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int alen = a.shape[0]
    cdef np.int64_t asum = 0
    for i in range(alen):
        asum += a[i]
    return np.int64(asum)

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "nansum of 1d numpy array with dtype=np.float64 along axis=0."
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
def nansum_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int32 along axis=0."
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
def nansum_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int32 along axis=1"
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
def nansum_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0
    for j in range(acol):
        for i in range(arow):
            asum += a[i,j]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int64 along axis=0."
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
def nansum_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int64 along axis=1"
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
def nansum_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int arow = a.shape[0], acol = a.shape[1]
    cdef np.int64_t asum = 0
    for j in range(acol):
        for i in range(arow):
            asum += a[i,j]
    return np.int64(asum) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nansum_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.float64 along axis=0."
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
def nansum_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.float64 along axis=1."
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
def nansum_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "nansum of 2d numpy array with dtype=np.float64 along axis=None."
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
def nansum_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int32 along axis=0."
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
def nansum_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int32 along axis=1"
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
def nansum_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int32 along axis=2"
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
def nansum_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int32 along axis=None."
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
def nansum_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int64 along axis=0."
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
def nansum_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int64 along axis=1"
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
def nansum_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int64 along axis=2"
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
def nansum_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.int64 along axis=None."
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
def nansum_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.float64 along axis=0."
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
def nansum_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.float64 along axis=1."
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
def nansum_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.float64 along axis=2."
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
def nansum_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "nansum of 3d numpy array with dtype=np.float64 along axis=None."
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
