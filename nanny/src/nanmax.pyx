
cdef np.int32_t MININT32 = np.iinfo(np.int32).min
cdef np.int64_t MININT64 = np.iinfo(np.int64).min

select_nanmax = {}

#     Dim dtype axis
select_nanmax[(1, f64, 0)] = nanmax_1d_float64_axis0
select_nanmax[(1, f64, N)] = nanmax_1d_float64_axis0
select_nanmax[(2, f64, 0)] = nanmax_2d_float64_axis0
select_nanmax[(2, f64, 1)] = nanmax_2d_float64_axis1
select_nanmax[(2, f64, N)] = nanmax_2d_float64_axisNone
select_nanmax[(3, f64, 0)] = nanmax_3d_float64_axis0
select_nanmax[(3, f64, 1)] = nanmax_3d_float64_axis1
select_nanmax[(3, f64, 2)] = nanmax_3d_float64_axis2
select_nanmax[(3, f64, N)] = nanmax_3d_float64_axisNone

select_nanmax[(1, i32, 0)] = nanmax_1d_int32_axis0
select_nanmax[(1, i32, N)] = nanmax_1d_int32_axis0
select_nanmax[(2, i32, 0)] = nanmax_2d_int32_axis0
select_nanmax[(2, i32, 1)] = nanmax_2d_int32_axis1
select_nanmax[(2, i32, N)] = nanmax_2d_int32_axisNone
select_nanmax[(3, i32, 0)] = nanmax_3d_int32_axis0
select_nanmax[(3, i32, 1)] = nanmax_3d_int32_axis1
select_nanmax[(3, i32, 2)] = nanmax_3d_int32_axis2
select_nanmax[(3, i32, N)] = nanmax_3d_int32_axisNone

select_nanmax[(1, i64, 0)] = nanmax_1d_int64_axis0
select_nanmax[(1, i64, N)] = nanmax_1d_int64_axis0
select_nanmax[(2, i64, 0)] = nanmax_2d_int64_axis0
select_nanmax[(2, i64, 1)] = nanmax_2d_int64_axis1
select_nanmax[(2, i64, N)] = nanmax_2d_int64_axisNone
select_nanmax[(3, i64, 0)] = nanmax_3d_int64_axis0
select_nanmax[(3, i64, 1)] = nanmax_3d_int64_axis1
select_nanmax[(3, i64, 2)] = nanmax_3d_int64_axis2
select_nanmax[(3, i64, N)] = nanmax_3d_int64_axisNone


def nanmax(arr, axis=None):
    """
    Return the maximum of array elements over the given axis ignoring any NaNs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose maximum is desired. If `a` is not
        an array, a conversion is attempted.
    axis : int, optional
        Axis along which the maximum is computed.The default is to compute
        the maximum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, a scalar is returned. The
        the same dtype as `a` is returned.

    See Also
    --------
    numpy.amax : Maximum across array including any Not a Numbers.
    numpy.nanmin : Minimum across array ignoring any Not a Numbers.
    isnan : Shows which elements are Not a Number (NaN).
    isfinite: Shows which elements are not: Not a Number, positive and
             negative infinity

    Notes
    -----
    Numpy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754). This means that Not a Number is not equivalent to infinity.
    Positive infinity is treated as a very large number and negative infinity
    is treated as a very small (i.e. negative) number.

    If the input has a integer type, an integer type is returned unless
    the input contains NaNs and infinity.

    Examples
    --------
    >>> a = np.array([[1, 2], [3, np.nan]])
    >>> ny.nanmax(a)
    3.0
    >>> ny.nanmax(a, axis=0)
    array([ 3.,  2.])
    >>> ny.nanmax(a, axis=1)
    array([ 2.,  3.])

    When positive infinity and negative infinity are present:

    >>> ny.nanmax([1, 2, np.nan, np.NINF])
    2.0
    >>> ny.nanmax([1, 2, np.nan, np.inf])
    inf
    
    """
    arr = np.asarray(arr)
    ndim = arr.ndim
    dtype = arr.dtype
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    try:
        func = select_nanmax[(ndim, dtype, axis)]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func(arr)

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "nanmax of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0]
    cdef np.int32_t amax = MININT32, ai
    for i in range(n0):
        ai = a[i]
        if ai >= amax:
            amax = ai
    return np.int32(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "nanmax of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0]
    cdef np.int64_t amax = MININT64, ai
    for i in range(n0):
        ai = a[i]
        if ai >= amax:
            amax = ai
    return np.int64(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "nanmax of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0], allnan = 1
    cdef np.float64_t amax = np.NINF, ai
    for i in range(n0):
        ai = a[i]
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan == 0:
        return np.float64(amax)
    else:
        return NAN

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amax, ai  
    cdef np.ndarray[np.int32_t, ndim=1] y = np.empty(n1, dtype=np.int32)
    for j in range(n1):
        amax = MININT32
        for i in range(n0):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
        y[j] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amax = MININT32 
    cdef np.ndarray[np.int32_t, ndim=1] y = np.empty(n0, dtype=np.int32)
    for i in range(n0):
        amax = MININT32
        for j in range(n1):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
        y[i] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amax = MININT32, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
    return np.int32(amax) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amax, ai  
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(n1, dtype=np.int64)
    for j in range(n1):
        amax = MININT64
        for i in range(n0):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
        y[j] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amax, ai 
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(n0, dtype=np.int64)
    for i in range(n0):
        amax = MININT64
        for j in range(n1):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
        y[i] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amax = MININT64, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
    return np.int64(amax) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan 
    cdef np.float64_t amax, ai 
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(n1, dtype=np.float64)
    for j in range(n1):
        amax = np.NINF
        allnan = 1
        for i in range(n0):
            ai = a[i,j]
            if ai >= amax :
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[j] = amax
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan
    cdef np.float64_t amax, ai  
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(n0, dtype=np.float64)
    for j in range(n0):
        amax = np.NINF
        allnan = 1
        for i in range(n1):
            ai = a[j,i]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[j] = amax
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "nanmax of 2d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan = 1
    cdef np.float64_t amax = np.NINF, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai >= amax:
                amax = ai
                allnan = 0
    if allnan == 0:
        return np.float64(amax)
    else:
        return NAN

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int32_t amax, ai  
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n1, n2), dtype=np.int32)
    for j in range(n1):
        for k in range(n2):
            amax = MININT32
            for i in range(n0):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[j, k] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai   
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n0, n2), dtype=np.int32)
    for i in range(n0):
        for k in range(n2):
            amax = MININT32
            for j in range(n1):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[i, k] = amax 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int32 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai   
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n0, n1), dtype=np.int32)
    for i in range(n0):
        for j in range(n1):
            amax = MININT32
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[i, j] = amax 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai
    for i in range(n0):
        for j in range(n1):
            amax = MININT32
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
    return np.int32(amax) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai  
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n1, n2), dtype=np.int64)
    for j in range(n1):
        for k in range(n2):
            amax = MININT64
            for i in range(n0):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[j, k] = amax    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai 
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n2), dtype=np.int64)
    for i in range(n0):
        for k in range(n2):
            amax = MININT64
            for j in range(n1):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[i, k] = amax 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int64 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax, ai 
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n1), dtype=np.int64)
    for i in range(n0):
        for j in range(n1):
            amax = MININT64
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
            y[i, j] = amax 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amax = MININT64, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
    return np.int64(amax) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amax, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            amax = np.NINF
            allnan = 1
            for i in range(n0):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:   
                y[j, k] = amax
            else:
                y[j, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amax, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            amax = np.NINF
            allnan = 1
            for j in range(n1):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:   
                y[i, k] = amax
            else:
                y[i, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.float64 along axis=2."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amax, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            amax = np.NINF
            allnan = 1
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:   
                y[i, j] = amax
            else:
                y[i, j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "nanmax of 3d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan = 1
    cdef np.float64_t amax = np.NINF, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai >= amax:
                    amax = ai
                    allnan = 0
    if allnan == 0:                
        return np.float64(amax)
    else:
        return NAN
