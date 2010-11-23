
cdef np.int32_t MAXINT32 = np.iinfo(np.int32).max
cdef np.int64_t MAXINT64 = np.iinfo(np.int64).max

cdef dict nanmin_dict = {}

#          Dim dtype axis
nanmin_dict[(1, f64, 0)] = nanmin_1d_float64_axis0
nanmin_dict[(1, f64, N)] = nanmin_1d_float64_axis0
nanmin_dict[(2, f64, 0)] = nanmin_2d_float64_axis0
nanmin_dict[(2, f64, 1)] = nanmin_2d_float64_axis1
nanmin_dict[(2, f64, N)] = nanmin_2d_float64_axisNone
nanmin_dict[(3, f64, 0)] = nanmin_3d_float64_axis0
nanmin_dict[(3, f64, 1)] = nanmin_3d_float64_axis1
nanmin_dict[(3, f64, 2)] = nanmin_3d_float64_axis2
nanmin_dict[(3, f64, N)] = nanmin_3d_float64_axisNone

nanmin_dict[(1, i32, 0)] = nanmin_1d_int32_axis0
nanmin_dict[(1, i32, N)] = nanmin_1d_int32_axis0
nanmin_dict[(2, i32, 0)] = nanmin_2d_int32_axis0
nanmin_dict[(2, i32, 1)] = nanmin_2d_int32_axis1
nanmin_dict[(2, i32, N)] = nanmin_2d_int32_axisNone
nanmin_dict[(3, i32, 0)] = nanmin_3d_int32_axis0
nanmin_dict[(3, i32, 1)] = nanmin_3d_int32_axis1
nanmin_dict[(3, i32, 2)] = nanmin_3d_int32_axis2
nanmin_dict[(3, i32, N)] = nanmin_3d_int32_axisNone

nanmin_dict[(1, i64, 0)] = nanmin_1d_int64_axis0
nanmin_dict[(1, i64, N)] = nanmin_1d_int64_axis0
nanmin_dict[(2, i64, 0)] = nanmin_2d_int64_axis0
nanmin_dict[(2, i64, 1)] = nanmin_2d_int64_axis1
nanmin_dict[(2, i64, N)] = nanmin_2d_int64_axisNone
nanmin_dict[(3, i64, 0)] = nanmin_3d_int64_axis0
nanmin_dict[(3, i64, 1)] = nanmin_3d_int64_axis1
nanmin_dict[(3, i64, 2)] = nanmin_3d_int64_axis2
nanmin_dict[(3, i64, N)] = nanmin_3d_int64_axisNone


def nanmin(arr, axis=None):
    """
    Return the minimum of array elements over the given axis ignoring any NaNs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose minimum is desired. If `a` is not
        an array, a conversion is attempted.
    axis : int, optional
        Axis along which the minimum is computed.The default is to compute
        the minimum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `a`, with the specified axis removed.
        If `a` is a 0-d array, or if axis is None, a scalar is returned. The
        the same dtype as `a` is returned.

    See Also
    --------
    numpy.amin : Maximum across array including any Not a Numbers.
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
    >>> ds.nanmin(a)
    3.0
    >>> ds.nanmin(a, axis=0)
    array([ 3.,  2.])
    >>> ds.nanmin(a, axis=1)
    array([ 2.,  3.])

    When positive infinity and negative infinity are present:

    >>> ds.nanmin([1, 2, np.nan, np.NINF])
    2.0
    >>> ds.nanmin([1, 2, np.nan, np.inf])
    inf
    
    """
    func, arr = nanmin_selector(arr, axis)
    return func(arr)

def nanmin_selector(arr, axis):
    "Return nanmin function that matches `arr` and `axis` and return `arr`."
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    cdef int size = a.size
    if size == 0:
        msg = "numpy.nanmin() raises on size=0 input; sdsnaa does too." 
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

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "nanmin of 1d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0]
    cdef np.int32_t amin = MAXINT32, ai
    for i in range(n0):
        ai = a[i]
        if ai <= amin:
            amin = ai
    return np.int32(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "nanmin of 1d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0]
    cdef np.int64_t amin = MAXINT64, ai
    for i in range(n0):
        ai = a[i]
        if ai <= amin:
            amin = ai
    return np.int64(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "nanmin of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int n0 = a.shape[0], allnan = 1
    cdef np.float64_t amin = np.inf, ai
    for i in range(n0):
        ai = a[i]
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan == 0:
        return np.float64(amin)
    else:
        return NAN

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amin, ai  
    cdef np.ndarray[np.int32_t, ndim=1] y = np.empty(n1, dtype=np.int32)
    for j in range(n1):
        amin = MAXINT32
        for i in range(n0):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
        y[j] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amin 
    cdef np.ndarray[np.int32_t, ndim=1] y = np.empty(n0, dtype=np.int32)
    for i in range(n0):
        amin = MAXINT32
        for j in range(n1):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
        y[i] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int32_t amin = MAXINT32, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
    return np.int32(amin) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amin, ai  
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(n1, dtype=np.int64)
    for j in range(n1):
        amin = MAXINT64
        for i in range(n0):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
        y[j] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amin, ai 
    cdef np.ndarray[np.int64_t, ndim=1] y = np.empty(n0, dtype=np.int64)
    for i in range(n0):
        amin = MAXINT64
        for j in range(n1):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
        y[i] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1]
    cdef np.int64_t amin = MAXINT64, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
    return np.int64(amin) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan 
    cdef np.float64_t amin, ai 
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(n1, dtype=np.float64)
    for j in range(n1):
        amin = np.inf
        allnan = 1
        for i in range(n0):
            ai = a[i,j]
            if ai <= amin :
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[j] = amin
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan
    cdef np.float64_t amin, ai  
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(n0, dtype=np.float64)
    for j in range(n0):
        amin = np.inf
        allnan = 1
        for i in range(n1):
            ai = a[j,i]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[j] = amin
        else:
            y[j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "nanmin of 2d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j
    cdef int n0 = a.shape[0], n1 = a.shape[1], allnan = 1
    cdef np.float64_t amin = np.inf, ai
    for i in range(n0):
        for j in range(n1):
            ai = a[i,j]
            if ai <= amin:
                amin = ai
                allnan = 0
    if allnan == 0:
        return np.float64(amin)
    else:
        return NAN

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int32 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int32_t amin, ai  
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n1, n2), dtype=np.int32)
    for j in range(n1):
        for k in range(n2):
            amin = MAXINT32
            for i in range(n0):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[j, k] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int32 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin, ai   
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n0, n2), dtype=np.int32)
    for i in range(n0):
        for k in range(n2):
            amin = MAXINT32
            for j in range(n1):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[i, k] = amin 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int32 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin, ai   
    cdef np.ndarray[np.int32_t, ndim=2] y = np.empty((n0, n1), dtype=np.int32)
    for i in range(n0):
        for j in range(n1):
            amin = MAXINT32
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[i, j] = amin 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int32 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin = MAXINT32, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
    return np.int32(amin) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin, ai  
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n1, n2), dtype=np.int64)
    for j in range(n1):
        for k in range(n2):
            amin = MAXINT64
            for i in range(n0):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[j, k] = amin    
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int64 along axis=1"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin, ai 
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n2), dtype=np.int64)
    for i in range(n0):
        for k in range(n2):
            amin = MAXINT64
            for j in range(n1):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[i, k] = amin 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int64 along axis=2"
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin, ai 
    cdef np.ndarray[np.int64_t, ndim=2] y = np.empty((n0, n1), dtype=np.int64)
    for i in range(n0):
        for j in range(n1):
            amin = MAXINT64
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
            y[i, j] = amin 
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.int64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2]
    cdef np.int64_t amin = MAXINT64, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
    return np.int64(amin) 

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amin, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n1, n2),
                                                       dtype=np.float64)
    for j in range(n1):
        for k in range(n2):
            amin = np.inf
            allnan = 1
            for i in range(n0):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:   
                y[j, k] = amin
            else:
                y[j, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amin, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n2),
                                                       dtype=np.float64)
    for i in range(n0):
        for k in range(n2):
            amin = np.inf
            allnan = 1
            for j in range(n1):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:   
                y[i, k] = amin
            else:
                y[i, k] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.float64 along axis=2."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan
    cdef np.float64_t amin, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((n0, n1),
                                                       dtype=np.float64)
    for i in range(n0):
        for j in range(n1):
            amin = np.inf
            allnan = 1
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:   
                y[i, j] = amin
            else:
                y[i, j] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "nanmin of 3d numpy array with dtype=np.float64 along axis=None."
    cdef Py_ssize_t i, j, k
    cdef int n0 = a.shape[0], n1 = a.shape[1], n2 = a.shape[2], allnan = 1
    cdef np.float64_t amin = np.inf, ai
    for i in range(n0):
        for j in range(n1):
            for k in range(n2):
                ai = a[i,j,k]
                if ai <= amin:
                    amin = ai
                    allnan = 0
    if allnan == 0:                
        return np.float64(amin)
    else:
        return NAN
