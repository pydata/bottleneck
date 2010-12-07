
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
    Minimum along the specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is computed. The default is to compute
        the minimum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    
    Examples
    --------
    >>> bn.nanmin(1)
    1
    >>> bn.nanmin([1])
    1
    >>> bn.nanmin([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmin(a)
    1.0
    >>> bn.nanmin(a, axis=0)
    array([ 1.,  4.])
    
    """
    func, arr = nanmin_selector(arr, axis)
    return func(arr)

def nanmin_selector(arr, axis):
    """
    Return nanmin function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanmin()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use
    to calculate the minimum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is to be computed. The default
        (axis=None) is to compute the minimum of the flattened array.
    
    Returns
    -------
    func : function
        The nanmin function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to find
        the minimum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the minimum of `arr` along
    axis=0:

    >>> func, a = bn.func.nanmin_selector(arr, axis=0)
    >>> func
    <built-in function nanmin_1d_float64_axis0> 
    
    Use the returned function and array to determine the minimum:
    
    >>> func(a)
    1.0

    """
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    cdef int size = a.size
    if size == 0:
        msg = "numpy.nanmin() raises on size=0 input; so Bottleneck does too." 
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
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                          NPY_int32, 0)
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
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                          NPY_int32, 0)
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
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                          NPY_int64, 0)
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
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                          NPY_int64, 0)
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
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
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
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
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
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                          NPY_int32, 0)
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
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                          NPY_int32, 0)
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
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_int32, 0)
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
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                          NPY_int64, 0)
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
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                          NPY_int64, 0)
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
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                          NPY_int64, 0)
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
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
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
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
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
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
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
