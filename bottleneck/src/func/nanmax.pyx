"nanmax auto-generated from template"

def nanmax(arr, axis=None):
    """
    Maximum along the specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the maximum is computed. The default is to compute
        the maximum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
    
    Examples
    --------
    >>> bn.nanmax(1)
    1
    >>> bn.nanmax([1])
    1
    >>> bn.nanmax([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmax(a)
    4.0
    >>> bn.nanmax(a, axis=0)
    array([ 1.,  4.])
    
    """
    func, arr = nanmax_selector(arr, axis)
    return func(arr)

def nanmax_selector(arr, axis):
    """
    Return nanmax function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.nanmax()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use
    to calculate the maximum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the maximum is to be computed. The default
        (axis=None) is to compute the maximum of the flattened array.
    
    Returns
    -------
    func : function
        The nanamx function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to find
        the maximum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the maximum of `arr` along
    axis=0:

    >>> func, a = bn.func.nanmax_selector(arr, axis=0)
    >>> func
    <built-in function nanmax_1d_float64_axis0> 
    
    Use the returned function and array to determine the maximum:
    
    >>> func(a)
    3.0

    """
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    cdef int size = a.size
    if size == 0:
        msg = "numpy.nanmax() raises on size=0 input; so Bottleneck does too." 
        raise ValueError, msg
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmax_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Maximum of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        amax = MINint32
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
        y[i1] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Maximum of 2d array with dtype=int32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        amax = MINint32
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
        y[i0] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Maximum of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        amax = MINint64
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
        y[i1] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Maximum of 2d array with dtype=int64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        amax = MINint64
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
        y[i0] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Maximum of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amax = MINint32
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i1, i2] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Maximum of 3d array with dtype=int32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amax = MINint32
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i0, i2] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Maximum of 3d array with dtype=int32 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amax = MINint32
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i0, i1] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Maximum of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amax = MINint64
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i1, i2] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Maximum of 3d array with dtype=int64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amax = MINint64
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i0, i2] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Maximum of 3d array with dtype=int64 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amax = MINint64
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
            y[i0, i1] = amax
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Maximum of 1d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amax = MINfloat32
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan == 0:       
        return np.float32(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Maximum of 1d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amax = MINfloat64
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            allnan = 0
    if allnan == 0:       
        return np.float64(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Maximum of 2d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amax = MINfloat32
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
    if allnan == 0:       
        return np.float32(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Maximum of 2d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amax = MINfloat64
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
    if allnan == 0:       
        return np.float64(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a):
    "Maximum of 3d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amax = MINfloat32
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
    if allnan == 0:       
        return np.float32(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Maximum of 3d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amax = MINfloat64
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
    if allnan == 0:       
        return np.float64(amax)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Maximum of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        amax = MINfloat32
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[i1] = amax
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Maximum of 2d array with dtype=float32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        amax = MINfloat32
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[i0] = amax
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Maximum of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        amax = MINfloat64
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[i1] = amax
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Maximum of 2d array with dtype=float64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        amax = MINfloat64
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:       
            y[i0] = amax
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Maximum of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amax = MINfloat32
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i1, i2] = amax
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Maximum of 3d array with dtype=float32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amax = MINfloat32
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i2] = amax
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Maximum of 3d array with dtype=float32 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amax = MINfloat32
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i1] = amax
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Maximum of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amax = MINfloat64
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i1, i2] = amax
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Maximum of 3d array with dtype=float64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amax = MINfloat64
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i2] = amax
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Maximum of 3d array with dtype=float64 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amax = MINfloat64
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i1] = amax
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Maximum of 1d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amax = MINint32
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
    return np.int32(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Maximum of 1d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amax = MINint64
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
    return np.int64(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Maximum of 2d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amax = MINint32
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
    return np.int32(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Maximum of 2d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amax = MINint64
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
    return np.int64(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "Maximum of 3d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amax = MINint32
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
    return np.int32(amax)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmax_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "Maximum of 3d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amax = MINint64
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
    return np.int64(amax)

cdef dict nanmax_dict = {}
nanmax_dict[(2, int32, 0)] = nanmax_2d_int32_axis0
nanmax_dict[(2, int32, 1)] = nanmax_2d_int32_axis1
nanmax_dict[(2, int64, 0)] = nanmax_2d_int64_axis0
nanmax_dict[(2, int64, 1)] = nanmax_2d_int64_axis1
nanmax_dict[(3, int32, 0)] = nanmax_3d_int32_axis0
nanmax_dict[(3, int32, 1)] = nanmax_3d_int32_axis1
nanmax_dict[(3, int32, 2)] = nanmax_3d_int32_axis2
nanmax_dict[(3, int64, 0)] = nanmax_3d_int64_axis0
nanmax_dict[(3, int64, 1)] = nanmax_3d_int64_axis1
nanmax_dict[(3, int64, 2)] = nanmax_3d_int64_axis2
nanmax_dict[(1, float32, 0)] = nanmax_1d_float32_axisNone
nanmax_dict[(1, float32, None)] = nanmax_1d_float32_axisNone
nanmax_dict[(1, float64, 0)] = nanmax_1d_float64_axisNone
nanmax_dict[(1, float64, None)] = nanmax_1d_float64_axisNone
nanmax_dict[(2, float32, None)] = nanmax_2d_float32_axisNone
nanmax_dict[(2, float64, None)] = nanmax_2d_float64_axisNone
nanmax_dict[(3, float32, None)] = nanmax_3d_float32_axisNone
nanmax_dict[(3, float64, None)] = nanmax_3d_float64_axisNone
nanmax_dict[(2, float32, 0)] = nanmax_2d_float32_axis0
nanmax_dict[(2, float32, 1)] = nanmax_2d_float32_axis1
nanmax_dict[(2, float64, 0)] = nanmax_2d_float64_axis0
nanmax_dict[(2, float64, 1)] = nanmax_2d_float64_axis1
nanmax_dict[(3, float32, 0)] = nanmax_3d_float32_axis0
nanmax_dict[(3, float32, 1)] = nanmax_3d_float32_axis1
nanmax_dict[(3, float32, 2)] = nanmax_3d_float32_axis2
nanmax_dict[(3, float64, 0)] = nanmax_3d_float64_axis0
nanmax_dict[(3, float64, 1)] = nanmax_3d_float64_axis1
nanmax_dict[(3, float64, 2)] = nanmax_3d_float64_axis2
nanmax_dict[(1, int32, 0)] = nanmax_1d_int32_axisNone
nanmax_dict[(1, int32, None)] = nanmax_1d_int32_axisNone
nanmax_dict[(1, int64, 0)] = nanmax_1d_int64_axisNone
nanmax_dict[(1, int64, None)] = nanmax_1d_int64_axisNone
nanmax_dict[(2, int32, None)] = nanmax_2d_int32_axisNone
nanmax_dict[(2, int64, None)] = nanmax_2d_int64_axisNone
nanmax_dict[(3, int32, None)] = nanmax_3d_int32_axisNone
nanmax_dict[(3, int64, None)] = nanmax_3d_int64_axisNone