"nanmin auto-generated from template"

def nanmin(arr, axis=None):
    """
    Minimum values along specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the minimum is computed. The default (axis=None) is
        to compute the minimum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    See also
    --------
    bottleneck.nanmax: Maximum along specified axis, ignoring NaNs.
    bottleneck.nanargmin: Indices of minimum values along axis, ignoring NaNs. 

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
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    cdef int size = PyArray_SIZE(a)
    if size == 0:
        msg = "numpy.nanmin() raises on size=0 input; so Bottleneck does too." 
        raise ValueError, msg
    if (axis < 0) and (axis is not None):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmin_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmin_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Minimum of 2d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        amin = MAXint32
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
        y[i1] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Minimum of 2d array with dtype=int32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        amin = MAXint32
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
        y[i0] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Minimum of 2d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        amin = MAXint64
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
        y[i1] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Minimum of 2d array with dtype=int64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        amin = MAXint64
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
        y[i0] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Minimum of 3d array with dtype=int32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amin = MAXint32
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i1, i2] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Minimum of 3d array with dtype=int32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amin = MAXint32
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i0, i2] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Minimum of 3d array with dtype=int32 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amin = MAXint32
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i0, i1] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Minimum of 3d array with dtype=int64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amin = MAXint64
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i1, i2] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Minimum of 3d array with dtype=int64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amin = MAXint64
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i0, i2] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Minimum of 3d array with dtype=int64 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amin = MAXint64
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
            y[i0, i1] = amin
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Minimum of 1d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amin = MAXfloat32
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan == 0:       
        return np.float32(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Minimum of 1d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amin = MAXfloat64
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            allnan = 0
    if allnan == 0:       
        return np.float64(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Minimum of 2d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amin = MAXfloat32
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
    if allnan == 0:       
        return np.float32(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Minimum of 2d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amin = MAXfloat64
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
    if allnan == 0:       
        return np.float64(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a):
    "Minimum of 3d array with dtype=float32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amin = MAXfloat32
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
    if allnan == 0:       
        return np.float32(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Minimum of 3d array with dtype=float64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amin = MAXfloat64
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
    if allnan == 0:       
        return np.float64(amin)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Minimum of 2d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        amin = MAXfloat32
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[i1] = amin
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Minimum of 2d array with dtype=float32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        amin = MAXfloat32
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[i0] = amin
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Minimum of 2d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        amin = MAXfloat64
        allnan = 1
        for i0 in range(n0):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[i1] = amin
        else:
            y[i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Minimum of 2d array with dtype=float64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        amin = MAXfloat64
        allnan = 1
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan == 0:       
            y[i0] = amin
        else:
            y[i0] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Minimum of 3d array with dtype=float32 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amin = MAXfloat32
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i1, i2] = amin
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Minimum of 3d array with dtype=float32 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amin = MAXfloat32
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i2] = amin
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Minimum of 3d array with dtype=float32 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amin = MAXfloat32
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i1] = amin
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Minimum of 3d array with dtype=float64 along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            amin = MAXfloat64
            allnan = 1
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i1, i2] = amin
            else:
                y[i1, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Minimum of 3d array with dtype=float64 along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            amin = MAXfloat64
            allnan = 1
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i2] = amin
            else:
                y[i0, i2] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Minimum of 3d array with dtype=float64 along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            amin = MAXfloat64
            allnan = 1
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
            if allnan == 0:       
                y[i0, i1] = amin
            else:
                y[i0, i1] = NAN
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Minimum of 1d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amin = MAXint32
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
    return np.int32(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Minimum of 1d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    amin = MAXint64
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
    return np.int64(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Minimum of 2d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amin = MAXint32
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
    return np.int32(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Minimum of 2d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    amin = MAXint64
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
    return np.int64(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "Minimum of 3d array with dtype=int32 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amin = MAXint32
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
    return np.int32(amin)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanmin_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "Minimum of 3d array with dtype=int64 along axis=None ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    amin = MAXint64
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
    return np.int64(amin)

cdef dict nanmin_dict = {}
nanmin_dict[(2, NPY_int32, 0)] = nanmin_2d_int32_axis0
nanmin_dict[(2, NPY_int32, 1)] = nanmin_2d_int32_axis1
nanmin_dict[(2, NPY_int64, 0)] = nanmin_2d_int64_axis0
nanmin_dict[(2, NPY_int64, 1)] = nanmin_2d_int64_axis1
nanmin_dict[(3, NPY_int32, 0)] = nanmin_3d_int32_axis0
nanmin_dict[(3, NPY_int32, 1)] = nanmin_3d_int32_axis1
nanmin_dict[(3, NPY_int32, 2)] = nanmin_3d_int32_axis2
nanmin_dict[(3, NPY_int64, 0)] = nanmin_3d_int64_axis0
nanmin_dict[(3, NPY_int64, 1)] = nanmin_3d_int64_axis1
nanmin_dict[(3, NPY_int64, 2)] = nanmin_3d_int64_axis2
nanmin_dict[(1, NPY_float32, 0)] = nanmin_1d_float32_axisNone
nanmin_dict[(1, NPY_float32, None)] = nanmin_1d_float32_axisNone
nanmin_dict[(1, NPY_float64, 0)] = nanmin_1d_float64_axisNone
nanmin_dict[(1, NPY_float64, None)] = nanmin_1d_float64_axisNone
nanmin_dict[(2, NPY_float32, None)] = nanmin_2d_float32_axisNone
nanmin_dict[(2, NPY_float64, None)] = nanmin_2d_float64_axisNone
nanmin_dict[(3, NPY_float32, None)] = nanmin_3d_float32_axisNone
nanmin_dict[(3, NPY_float64, None)] = nanmin_3d_float64_axisNone
nanmin_dict[(2, NPY_float32, 0)] = nanmin_2d_float32_axis0
nanmin_dict[(2, NPY_float32, 1)] = nanmin_2d_float32_axis1
nanmin_dict[(2, NPY_float64, 0)] = nanmin_2d_float64_axis0
nanmin_dict[(2, NPY_float64, 1)] = nanmin_2d_float64_axis1
nanmin_dict[(3, NPY_float32, 0)] = nanmin_3d_float32_axis0
nanmin_dict[(3, NPY_float32, 1)] = nanmin_3d_float32_axis1
nanmin_dict[(3, NPY_float32, 2)] = nanmin_3d_float32_axis2
nanmin_dict[(3, NPY_float64, 0)] = nanmin_3d_float64_axis0
nanmin_dict[(3, NPY_float64, 1)] = nanmin_3d_float64_axis1
nanmin_dict[(3, NPY_float64, 2)] = nanmin_3d_float64_axis2
nanmin_dict[(1, NPY_int32, 0)] = nanmin_1d_int32_axisNone
nanmin_dict[(1, NPY_int32, None)] = nanmin_1d_int32_axisNone
nanmin_dict[(1, NPY_int64, 0)] = nanmin_1d_int64_axisNone
nanmin_dict[(1, NPY_int64, None)] = nanmin_1d_int64_axisNone
nanmin_dict[(2, NPY_int32, None)] = nanmin_2d_int32_axisNone
nanmin_dict[(2, NPY_int64, None)] = nanmin_2d_int64_axisNone
nanmin_dict[(3, NPY_int32, None)] = nanmin_3d_int32_axisNone
nanmin_dict[(3, NPY_int64, None)] = nanmin_3d_int64_axisNone

cdef dict nanmin_slow_dict = {}
nanmin_slow_dict[0] = nanmin_slow_axis0
nanmin_slow_dict[1] = nanmin_slow_axis1
nanmin_slow_dict[2] = nanmin_slow_axis2
nanmin_slow_dict[3] = nanmin_slow_axis3
nanmin_slow_dict[4] = nanmin_slow_axis4
nanmin_slow_dict[5] = nanmin_slow_axis5
nanmin_slow_dict[6] = nanmin_slow_axis6
nanmin_slow_dict[7] = nanmin_slow_axis7
nanmin_slow_dict[8] = nanmin_slow_axis8
nanmin_slow_dict[9] = nanmin_slow_axis9
nanmin_slow_dict[10] = nanmin_slow_axis10
nanmin_slow_dict[11] = nanmin_slow_axis11
nanmin_slow_dict[12] = nanmin_slow_axis12
nanmin_slow_dict[13] = nanmin_slow_axis13
nanmin_slow_dict[14] = nanmin_slow_axis14
nanmin_slow_dict[15] = nanmin_slow_axis15
nanmin_slow_dict[16] = nanmin_slow_axis16
nanmin_slow_dict[17] = nanmin_slow_axis17
nanmin_slow_dict[18] = nanmin_slow_axis18
nanmin_slow_dict[19] = nanmin_slow_axis19
nanmin_slow_dict[20] = nanmin_slow_axis20
nanmin_slow_dict[21] = nanmin_slow_axis21
nanmin_slow_dict[22] = nanmin_slow_axis22
nanmin_slow_dict[23] = nanmin_slow_axis23
nanmin_slow_dict[24] = nanmin_slow_axis24
nanmin_slow_dict[25] = nanmin_slow_axis25
nanmin_slow_dict[26] = nanmin_slow_axis26
nanmin_slow_dict[27] = nanmin_slow_axis27
nanmin_slow_dict[28] = nanmin_slow_axis28
nanmin_slow_dict[29] = nanmin_slow_axis29
nanmin_slow_dict[30] = nanmin_slow_axis30
nanmin_slow_dict[31] = nanmin_slow_axis31
nanmin_slow_dict[32] = nanmin_slow_axis32
nanmin_slow_dict[None] = nanmin_slow_axisNone

def nanmin_slow_axis0(arr):
    "Unaccelerated (slow) nanmin along axis 0."
    return bn.slow.nanmin(arr, axis=0)

def nanmin_slow_axis1(arr):
    "Unaccelerated (slow) nanmin along axis 1."
    return bn.slow.nanmin(arr, axis=1)

def nanmin_slow_axis2(arr):
    "Unaccelerated (slow) nanmin along axis 2."
    return bn.slow.nanmin(arr, axis=2)

def nanmin_slow_axis3(arr):
    "Unaccelerated (slow) nanmin along axis 3."
    return bn.slow.nanmin(arr, axis=3)

def nanmin_slow_axis4(arr):
    "Unaccelerated (slow) nanmin along axis 4."
    return bn.slow.nanmin(arr, axis=4)

def nanmin_slow_axis5(arr):
    "Unaccelerated (slow) nanmin along axis 5."
    return bn.slow.nanmin(arr, axis=5)

def nanmin_slow_axis6(arr):
    "Unaccelerated (slow) nanmin along axis 6."
    return bn.slow.nanmin(arr, axis=6)

def nanmin_slow_axis7(arr):
    "Unaccelerated (slow) nanmin along axis 7."
    return bn.slow.nanmin(arr, axis=7)

def nanmin_slow_axis8(arr):
    "Unaccelerated (slow) nanmin along axis 8."
    return bn.slow.nanmin(arr, axis=8)

def nanmin_slow_axis9(arr):
    "Unaccelerated (slow) nanmin along axis 9."
    return bn.slow.nanmin(arr, axis=9)

def nanmin_slow_axis10(arr):
    "Unaccelerated (slow) nanmin along axis 10."
    return bn.slow.nanmin(arr, axis=10)

def nanmin_slow_axis11(arr):
    "Unaccelerated (slow) nanmin along axis 11."
    return bn.slow.nanmin(arr, axis=11)

def nanmin_slow_axis12(arr):
    "Unaccelerated (slow) nanmin along axis 12."
    return bn.slow.nanmin(arr, axis=12)

def nanmin_slow_axis13(arr):
    "Unaccelerated (slow) nanmin along axis 13."
    return bn.slow.nanmin(arr, axis=13)

def nanmin_slow_axis14(arr):
    "Unaccelerated (slow) nanmin along axis 14."
    return bn.slow.nanmin(arr, axis=14)

def nanmin_slow_axis15(arr):
    "Unaccelerated (slow) nanmin along axis 15."
    return bn.slow.nanmin(arr, axis=15)

def nanmin_slow_axis16(arr):
    "Unaccelerated (slow) nanmin along axis 16."
    return bn.slow.nanmin(arr, axis=16)

def nanmin_slow_axis17(arr):
    "Unaccelerated (slow) nanmin along axis 17."
    return bn.slow.nanmin(arr, axis=17)

def nanmin_slow_axis18(arr):
    "Unaccelerated (slow) nanmin along axis 18."
    return bn.slow.nanmin(arr, axis=18)

def nanmin_slow_axis19(arr):
    "Unaccelerated (slow) nanmin along axis 19."
    return bn.slow.nanmin(arr, axis=19)

def nanmin_slow_axis20(arr):
    "Unaccelerated (slow) nanmin along axis 20."
    return bn.slow.nanmin(arr, axis=20)

def nanmin_slow_axis21(arr):
    "Unaccelerated (slow) nanmin along axis 21."
    return bn.slow.nanmin(arr, axis=21)

def nanmin_slow_axis22(arr):
    "Unaccelerated (slow) nanmin along axis 22."
    return bn.slow.nanmin(arr, axis=22)

def nanmin_slow_axis23(arr):
    "Unaccelerated (slow) nanmin along axis 23."
    return bn.slow.nanmin(arr, axis=23)

def nanmin_slow_axis24(arr):
    "Unaccelerated (slow) nanmin along axis 24."
    return bn.slow.nanmin(arr, axis=24)

def nanmin_slow_axis25(arr):
    "Unaccelerated (slow) nanmin along axis 25."
    return bn.slow.nanmin(arr, axis=25)

def nanmin_slow_axis26(arr):
    "Unaccelerated (slow) nanmin along axis 26."
    return bn.slow.nanmin(arr, axis=26)

def nanmin_slow_axis27(arr):
    "Unaccelerated (slow) nanmin along axis 27."
    return bn.slow.nanmin(arr, axis=27)

def nanmin_slow_axis28(arr):
    "Unaccelerated (slow) nanmin along axis 28."
    return bn.slow.nanmin(arr, axis=28)

def nanmin_slow_axis29(arr):
    "Unaccelerated (slow) nanmin along axis 29."
    return bn.slow.nanmin(arr, axis=29)

def nanmin_slow_axis30(arr):
    "Unaccelerated (slow) nanmin along axis 30."
    return bn.slow.nanmin(arr, axis=30)

def nanmin_slow_axis31(arr):
    "Unaccelerated (slow) nanmin along axis 31."
    return bn.slow.nanmin(arr, axis=31)

def nanmin_slow_axis32(arr):
    "Unaccelerated (slow) nanmin along axis 32."
    return bn.slow.nanmin(arr, axis=32)

def nanmin_slow_axisNone(arr):
    "Unaccelerated (slow) nanmin along axis None."
    return bn.slow.nanmin(arr, axis=None)
