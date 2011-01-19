"nanmax auto-generated from template"

def nanmax(arr, axis=None):
    """
    Maximum values along specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the maximum is computed. The default (axis=None) is
        to compute the maximum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    See also
    --------
    bottleneck.nanmin: Minimum along specified axis, ignoring NaNs.
    bottleneck.nanargmax: Indices of maximum values along axis, ignoring NaNs. 
    
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
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    cdef int size = PyArray_SIZE(a)
    if size == 0:
        msg = "numpy.nanmax() raises on size=0 input; so Bottleneck does too." 
        raise ValueError, msg
    if (axis < 0) and (axis is not None):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanmax_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanmax_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
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
nanmax_dict[(2, NPY_int32, 0)] = nanmax_2d_int32_axis0
nanmax_dict[(2, NPY_int32, 1)] = nanmax_2d_int32_axis1
nanmax_dict[(2, NPY_int64, 0)] = nanmax_2d_int64_axis0
nanmax_dict[(2, NPY_int64, 1)] = nanmax_2d_int64_axis1
nanmax_dict[(3, NPY_int32, 0)] = nanmax_3d_int32_axis0
nanmax_dict[(3, NPY_int32, 1)] = nanmax_3d_int32_axis1
nanmax_dict[(3, NPY_int32, 2)] = nanmax_3d_int32_axis2
nanmax_dict[(3, NPY_int64, 0)] = nanmax_3d_int64_axis0
nanmax_dict[(3, NPY_int64, 1)] = nanmax_3d_int64_axis1
nanmax_dict[(3, NPY_int64, 2)] = nanmax_3d_int64_axis2
nanmax_dict[(1, NPY_float32, 0)] = nanmax_1d_float32_axisNone
nanmax_dict[(1, NPY_float32, None)] = nanmax_1d_float32_axisNone
nanmax_dict[(1, NPY_float64, 0)] = nanmax_1d_float64_axisNone
nanmax_dict[(1, NPY_float64, None)] = nanmax_1d_float64_axisNone
nanmax_dict[(2, NPY_float32, None)] = nanmax_2d_float32_axisNone
nanmax_dict[(2, NPY_float64, None)] = nanmax_2d_float64_axisNone
nanmax_dict[(3, NPY_float32, None)] = nanmax_3d_float32_axisNone
nanmax_dict[(3, NPY_float64, None)] = nanmax_3d_float64_axisNone
nanmax_dict[(2, NPY_float32, 0)] = nanmax_2d_float32_axis0
nanmax_dict[(2, NPY_float32, 1)] = nanmax_2d_float32_axis1
nanmax_dict[(2, NPY_float64, 0)] = nanmax_2d_float64_axis0
nanmax_dict[(2, NPY_float64, 1)] = nanmax_2d_float64_axis1
nanmax_dict[(3, NPY_float32, 0)] = nanmax_3d_float32_axis0
nanmax_dict[(3, NPY_float32, 1)] = nanmax_3d_float32_axis1
nanmax_dict[(3, NPY_float32, 2)] = nanmax_3d_float32_axis2
nanmax_dict[(3, NPY_float64, 0)] = nanmax_3d_float64_axis0
nanmax_dict[(3, NPY_float64, 1)] = nanmax_3d_float64_axis1
nanmax_dict[(3, NPY_float64, 2)] = nanmax_3d_float64_axis2
nanmax_dict[(1, NPY_int32, 0)] = nanmax_1d_int32_axisNone
nanmax_dict[(1, NPY_int32, None)] = nanmax_1d_int32_axisNone
nanmax_dict[(1, NPY_int64, 0)] = nanmax_1d_int64_axisNone
nanmax_dict[(1, NPY_int64, None)] = nanmax_1d_int64_axisNone
nanmax_dict[(2, NPY_int32, None)] = nanmax_2d_int32_axisNone
nanmax_dict[(2, NPY_int64, None)] = nanmax_2d_int64_axisNone
nanmax_dict[(3, NPY_int32, None)] = nanmax_3d_int32_axisNone
nanmax_dict[(3, NPY_int64, None)] = nanmax_3d_int64_axisNone

cdef dict nanmax_slow_dict = {}
nanmax_slow_dict[0] = nanmax_slow_axis0
nanmax_slow_dict[1] = nanmax_slow_axis1
nanmax_slow_dict[2] = nanmax_slow_axis2
nanmax_slow_dict[3] = nanmax_slow_axis3
nanmax_slow_dict[4] = nanmax_slow_axis4
nanmax_slow_dict[5] = nanmax_slow_axis5
nanmax_slow_dict[6] = nanmax_slow_axis6
nanmax_slow_dict[7] = nanmax_slow_axis7
nanmax_slow_dict[8] = nanmax_slow_axis8
nanmax_slow_dict[9] = nanmax_slow_axis9
nanmax_slow_dict[10] = nanmax_slow_axis10
nanmax_slow_dict[11] = nanmax_slow_axis11
nanmax_slow_dict[12] = nanmax_slow_axis12
nanmax_slow_dict[13] = nanmax_slow_axis13
nanmax_slow_dict[14] = nanmax_slow_axis14
nanmax_slow_dict[15] = nanmax_slow_axis15
nanmax_slow_dict[16] = nanmax_slow_axis16
nanmax_slow_dict[17] = nanmax_slow_axis17
nanmax_slow_dict[18] = nanmax_slow_axis18
nanmax_slow_dict[19] = nanmax_slow_axis19
nanmax_slow_dict[20] = nanmax_slow_axis20
nanmax_slow_dict[21] = nanmax_slow_axis21
nanmax_slow_dict[22] = nanmax_slow_axis22
nanmax_slow_dict[23] = nanmax_slow_axis23
nanmax_slow_dict[24] = nanmax_slow_axis24
nanmax_slow_dict[25] = nanmax_slow_axis25
nanmax_slow_dict[26] = nanmax_slow_axis26
nanmax_slow_dict[27] = nanmax_slow_axis27
nanmax_slow_dict[28] = nanmax_slow_axis28
nanmax_slow_dict[29] = nanmax_slow_axis29
nanmax_slow_dict[30] = nanmax_slow_axis30
nanmax_slow_dict[31] = nanmax_slow_axis31
nanmax_slow_dict[32] = nanmax_slow_axis32
nanmax_slow_dict[None] = nanmax_slow_axisNone

def nanmax_slow_axis0(arr):
    "Unaccelerated (slow) nanmax along axis 0."
    return bn.slow.nanmax(arr, axis=0)

def nanmax_slow_axis1(arr):
    "Unaccelerated (slow) nanmax along axis 1."
    return bn.slow.nanmax(arr, axis=1)

def nanmax_slow_axis2(arr):
    "Unaccelerated (slow) nanmax along axis 2."
    return bn.slow.nanmax(arr, axis=2)

def nanmax_slow_axis3(arr):
    "Unaccelerated (slow) nanmax along axis 3."
    return bn.slow.nanmax(arr, axis=3)

def nanmax_slow_axis4(arr):
    "Unaccelerated (slow) nanmax along axis 4."
    return bn.slow.nanmax(arr, axis=4)

def nanmax_slow_axis5(arr):
    "Unaccelerated (slow) nanmax along axis 5."
    return bn.slow.nanmax(arr, axis=5)

def nanmax_slow_axis6(arr):
    "Unaccelerated (slow) nanmax along axis 6."
    return bn.slow.nanmax(arr, axis=6)

def nanmax_slow_axis7(arr):
    "Unaccelerated (slow) nanmax along axis 7."
    return bn.slow.nanmax(arr, axis=7)

def nanmax_slow_axis8(arr):
    "Unaccelerated (slow) nanmax along axis 8."
    return bn.slow.nanmax(arr, axis=8)

def nanmax_slow_axis9(arr):
    "Unaccelerated (slow) nanmax along axis 9."
    return bn.slow.nanmax(arr, axis=9)

def nanmax_slow_axis10(arr):
    "Unaccelerated (slow) nanmax along axis 10."
    return bn.slow.nanmax(arr, axis=10)

def nanmax_slow_axis11(arr):
    "Unaccelerated (slow) nanmax along axis 11."
    return bn.slow.nanmax(arr, axis=11)

def nanmax_slow_axis12(arr):
    "Unaccelerated (slow) nanmax along axis 12."
    return bn.slow.nanmax(arr, axis=12)

def nanmax_slow_axis13(arr):
    "Unaccelerated (slow) nanmax along axis 13."
    return bn.slow.nanmax(arr, axis=13)

def nanmax_slow_axis14(arr):
    "Unaccelerated (slow) nanmax along axis 14."
    return bn.slow.nanmax(arr, axis=14)

def nanmax_slow_axis15(arr):
    "Unaccelerated (slow) nanmax along axis 15."
    return bn.slow.nanmax(arr, axis=15)

def nanmax_slow_axis16(arr):
    "Unaccelerated (slow) nanmax along axis 16."
    return bn.slow.nanmax(arr, axis=16)

def nanmax_slow_axis17(arr):
    "Unaccelerated (slow) nanmax along axis 17."
    return bn.slow.nanmax(arr, axis=17)

def nanmax_slow_axis18(arr):
    "Unaccelerated (slow) nanmax along axis 18."
    return bn.slow.nanmax(arr, axis=18)

def nanmax_slow_axis19(arr):
    "Unaccelerated (slow) nanmax along axis 19."
    return bn.slow.nanmax(arr, axis=19)

def nanmax_slow_axis20(arr):
    "Unaccelerated (slow) nanmax along axis 20."
    return bn.slow.nanmax(arr, axis=20)

def nanmax_slow_axis21(arr):
    "Unaccelerated (slow) nanmax along axis 21."
    return bn.slow.nanmax(arr, axis=21)

def nanmax_slow_axis22(arr):
    "Unaccelerated (slow) nanmax along axis 22."
    return bn.slow.nanmax(arr, axis=22)

def nanmax_slow_axis23(arr):
    "Unaccelerated (slow) nanmax along axis 23."
    return bn.slow.nanmax(arr, axis=23)

def nanmax_slow_axis24(arr):
    "Unaccelerated (slow) nanmax along axis 24."
    return bn.slow.nanmax(arr, axis=24)

def nanmax_slow_axis25(arr):
    "Unaccelerated (slow) nanmax along axis 25."
    return bn.slow.nanmax(arr, axis=25)

def nanmax_slow_axis26(arr):
    "Unaccelerated (slow) nanmax along axis 26."
    return bn.slow.nanmax(arr, axis=26)

def nanmax_slow_axis27(arr):
    "Unaccelerated (slow) nanmax along axis 27."
    return bn.slow.nanmax(arr, axis=27)

def nanmax_slow_axis28(arr):
    "Unaccelerated (slow) nanmax along axis 28."
    return bn.slow.nanmax(arr, axis=28)

def nanmax_slow_axis29(arr):
    "Unaccelerated (slow) nanmax along axis 29."
    return bn.slow.nanmax(arr, axis=29)

def nanmax_slow_axis30(arr):
    "Unaccelerated (slow) nanmax along axis 30."
    return bn.slow.nanmax(arr, axis=30)

def nanmax_slow_axis31(arr):
    "Unaccelerated (slow) nanmax along axis 31."
    return bn.slow.nanmax(arr, axis=31)

def nanmax_slow_axis32(arr):
    "Unaccelerated (slow) nanmax along axis 32."
    return bn.slow.nanmax(arr, axis=32)

def nanmax_slow_axisNone(arr):
    "Unaccelerated (slow) nanmax along axis None."
    return bn.slow.nanmax(arr, axis=None)
