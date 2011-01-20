"nanargmin auto-generated from template"

CANNOTCONVERT = "Bottleneck copies NumPy bahavior: "
CANNOTCONVERT += "'cannot convert float NaN to integer'"

def nanargmin(arr, axis=None):
    """
    Indices of the minimum values along an axis, ignoring NaNs.
    
    Parameters
    ----------
    a : array_like
        Input data.
    axis : {int, None}, optional
        Axis along which to operate. By default (axis=None) flattened input
        is used.
   
    See also
    --------
    bottleneck.nanargmax: Indices of the maximum values along an axis.
    bottleneck.nanmin: Minimum values along specified axis, ignoring NaNs.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.
    
    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> bn.nanargmin(a)
    2
    >>> a.flat[1]
    2.0
    >>> bn.nanargmax(a, axis=0)
    array([1, 1])
    >>> bn.nanargmax(a, axis=1)
    array([1, 0])
    
    """
    func, arr = nanargmin_selector(arr, axis)
    return func(arr)

def nanargmin_selector(arr, axis):
    """
    Return nanargmin function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanargmin() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to find the indices of the minimum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the indices are found. The default (axis=None) is to
        find the index of the minimum value in the flattened array.
    
    Returns
    -------
    func : function
        The nanargmin function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the nanargmin of `arr` along
    axis=0:

    >>> func, a = bn.func.nanargmin_selector(arr, axis=0)
    >>> func
    <built-in function nanargmin_1d_float64_axis0> 
    
    Use the returned function and array to determine the maximum:
    
    >>> func(a)
    0

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
        msg = "numpy.nanargmin() raises on size=0; so Bottleneck does too." 
        raise ValueError, msg
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = a.ravel()
        axis = 0
        ndim = 1
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nanargmin_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanargmin_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "Index of max of 1d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amin = MAXint32
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            idx = i0
    return np.int64(idx)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "Index of max of 1d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amin = MAXint64
    for i0 in range(n0):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            idx = i0
    return np.int64(idx)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Index of max of 2d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        amin = MAXint32
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                idx = i0
        y[i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Index of max of 2d, int32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        amin = MAXint32
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                idx = i1
        y[i0] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Index of max of 2d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        amin = MAXint64
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                idx = i0
        y[i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Index of max of 2d, int64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        amin = MAXint64
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                idx = i1
        y[i0] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXint32
            for i0 in range(n0 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i0
            y[i1, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXint32
            for i1 in range(n1 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i1
            y[i0, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i1 in range(n1 - 1, -1, -1):
            amin = MAXint32
            for i2 in range(n2 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i2
            y[i0, i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXint64
            for i0 in range(n0 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i0
            y[i1, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXint64
            for i1 in range(n1 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i1
            y[i0, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i1 in range(n1 - 1, -1, -1):
            amin = MAXint64
            for i2 in range(n2 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    idx = i2
            y[i0, i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Index of max of 1d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amin = MAXfloat32
    for i0 in range(n0 - 1, -1, -1):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            allnan = 0
            idx = i0
    if allnan == 0:       
        return np.int64(idx)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Index of max of 1d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amin = MAXfloat64
    for i0 in range(n0 - 1, -1, -1):
        ai = a[i0]
        if ai <= amin:
            amin = ai
            allnan = 0
            idx = i0
    if allnan == 0:       
        return np.int64(idx)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Index of max of 2d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        amin = MAXfloat32
        allnan = 1
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
                idx = i0
        if allnan == 0:       
            y[i1] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Index of max of 2d, float32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        amin = MAXfloat32
        allnan = 1
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
                idx = i1
        if allnan == 0:       
            y[i0] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Index of max of 2d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        amin = MAXfloat64
        allnan = 1
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
                idx = i0
        if allnan == 0:       
            y[i1] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Index of max of 2d, float64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        amin = MAXfloat64
        allnan = 1
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai <= amin:
                amin = ai
                allnan = 0
                idx = i1
        if allnan == 0:       
            y[i0] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXfloat32
            allnan = 1
            for i0 in range(n0 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i0
            if allnan == 0:       
                y[i1, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXfloat32
            allnan = 1
            for i1 in range(n1 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i1
            if allnan == 0:       
                y[i0, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i1 in range(n1 - 1, -1, -1):
            amin = MAXfloat32
            allnan = 1
            for i2 in range(n2 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i2
            if allnan == 0:       
                y[i0, i1] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXfloat64
            allnan = 1
            for i0 in range(n0 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i0
            if allnan == 0:       
                y[i1, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i2 in range(n2 - 1, -1, -1):
            amin = MAXfloat64
            allnan = 1
            for i1 in range(n1 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i1
            if allnan == 0:       
                y[i0, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmin_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amin, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0 - 1, -1, -1):
        for i1 in range(n1 - 1, -1, -1):
            amin = MAXfloat64
            allnan = 1
            for i2 in range(n2 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai <= amin:
                    amin = ai
                    allnan = 0
                    idx = i2
            if allnan == 0:       
                y[i0, i1] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

cdef dict nanargmin_dict = {}
nanargmin_dict[(1, NPY_int32, 0)] = nanargmin_1d_int32_axis0
nanargmin_dict[(1, NPY_int64, 0)] = nanargmin_1d_int64_axis0
nanargmin_dict[(2, NPY_int32, 0)] = nanargmin_2d_int32_axis0
nanargmin_dict[(2, NPY_int32, 1)] = nanargmin_2d_int32_axis1
nanargmin_dict[(2, NPY_int64, 0)] = nanargmin_2d_int64_axis0
nanargmin_dict[(2, NPY_int64, 1)] = nanargmin_2d_int64_axis1
nanargmin_dict[(3, NPY_int32, 0)] = nanargmin_3d_int32_axis0
nanargmin_dict[(3, NPY_int32, 1)] = nanargmin_3d_int32_axis1
nanargmin_dict[(3, NPY_int32, 2)] = nanargmin_3d_int32_axis2
nanargmin_dict[(3, NPY_int64, 0)] = nanargmin_3d_int64_axis0
nanargmin_dict[(3, NPY_int64, 1)] = nanargmin_3d_int64_axis1
nanargmin_dict[(3, NPY_int64, 2)] = nanargmin_3d_int64_axis2
nanargmin_dict[(1, NPY_float32, 0)] = nanargmin_1d_float32_axis0
nanargmin_dict[(1, NPY_float64, 0)] = nanargmin_1d_float64_axis0
nanargmin_dict[(2, NPY_float32, 0)] = nanargmin_2d_float32_axis0
nanargmin_dict[(2, NPY_float32, 1)] = nanargmin_2d_float32_axis1
nanargmin_dict[(2, NPY_float64, 0)] = nanargmin_2d_float64_axis0
nanargmin_dict[(2, NPY_float64, 1)] = nanargmin_2d_float64_axis1
nanargmin_dict[(3, NPY_float32, 0)] = nanargmin_3d_float32_axis0
nanargmin_dict[(3, NPY_float32, 1)] = nanargmin_3d_float32_axis1
nanargmin_dict[(3, NPY_float32, 2)] = nanargmin_3d_float32_axis2
nanargmin_dict[(3, NPY_float64, 0)] = nanargmin_3d_float64_axis0
nanargmin_dict[(3, NPY_float64, 1)] = nanargmin_3d_float64_axis1
nanargmin_dict[(3, NPY_float64, 2)] = nanargmin_3d_float64_axis2

cdef dict nanargmin_slow_dict = {}
nanargmin_slow_dict[0] = nanargmin_slow_axis0
nanargmin_slow_dict[1] = nanargmin_slow_axis1
nanargmin_slow_dict[2] = nanargmin_slow_axis2
nanargmin_slow_dict[3] = nanargmin_slow_axis3
nanargmin_slow_dict[4] = nanargmin_slow_axis4
nanargmin_slow_dict[5] = nanargmin_slow_axis5
nanargmin_slow_dict[6] = nanargmin_slow_axis6
nanargmin_slow_dict[7] = nanargmin_slow_axis7
nanargmin_slow_dict[8] = nanargmin_slow_axis8
nanargmin_slow_dict[9] = nanargmin_slow_axis9
nanargmin_slow_dict[10] = nanargmin_slow_axis10
nanargmin_slow_dict[11] = nanargmin_slow_axis11
nanargmin_slow_dict[12] = nanargmin_slow_axis12
nanargmin_slow_dict[13] = nanargmin_slow_axis13
nanargmin_slow_dict[14] = nanargmin_slow_axis14
nanargmin_slow_dict[15] = nanargmin_slow_axis15
nanargmin_slow_dict[16] = nanargmin_slow_axis16
nanargmin_slow_dict[17] = nanargmin_slow_axis17
nanargmin_slow_dict[18] = nanargmin_slow_axis18
nanargmin_slow_dict[19] = nanargmin_slow_axis19
nanargmin_slow_dict[20] = nanargmin_slow_axis20
nanargmin_slow_dict[21] = nanargmin_slow_axis21
nanargmin_slow_dict[22] = nanargmin_slow_axis22
nanargmin_slow_dict[23] = nanargmin_slow_axis23
nanargmin_slow_dict[24] = nanargmin_slow_axis24
nanargmin_slow_dict[25] = nanargmin_slow_axis25
nanargmin_slow_dict[26] = nanargmin_slow_axis26
nanargmin_slow_dict[27] = nanargmin_slow_axis27
nanargmin_slow_dict[28] = nanargmin_slow_axis28
nanargmin_slow_dict[29] = nanargmin_slow_axis29
nanargmin_slow_dict[30] = nanargmin_slow_axis30
nanargmin_slow_dict[31] = nanargmin_slow_axis31
nanargmin_slow_dict[32] = nanargmin_slow_axis32
nanargmin_slow_dict[None] = nanargmin_slow_axisNone

def nanargmin_slow_axis0(arr):
    "Unaccelerated (slow) nanargmin along axis 0."
    return bn.slow.nanargmin(arr, axis=0)

def nanargmin_slow_axis1(arr):
    "Unaccelerated (slow) nanargmin along axis 1."
    return bn.slow.nanargmin(arr, axis=1)

def nanargmin_slow_axis2(arr):
    "Unaccelerated (slow) nanargmin along axis 2."
    return bn.slow.nanargmin(arr, axis=2)

def nanargmin_slow_axis3(arr):
    "Unaccelerated (slow) nanargmin along axis 3."
    return bn.slow.nanargmin(arr, axis=3)

def nanargmin_slow_axis4(arr):
    "Unaccelerated (slow) nanargmin along axis 4."
    return bn.slow.nanargmin(arr, axis=4)

def nanargmin_slow_axis5(arr):
    "Unaccelerated (slow) nanargmin along axis 5."
    return bn.slow.nanargmin(arr, axis=5)

def nanargmin_slow_axis6(arr):
    "Unaccelerated (slow) nanargmin along axis 6."
    return bn.slow.nanargmin(arr, axis=6)

def nanargmin_slow_axis7(arr):
    "Unaccelerated (slow) nanargmin along axis 7."
    return bn.slow.nanargmin(arr, axis=7)

def nanargmin_slow_axis8(arr):
    "Unaccelerated (slow) nanargmin along axis 8."
    return bn.slow.nanargmin(arr, axis=8)

def nanargmin_slow_axis9(arr):
    "Unaccelerated (slow) nanargmin along axis 9."
    return bn.slow.nanargmin(arr, axis=9)

def nanargmin_slow_axis10(arr):
    "Unaccelerated (slow) nanargmin along axis 10."
    return bn.slow.nanargmin(arr, axis=10)

def nanargmin_slow_axis11(arr):
    "Unaccelerated (slow) nanargmin along axis 11."
    return bn.slow.nanargmin(arr, axis=11)

def nanargmin_slow_axis12(arr):
    "Unaccelerated (slow) nanargmin along axis 12."
    return bn.slow.nanargmin(arr, axis=12)

def nanargmin_slow_axis13(arr):
    "Unaccelerated (slow) nanargmin along axis 13."
    return bn.slow.nanargmin(arr, axis=13)

def nanargmin_slow_axis14(arr):
    "Unaccelerated (slow) nanargmin along axis 14."
    return bn.slow.nanargmin(arr, axis=14)

def nanargmin_slow_axis15(arr):
    "Unaccelerated (slow) nanargmin along axis 15."
    return bn.slow.nanargmin(arr, axis=15)

def nanargmin_slow_axis16(arr):
    "Unaccelerated (slow) nanargmin along axis 16."
    return bn.slow.nanargmin(arr, axis=16)

def nanargmin_slow_axis17(arr):
    "Unaccelerated (slow) nanargmin along axis 17."
    return bn.slow.nanargmin(arr, axis=17)

def nanargmin_slow_axis18(arr):
    "Unaccelerated (slow) nanargmin along axis 18."
    return bn.slow.nanargmin(arr, axis=18)

def nanargmin_slow_axis19(arr):
    "Unaccelerated (slow) nanargmin along axis 19."
    return bn.slow.nanargmin(arr, axis=19)

def nanargmin_slow_axis20(arr):
    "Unaccelerated (slow) nanargmin along axis 20."
    return bn.slow.nanargmin(arr, axis=20)

def nanargmin_slow_axis21(arr):
    "Unaccelerated (slow) nanargmin along axis 21."
    return bn.slow.nanargmin(arr, axis=21)

def nanargmin_slow_axis22(arr):
    "Unaccelerated (slow) nanargmin along axis 22."
    return bn.slow.nanargmin(arr, axis=22)

def nanargmin_slow_axis23(arr):
    "Unaccelerated (slow) nanargmin along axis 23."
    return bn.slow.nanargmin(arr, axis=23)

def nanargmin_slow_axis24(arr):
    "Unaccelerated (slow) nanargmin along axis 24."
    return bn.slow.nanargmin(arr, axis=24)

def nanargmin_slow_axis25(arr):
    "Unaccelerated (slow) nanargmin along axis 25."
    return bn.slow.nanargmin(arr, axis=25)

def nanargmin_slow_axis26(arr):
    "Unaccelerated (slow) nanargmin along axis 26."
    return bn.slow.nanargmin(arr, axis=26)

def nanargmin_slow_axis27(arr):
    "Unaccelerated (slow) nanargmin along axis 27."
    return bn.slow.nanargmin(arr, axis=27)

def nanargmin_slow_axis28(arr):
    "Unaccelerated (slow) nanargmin along axis 28."
    return bn.slow.nanargmin(arr, axis=28)

def nanargmin_slow_axis29(arr):
    "Unaccelerated (slow) nanargmin along axis 29."
    return bn.slow.nanargmin(arr, axis=29)

def nanargmin_slow_axis30(arr):
    "Unaccelerated (slow) nanargmin along axis 30."
    return bn.slow.nanargmin(arr, axis=30)

def nanargmin_slow_axis31(arr):
    "Unaccelerated (slow) nanargmin along axis 31."
    return bn.slow.nanargmin(arr, axis=31)

def nanargmin_slow_axis32(arr):
    "Unaccelerated (slow) nanargmin along axis 32."
    return bn.slow.nanargmin(arr, axis=32)

def nanargmin_slow_axisNone(arr):
    "Unaccelerated (slow) nanargmin along axis None."
    return bn.slow.nanargmin(arr, axis=None)
