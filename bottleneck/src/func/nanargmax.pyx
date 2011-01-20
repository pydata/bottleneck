"nanargmax auto-generated from template"

CANNOTCONVERT = "Bottleneck copies NumPy bahavior: "
CANNOTCONVERT += "'cannot convert float NaN to integer'"

def nanargmax(arr, axis=None):
    """
    Indices of the maximum values along an axis, ignoring NaNs.
    
    Parameters
    ----------
    a : array_like
        Input data.
    axis : {int, None}, optional
        Axis along which to operate. By default (axis=None) flattened input
        is used.
   
    See also
    --------
    bottleneck.nanargmin: Indices of the minimum values along an axis.
    bottleneck.nanmax: Maximum values along specified axis, ignoring NaNs.

    Returns
    -------
    index_array : ndarray
        An array of indices or a single index value.
    
    Examples
    --------
    >>> a = np.array([[np.nan, 4], [2, 3]])
    >>> bn.nanargmax(a)
    1
    >>> a.flat[1]
    4.0
    >>> bn.nanargmax(a, axis=0)
    array([1, 0])
    >>> bn.nanargmax(a, axis=1)
    array([1, 1])
    
    """
    func, arr = nanargmax_selector(arr, axis)
    return func(arr)

def nanargmax_selector(arr, axis):
    """
    Return nanargmax function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.nanargmax() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to find the indices of the maximum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the indices are found. The default (axis=None) is to
        find the index of the maximum value in the flattened array.
    
    Returns
    -------
    func : function
        The nanargmax function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the nanargmax of `arr` along
    axis=0:

    >>> func, a = bn.func.nanargmax_selector(arr, axis=0)
    >>> func
    <built-in function nanargmax_1d_float64_axis0> 
    
    Use the returned function and array to determine the maximum:
    
    >>> func(a)
    2

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
        msg = "numpy.nanargmax() raises on size=0; so Bottleneck does too." 
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
        func = nanargmax_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = nanargmax_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "Index of max of 1d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amax = MINint32
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            idx = i0
    return np.int64(idx)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "Index of max of 1d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amax = MINint64
    for i0 in range(n0):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            idx = i0
    return np.int64(idx)

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Index of max of 2d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
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
        amax = MINint32
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                idx = i0
        y[i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Index of max of 2d, int32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
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
        amax = MINint32
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                idx = i1
        y[i0] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Index of max of 2d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
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
        amax = MINint64
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                idx = i0
        y[i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Index of max of 2d, int64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
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
        amax = MINint64
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                idx = i1
        y[i0] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
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
            amax = MINint32
            for i0 in range(n0 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i0
            y[i1, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
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
            amax = MINint32
            for i1 in range(n1 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i1
            y[i0, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Index of max of 3d, int32 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int32_t amax, ai
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
            amax = MINint32
            for i2 in range(n2 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i2
            y[i0, i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
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
            amax = MINint64
            for i0 in range(n0 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i0
            y[i1, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
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
            amax = MINint64
            for i1 in range(n1 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i1
            y[i0, i2] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Index of max of 3d, int64 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.int64_t amax, ai
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
            amax = MINint64
            for i2 in range(n2 - 1, - 1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    idx = i2
            y[i0, i1] = idx
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Index of max of 1d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amax = MINfloat32
    for i0 in range(n0 - 1, -1, -1):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            allnan = 0
            idx = i0
    if allnan == 0:       
        return np.int64(idx)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Index of max of 1d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
    cdef Py_ssize_t idx = 0
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    amax = MINfloat64
    for i0 in range(n0 - 1, -1, -1):
        ai = a[i0]
        if ai >= amax:
            amax = ai
            allnan = 0
            idx = i0
    if allnan == 0:       
        return np.int64(idx)
    else:
        return NAN

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Index of max of 2d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
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
        amax = MINfloat32
        allnan = 1
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = i0
        if allnan == 0:       
            y[i1] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Index of max of 2d, float32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
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
        amax = MINfloat32
        allnan = 1
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = i1
        if allnan == 0:       
            y[i0] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Index of max of 2d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
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
        amax = MINfloat64
        allnan = 1
        for i0 in range(n0 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = i0
        if allnan == 0:       
            y[i1] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Index of max of 2d, float64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
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
        amax = MINfloat64
        allnan = 1
        for i1 in range(n1 - 1, -1, -1):
            ai = a[i0, i1]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = i1
        if allnan == 0:       
            y[i0] = idx
        else:
            raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
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
            amax = MINfloat32
            allnan = 1
            for i0 in range(n0 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i0
            if allnan == 0:       
                y[i1, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
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
            amax = MINfloat32
            allnan = 1
            for i1 in range(n1 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i1
            if allnan == 0:       
                y[i0, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Index of max of 3d, float32 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float32_t amax, ai
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
            amax = MINfloat32
            allnan = 1
            for i2 in range(n2 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i2
            if allnan == 0:       
                y[i0, i1] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=0 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
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
            amax = MINfloat64
            allnan = 1
            for i0 in range(n0 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i0
            if allnan == 0:       
                y[i1, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=1 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
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
            amax = MINfloat64
            allnan = 1
            for i1 in range(n1 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i1
            if allnan == 0:       
                y[i0, i2] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def nanargmax_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Index of max of 3d, float64 array along axis=2 ignoring NaNs."
    cdef int allnan = 1
    cdef np.float64_t amax, ai
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
            amax = MINfloat64
            allnan = 1
            for i2 in range(n2 - 1, -1, -1):
                ai = a[i0, i1, i2]
                if ai >= amax:
                    amax = ai
                    allnan = 0
                    idx = i2
            if allnan == 0:       
                y[i0, i1] = idx
            else:
                raise ValueError(CANNOTCONVERT)
    return y

cdef dict nanargmax_dict = {}
nanargmax_dict[(1, NPY_int32, 0)] = nanargmax_1d_int32_axis0
nanargmax_dict[(1, NPY_int64, 0)] = nanargmax_1d_int64_axis0
nanargmax_dict[(2, NPY_int32, 0)] = nanargmax_2d_int32_axis0
nanargmax_dict[(2, NPY_int32, 1)] = nanargmax_2d_int32_axis1
nanargmax_dict[(2, NPY_int64, 0)] = nanargmax_2d_int64_axis0
nanargmax_dict[(2, NPY_int64, 1)] = nanargmax_2d_int64_axis1
nanargmax_dict[(3, NPY_int32, 0)] = nanargmax_3d_int32_axis0
nanargmax_dict[(3, NPY_int32, 1)] = nanargmax_3d_int32_axis1
nanargmax_dict[(3, NPY_int32, 2)] = nanargmax_3d_int32_axis2
nanargmax_dict[(3, NPY_int64, 0)] = nanargmax_3d_int64_axis0
nanargmax_dict[(3, NPY_int64, 1)] = nanargmax_3d_int64_axis1
nanargmax_dict[(3, NPY_int64, 2)] = nanargmax_3d_int64_axis2
nanargmax_dict[(1, NPY_float32, 0)] = nanargmax_1d_float32_axis0
nanargmax_dict[(1, NPY_float64, 0)] = nanargmax_1d_float64_axis0
nanargmax_dict[(2, NPY_float32, 0)] = nanargmax_2d_float32_axis0
nanargmax_dict[(2, NPY_float32, 1)] = nanargmax_2d_float32_axis1
nanargmax_dict[(2, NPY_float64, 0)] = nanargmax_2d_float64_axis0
nanargmax_dict[(2, NPY_float64, 1)] = nanargmax_2d_float64_axis1
nanargmax_dict[(3, NPY_float32, 0)] = nanargmax_3d_float32_axis0
nanargmax_dict[(3, NPY_float32, 1)] = nanargmax_3d_float32_axis1
nanargmax_dict[(3, NPY_float32, 2)] = nanargmax_3d_float32_axis2
nanargmax_dict[(3, NPY_float64, 0)] = nanargmax_3d_float64_axis0
nanargmax_dict[(3, NPY_float64, 1)] = nanargmax_3d_float64_axis1
nanargmax_dict[(3, NPY_float64, 2)] = nanargmax_3d_float64_axis2

cdef dict nanargmax_slow_dict = {}
nanargmax_slow_dict[0] = nanargmax_slow_axis0
nanargmax_slow_dict[1] = nanargmax_slow_axis1
nanargmax_slow_dict[2] = nanargmax_slow_axis2
nanargmax_slow_dict[3] = nanargmax_slow_axis3
nanargmax_slow_dict[4] = nanargmax_slow_axis4
nanargmax_slow_dict[5] = nanargmax_slow_axis5
nanargmax_slow_dict[6] = nanargmax_slow_axis6
nanargmax_slow_dict[7] = nanargmax_slow_axis7
nanargmax_slow_dict[8] = nanargmax_slow_axis8
nanargmax_slow_dict[9] = nanargmax_slow_axis9
nanargmax_slow_dict[10] = nanargmax_slow_axis10
nanargmax_slow_dict[11] = nanargmax_slow_axis11
nanargmax_slow_dict[12] = nanargmax_slow_axis12
nanargmax_slow_dict[13] = nanargmax_slow_axis13
nanargmax_slow_dict[14] = nanargmax_slow_axis14
nanargmax_slow_dict[15] = nanargmax_slow_axis15
nanargmax_slow_dict[16] = nanargmax_slow_axis16
nanargmax_slow_dict[17] = nanargmax_slow_axis17
nanargmax_slow_dict[18] = nanargmax_slow_axis18
nanargmax_slow_dict[19] = nanargmax_slow_axis19
nanargmax_slow_dict[20] = nanargmax_slow_axis20
nanargmax_slow_dict[21] = nanargmax_slow_axis21
nanargmax_slow_dict[22] = nanargmax_slow_axis22
nanargmax_slow_dict[23] = nanargmax_slow_axis23
nanargmax_slow_dict[24] = nanargmax_slow_axis24
nanargmax_slow_dict[25] = nanargmax_slow_axis25
nanargmax_slow_dict[26] = nanargmax_slow_axis26
nanargmax_slow_dict[27] = nanargmax_slow_axis27
nanargmax_slow_dict[28] = nanargmax_slow_axis28
nanargmax_slow_dict[29] = nanargmax_slow_axis29
nanargmax_slow_dict[30] = nanargmax_slow_axis30
nanargmax_slow_dict[31] = nanargmax_slow_axis31
nanargmax_slow_dict[32] = nanargmax_slow_axis32
nanargmax_slow_dict[None] = nanargmax_slow_axisNone

def nanargmax_slow_axis0(arr):
    "Unaccelerated (slow) nanargmax along axis 0."
    return bn.slow.nanargmax(arr, axis=0)

def nanargmax_slow_axis1(arr):
    "Unaccelerated (slow) nanargmax along axis 1."
    return bn.slow.nanargmax(arr, axis=1)

def nanargmax_slow_axis2(arr):
    "Unaccelerated (slow) nanargmax along axis 2."
    return bn.slow.nanargmax(arr, axis=2)

def nanargmax_slow_axis3(arr):
    "Unaccelerated (slow) nanargmax along axis 3."
    return bn.slow.nanargmax(arr, axis=3)

def nanargmax_slow_axis4(arr):
    "Unaccelerated (slow) nanargmax along axis 4."
    return bn.slow.nanargmax(arr, axis=4)

def nanargmax_slow_axis5(arr):
    "Unaccelerated (slow) nanargmax along axis 5."
    return bn.slow.nanargmax(arr, axis=5)

def nanargmax_slow_axis6(arr):
    "Unaccelerated (slow) nanargmax along axis 6."
    return bn.slow.nanargmax(arr, axis=6)

def nanargmax_slow_axis7(arr):
    "Unaccelerated (slow) nanargmax along axis 7."
    return bn.slow.nanargmax(arr, axis=7)

def nanargmax_slow_axis8(arr):
    "Unaccelerated (slow) nanargmax along axis 8."
    return bn.slow.nanargmax(arr, axis=8)

def nanargmax_slow_axis9(arr):
    "Unaccelerated (slow) nanargmax along axis 9."
    return bn.slow.nanargmax(arr, axis=9)

def nanargmax_slow_axis10(arr):
    "Unaccelerated (slow) nanargmax along axis 10."
    return bn.slow.nanargmax(arr, axis=10)

def nanargmax_slow_axis11(arr):
    "Unaccelerated (slow) nanargmax along axis 11."
    return bn.slow.nanargmax(arr, axis=11)

def nanargmax_slow_axis12(arr):
    "Unaccelerated (slow) nanargmax along axis 12."
    return bn.slow.nanargmax(arr, axis=12)

def nanargmax_slow_axis13(arr):
    "Unaccelerated (slow) nanargmax along axis 13."
    return bn.slow.nanargmax(arr, axis=13)

def nanargmax_slow_axis14(arr):
    "Unaccelerated (slow) nanargmax along axis 14."
    return bn.slow.nanargmax(arr, axis=14)

def nanargmax_slow_axis15(arr):
    "Unaccelerated (slow) nanargmax along axis 15."
    return bn.slow.nanargmax(arr, axis=15)

def nanargmax_slow_axis16(arr):
    "Unaccelerated (slow) nanargmax along axis 16."
    return bn.slow.nanargmax(arr, axis=16)

def nanargmax_slow_axis17(arr):
    "Unaccelerated (slow) nanargmax along axis 17."
    return bn.slow.nanargmax(arr, axis=17)

def nanargmax_slow_axis18(arr):
    "Unaccelerated (slow) nanargmax along axis 18."
    return bn.slow.nanargmax(arr, axis=18)

def nanargmax_slow_axis19(arr):
    "Unaccelerated (slow) nanargmax along axis 19."
    return bn.slow.nanargmax(arr, axis=19)

def nanargmax_slow_axis20(arr):
    "Unaccelerated (slow) nanargmax along axis 20."
    return bn.slow.nanargmax(arr, axis=20)

def nanargmax_slow_axis21(arr):
    "Unaccelerated (slow) nanargmax along axis 21."
    return bn.slow.nanargmax(arr, axis=21)

def nanargmax_slow_axis22(arr):
    "Unaccelerated (slow) nanargmax along axis 22."
    return bn.slow.nanargmax(arr, axis=22)

def nanargmax_slow_axis23(arr):
    "Unaccelerated (slow) nanargmax along axis 23."
    return bn.slow.nanargmax(arr, axis=23)

def nanargmax_slow_axis24(arr):
    "Unaccelerated (slow) nanargmax along axis 24."
    return bn.slow.nanargmax(arr, axis=24)

def nanargmax_slow_axis25(arr):
    "Unaccelerated (slow) nanargmax along axis 25."
    return bn.slow.nanargmax(arr, axis=25)

def nanargmax_slow_axis26(arr):
    "Unaccelerated (slow) nanargmax along axis 26."
    return bn.slow.nanargmax(arr, axis=26)

def nanargmax_slow_axis27(arr):
    "Unaccelerated (slow) nanargmax along axis 27."
    return bn.slow.nanargmax(arr, axis=27)

def nanargmax_slow_axis28(arr):
    "Unaccelerated (slow) nanargmax along axis 28."
    return bn.slow.nanargmax(arr, axis=28)

def nanargmax_slow_axis29(arr):
    "Unaccelerated (slow) nanargmax along axis 29."
    return bn.slow.nanargmax(arr, axis=29)

def nanargmax_slow_axis30(arr):
    "Unaccelerated (slow) nanargmax along axis 30."
    return bn.slow.nanargmax(arr, axis=30)

def nanargmax_slow_axis31(arr):
    "Unaccelerated (slow) nanargmax along axis 31."
    return bn.slow.nanargmax(arr, axis=31)

def nanargmax_slow_axis32(arr):
    "Unaccelerated (slow) nanargmax along axis 32."
    return bn.slow.nanargmax(arr, axis=32)

def nanargmax_slow_axisNone(arr):
    "Unaccelerated (slow) nanargmax along axis None."
    return bn.slow.nanargmax(arr, axis=None)
