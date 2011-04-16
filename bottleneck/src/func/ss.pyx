# 64 bit version
"ss auto-generated from template"

def ss(arr, axis=0):
    """
    Sum of the square of each element along specified axis.

    Parameters
    ----------
    arr : array_like
        Array whose sum of squares is desired. If `arr` is not an array, a
        conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum if squared is computed. The default (axis=0)
        is to sum the squares along the first dimension.

    Returns
    -------
    y : ndarray
        The sum of a**2 along the given axis. 
    
    Examples
    --------
    >>> a = np.array([1., 2., 5.])
    >>> bn.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> bn.ss(b, axis=1)
    array([ 30., 65.])
    
    """
    func, arr = ss_selector(arr, axis)
    return func(arr)

def ss_selector(arr, axis):
    """
    Return ss function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.ss() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the sum of squares.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the sum of squares is to be computed.

    Returns
    -------
    func : function
        The ss function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the sum
        if squares.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 5.0])
    
    Obtain the function needed to determine the sum of squares of `arr` along
    axis=0:

    >>> func, a = bn.func.ss_selector(arr, axis=0)
    >>> func
    <built-in function ss_1d_float64_axisNone>    
    
    Use the returned function and array to determine the sum of squares:

    >>> func(a)
    30.0

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if dtype < NPY_int_:
        a = a.astype(np.int_)
        dtype = PyArray_TYPE(a)
    if (axis < 0) and (axis is not None):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = ss_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = ss_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int32 along axis=0."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        ssum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int32 along axis=1."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        ssum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i0] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int64 along axis=0."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        ssum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int64 along axis=1."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.int64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        ssum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i0] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int32 along axis=0."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            ssum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i1, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int32 along axis=1."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            ssum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int32 along axis=2."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            ssum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int64 along axis=0."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            ssum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i1, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int64 along axis=1."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            ssum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int64 along axis=2."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.int64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_int64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            ssum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_1d_float32_axisNone(np.ndarray[np.float32_t, ndim=1] a):
    "Sum of squares of 1d array with dtype=float32 along axis=None."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        ssum += ai * ai
    return np.float32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_1d_float64_axisNone(np.ndarray[np.float64_t, ndim=1] a):
    "Sum of squares of 1d array with dtype=float64 along axis=None."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        ssum += ai * ai
    return np.float64(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float32_axisNone(np.ndarray[np.float32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float32 along axis=None."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
    return np.float32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float64_axisNone(np.ndarray[np.float64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float64 along axis=None."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
    return np.float64(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float32_axisNone(np.ndarray[np.float32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float32 along axis=None."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
    return np.float32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float64_axisNone(np.ndarray[np.float64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float64 along axis=None."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
    return np.float64(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float32 along axis=0."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        ssum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float32 along axis=1."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        ssum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i0] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float64 along axis=0."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        ssum = 0
        for i0 in range(n0):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=float64 along axis=1."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        ssum = 0
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
        y[i0] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float32 along axis=0."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            ssum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i1, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float32 along axis=1."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            ssum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float32 along axis=2."
    cdef np.float32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            ssum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float64 along axis=0."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            ssum = 0
            for i0 in range(n0):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i1, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float64 along axis=1."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            ssum = 0
            for i1 in range(n1):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i2] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=float64 along axis=2."
    cdef np.float64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            ssum = 0
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
            y[i0, i1] = ssum
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_1d_int32_axisNone(np.ndarray[np.int32_t, ndim=1] a):
    "Sum of squares of 1d array with dtype=int32 along axis=None."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        ssum += ai * ai
    return np.int32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_1d_int64_axisNone(np.ndarray[np.int64_t, ndim=1] a):
    "Sum of squares of 1d array with dtype=int64 along axis=None."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    for i0 in range(n0):
        ai = a[i0]
        ssum += ai * ai
    return np.int64(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int32_axisNone(np.ndarray[np.int32_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int32 along axis=None."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
    return np.int32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_2d_int64_axisNone(np.ndarray[np.int64_t, ndim=2] a):
    "Sum of squares of 2d array with dtype=int64 along axis=None."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    for i0 in range(n0):
        for i1 in range(n1):
            ai = a[i0, i1]
            ssum += ai * ai
    return np.int64(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int32_axisNone(np.ndarray[np.int32_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int32 along axis=None."
    cdef np.int32_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
    return np.int32(ssum)

@cython.boundscheck(False)
@cython.wraparound(False)
def ss_3d_int64_axisNone(np.ndarray[np.int64_t, ndim=3] a):
    "Sum of squares of 3d array with dtype=int64 along axis=None."
    cdef np.int64_t ssum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef np.npy_intp *dim
    dim = PyArray_DIMS(a)
    cdef int n0 = dim[0]
    cdef int n1 = dim[1]
    cdef int n2 = dim[2]
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                ai = a[i0, i1, i2]
                ssum += ai * ai
    return np.int64(ssum)

cdef dict ss_dict = {}
ss_dict[(2, NPY_int32, 0)] = ss_2d_int32_axis0
ss_dict[(2, NPY_int32, 1)] = ss_2d_int32_axis1
ss_dict[(2, NPY_int64, 0)] = ss_2d_int64_axis0
ss_dict[(2, NPY_int64, 1)] = ss_2d_int64_axis1
ss_dict[(3, NPY_int32, 0)] = ss_3d_int32_axis0
ss_dict[(3, NPY_int32, 1)] = ss_3d_int32_axis1
ss_dict[(3, NPY_int32, 2)] = ss_3d_int32_axis2
ss_dict[(3, NPY_int64, 0)] = ss_3d_int64_axis0
ss_dict[(3, NPY_int64, 1)] = ss_3d_int64_axis1
ss_dict[(3, NPY_int64, 2)] = ss_3d_int64_axis2
ss_dict[(1, NPY_float32, 0)] = ss_1d_float32_axisNone
ss_dict[(1, NPY_float32, None)] = ss_1d_float32_axisNone
ss_dict[(1, NPY_float64, 0)] = ss_1d_float64_axisNone
ss_dict[(1, NPY_float64, None)] = ss_1d_float64_axisNone
ss_dict[(2, NPY_float32, None)] = ss_2d_float32_axisNone
ss_dict[(2, NPY_float64, None)] = ss_2d_float64_axisNone
ss_dict[(3, NPY_float32, None)] = ss_3d_float32_axisNone
ss_dict[(3, NPY_float64, None)] = ss_3d_float64_axisNone
ss_dict[(2, NPY_float32, 0)] = ss_2d_float32_axis0
ss_dict[(2, NPY_float32, 1)] = ss_2d_float32_axis1
ss_dict[(2, NPY_float64, 0)] = ss_2d_float64_axis0
ss_dict[(2, NPY_float64, 1)] = ss_2d_float64_axis1
ss_dict[(3, NPY_float32, 0)] = ss_3d_float32_axis0
ss_dict[(3, NPY_float32, 1)] = ss_3d_float32_axis1
ss_dict[(3, NPY_float32, 2)] = ss_3d_float32_axis2
ss_dict[(3, NPY_float64, 0)] = ss_3d_float64_axis0
ss_dict[(3, NPY_float64, 1)] = ss_3d_float64_axis1
ss_dict[(3, NPY_float64, 2)] = ss_3d_float64_axis2
ss_dict[(1, NPY_int32, 0)] = ss_1d_int32_axisNone
ss_dict[(1, NPY_int32, None)] = ss_1d_int32_axisNone
ss_dict[(1, NPY_int64, 0)] = ss_1d_int64_axisNone
ss_dict[(1, NPY_int64, None)] = ss_1d_int64_axisNone
ss_dict[(2, NPY_int32, None)] = ss_2d_int32_axisNone
ss_dict[(2, NPY_int64, None)] = ss_2d_int64_axisNone
ss_dict[(3, NPY_int32, None)] = ss_3d_int32_axisNone
ss_dict[(3, NPY_int64, None)] = ss_3d_int64_axisNone

def ss_slow_axis0(arr):
    "Unaccelerated (slow) ss along axis 0."
    return bn.slow.ss(arr, axis=0)

def ss_slow_axis1(arr):
    "Unaccelerated (slow) ss along axis 1."
    return bn.slow.ss(arr, axis=1)

def ss_slow_axis2(arr):
    "Unaccelerated (slow) ss along axis 2."
    return bn.slow.ss(arr, axis=2)

def ss_slow_axis3(arr):
    "Unaccelerated (slow) ss along axis 3."
    return bn.slow.ss(arr, axis=3)

def ss_slow_axis4(arr):
    "Unaccelerated (slow) ss along axis 4."
    return bn.slow.ss(arr, axis=4)

def ss_slow_axis5(arr):
    "Unaccelerated (slow) ss along axis 5."
    return bn.slow.ss(arr, axis=5)

def ss_slow_axis6(arr):
    "Unaccelerated (slow) ss along axis 6."
    return bn.slow.ss(arr, axis=6)

def ss_slow_axis7(arr):
    "Unaccelerated (slow) ss along axis 7."
    return bn.slow.ss(arr, axis=7)

def ss_slow_axis8(arr):
    "Unaccelerated (slow) ss along axis 8."
    return bn.slow.ss(arr, axis=8)

def ss_slow_axis9(arr):
    "Unaccelerated (slow) ss along axis 9."
    return bn.slow.ss(arr, axis=9)

def ss_slow_axis10(arr):
    "Unaccelerated (slow) ss along axis 10."
    return bn.slow.ss(arr, axis=10)

def ss_slow_axis11(arr):
    "Unaccelerated (slow) ss along axis 11."
    return bn.slow.ss(arr, axis=11)

def ss_slow_axis12(arr):
    "Unaccelerated (slow) ss along axis 12."
    return bn.slow.ss(arr, axis=12)

def ss_slow_axis13(arr):
    "Unaccelerated (slow) ss along axis 13."
    return bn.slow.ss(arr, axis=13)

def ss_slow_axis14(arr):
    "Unaccelerated (slow) ss along axis 14."
    return bn.slow.ss(arr, axis=14)

def ss_slow_axis15(arr):
    "Unaccelerated (slow) ss along axis 15."
    return bn.slow.ss(arr, axis=15)

def ss_slow_axis16(arr):
    "Unaccelerated (slow) ss along axis 16."
    return bn.slow.ss(arr, axis=16)

def ss_slow_axis17(arr):
    "Unaccelerated (slow) ss along axis 17."
    return bn.slow.ss(arr, axis=17)

def ss_slow_axis18(arr):
    "Unaccelerated (slow) ss along axis 18."
    return bn.slow.ss(arr, axis=18)

def ss_slow_axis19(arr):
    "Unaccelerated (slow) ss along axis 19."
    return bn.slow.ss(arr, axis=19)

def ss_slow_axis20(arr):
    "Unaccelerated (slow) ss along axis 20."
    return bn.slow.ss(arr, axis=20)

def ss_slow_axis21(arr):
    "Unaccelerated (slow) ss along axis 21."
    return bn.slow.ss(arr, axis=21)

def ss_slow_axis22(arr):
    "Unaccelerated (slow) ss along axis 22."
    return bn.slow.ss(arr, axis=22)

def ss_slow_axis23(arr):
    "Unaccelerated (slow) ss along axis 23."
    return bn.slow.ss(arr, axis=23)

def ss_slow_axis24(arr):
    "Unaccelerated (slow) ss along axis 24."
    return bn.slow.ss(arr, axis=24)

def ss_slow_axis25(arr):
    "Unaccelerated (slow) ss along axis 25."
    return bn.slow.ss(arr, axis=25)

def ss_slow_axis26(arr):
    "Unaccelerated (slow) ss along axis 26."
    return bn.slow.ss(arr, axis=26)

def ss_slow_axis27(arr):
    "Unaccelerated (slow) ss along axis 27."
    return bn.slow.ss(arr, axis=27)

def ss_slow_axis28(arr):
    "Unaccelerated (slow) ss along axis 28."
    return bn.slow.ss(arr, axis=28)

def ss_slow_axis29(arr):
    "Unaccelerated (slow) ss along axis 29."
    return bn.slow.ss(arr, axis=29)

def ss_slow_axis30(arr):
    "Unaccelerated (slow) ss along axis 30."
    return bn.slow.ss(arr, axis=30)

def ss_slow_axis31(arr):
    "Unaccelerated (slow) ss along axis 31."
    return bn.slow.ss(arr, axis=31)

def ss_slow_axis32(arr):
    "Unaccelerated (slow) ss along axis 32."
    return bn.slow.ss(arr, axis=32)

def ss_slow_axisNone(arr):
    "Unaccelerated (slow) ss along axis None."
    return bn.slow.ss(arr, axis=None)


cdef dict ss_slow_dict = {}
ss_slow_dict[0] = ss_slow_axis0
ss_slow_dict[1] = ss_slow_axis1
ss_slow_dict[2] = ss_slow_axis2
ss_slow_dict[3] = ss_slow_axis3
ss_slow_dict[4] = ss_slow_axis4
ss_slow_dict[5] = ss_slow_axis5
ss_slow_dict[6] = ss_slow_axis6
ss_slow_dict[7] = ss_slow_axis7
ss_slow_dict[8] = ss_slow_axis8
ss_slow_dict[9] = ss_slow_axis9
ss_slow_dict[10] = ss_slow_axis10
ss_slow_dict[11] = ss_slow_axis11
ss_slow_dict[12] = ss_slow_axis12
ss_slow_dict[13] = ss_slow_axis13
ss_slow_dict[14] = ss_slow_axis14
ss_slow_dict[15] = ss_slow_axis15
ss_slow_dict[16] = ss_slow_axis16
ss_slow_dict[17] = ss_slow_axis17
ss_slow_dict[18] = ss_slow_axis18
ss_slow_dict[19] = ss_slow_axis19
ss_slow_dict[20] = ss_slow_axis20
ss_slow_dict[21] = ss_slow_axis21
ss_slow_dict[22] = ss_slow_axis22
ss_slow_dict[23] = ss_slow_axis23
ss_slow_dict[24] = ss_slow_axis24
ss_slow_dict[25] = ss_slow_axis25
ss_slow_dict[26] = ss_slow_axis26
ss_slow_dict[27] = ss_slow_axis27
ss_slow_dict[28] = ss_slow_axis28
ss_slow_dict[29] = ss_slow_axis29
ss_slow_dict[30] = ss_slow_axis30
ss_slow_dict[31] = ss_slow_axis31
ss_slow_dict[32] = ss_slow_axis32
ss_slow_dict[None] = ss_slow_axisNone