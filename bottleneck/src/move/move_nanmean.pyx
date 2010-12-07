"move_nanmean"

# key is (ndim, dtype, axis)
cdef dict move_nanmean_dict = {}
move_nanmean_dict[(1, f64, 0)] = move_nanmean_1d_float64_axis0
move_nanmean_dict[(2, f64, 0)] = move_nanmean_2d_float64_axis0
move_nanmean_dict[(2, f64, 1)] = move_nanmean_2d_float64_axis1
move_nanmean_dict[(3, f64, 0)] = move_nanmean_3d_float64_axis0
move_nanmean_dict[(3, f64, 1)] = move_nanmean_3d_float64_axis1
move_nanmean_dict[(3, f64, 2)] = move_nanmean_3d_float64_axis2
move_nanmean_dict[(1, i64, 0)] = move_nanmean_1d_int64_axis0
move_nanmean_dict[(2, i64, 0)] = move_nanmean_2d_int64_axis0
move_nanmean_dict[(2, i64, 1)] = move_nanmean_2d_int64_axis1
move_nanmean_dict[(3, i64, 0)] = move_nanmean_3d_int64_axis0
move_nanmean_dict[(3, i64, 1)] = move_nanmean_3d_int64_axis1
move_nanmean_dict[(3, i64, 2)] = move_nanmean_3d_int64_axis2
move_nanmean_dict[(1, i32, 0)] = move_nanmean_1d_int32_axis0
move_nanmean_dict[(2, i32, 0)] = move_nanmean_2d_int32_axis0
move_nanmean_dict[(2, i32, 1)] = move_nanmean_2d_int32_axis1
move_nanmean_dict[(3, i32, 0)] = move_nanmean_3d_int32_axis0
move_nanmean_dict[(3, i32, 1)] = move_nanmean_3d_int32_axis1
move_nanmean_dict[(3, i32, 2)] = move_nanmean_3d_int32_axis2


def move_nanmean(arr, int window, int axis=0):
    """
    Moving window mean along the specified axis, ignoring NaNs.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving mean. By default the moving
        mean is taken over the first axis (axis=0). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis. The output
        has the same shape as the input. 

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_nanmean(arr, window=2)
    array([ nan,  1.5,  2.5,  3.5])

    """
    func, arr = move_nanmean_selector(arr, window, axis)
    return func(arr, window)

def move_nanmean_selector(arr, int window, int axis):
    """
    Return move_nanmean function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_nanmean() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the moving mean is to be computed. The default
        (axis=0) is to compute the moving mean along the first axis.
    
    Returns
    -------
    func : function
        The moving nanmean function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the mean.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarra; otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    
    Obtain the function needed to determine the sum of `arr` along axis=0:
    
    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_nanmean_selector(arr, window=2, axis=0)
    <built-in function move_nanmean_1d_float64_axis0>    
    
    Use the returned function and array to determine the sum:

    >>> func(a, window)
    array([ nan,  1.5,  2.5,  3.5])

    """
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef np.dtype dtype = a.dtype
    cdef int ndim = a.ndim
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = move_nanmean_dict[key]
    except KeyError:
        tup = (str(ndim), str(axis))
        raise TypeError, "Unsupported ndim/axis (%s/%s)." % tup
    return func, a

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a, int window):
    "Moving nanmean along a 1d numpy array of dtype=np.int32."

    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for i in range(window - 1):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        y[i] = NAN
    i = window - 1
    ai = a[i]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i] = <np.float64_t> asum / count
    else:
       y[i] = NAN
    for i in range(window, a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i] = <np.float64_t> asum / count
        else:
            y[i] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a, int window):
    "Moving nanmean along a 1d numpy array of dtype=np.int64."

    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for i in range(window - 1):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        y[i] = NAN
    i = window - 1
    ai = a[i]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i] = <np.float64_t> asum / count
    else:
       y[i] = NAN
    for i in range(window, a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i] = <np.float64_t> asum / count
        else:
            y[i] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a,
                                  int window):
    "Moving nanmean along a 1d numpy array of dtype=np.float64."

    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for i in range(window - 1):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        y[i] = NAN
    i = window - 1
    ai = a[i]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i] = asum / count
    else:
       y[i] = NAN
    for i in range(window, a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i] = asum / count
        else:
            y[i] = NAN

    return y

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a, int window):
    "Moving nanmean of a 2d numpy array of dtype=np.int32 along axis=0."
    
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        i = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = <np.float64_t> asum / count
        else:
           y[i,j] = NAN
        for i in range(window, a0):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i - window,j]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = <np.float64_t> asum / count
            else:
                y[i,j] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a, int window):
    "Moving nanmean of a 2d numpy array of dtype=np.int64 along axis=0."
    
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        i = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = <np.float64_t> asum / count
        else:
           y[i,j] = NAN
        for i in range(window, a0):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i - window,j]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = <np.float64_t> asum / count
            else:
                y[i,j] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                                  int window):
    "Moving nanmean of a 2d numpy array of dtype=np.float64 along axis=0."
    
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        i = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = asum / count
        else:
           y[i,j] = NAN
        for i in range(window, a0):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i - window,j]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = asum / count
            else:
                y[i,j] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a, int window):
    "Moving nanmean of a 2d numpy array of dtype=np.int32 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum, ai, aold
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        asum = 0
        count = 0
        for j in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        j = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = <np.float64_t> asum / count
        else:
           y[i,j] = NAN
        for j in range(window, a1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i, j - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = <np.float64_t> asum / count
            else:
                y[i,j] = NAN

    return y                

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a, int window):
    "Moving nanmean of a 2d numpy array of dtype=np.int64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum, ai, aold
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        asum = 0
        count = 0
        for j in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        j = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = <np.float64_t> asum / count
        else:
           y[i,j] = NAN
        for j in range(window, a1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i, j - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = <np.float64_t> asum / count
            else:
                y[i,j] = NAN

    return y                

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                                  int window):
    "Moving nanmean of a 2d numpy array of dtype=np.float64 along axis=1."
    cdef Py_ssize_t i, j
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef double asum, ai, aold
    cdef np.npy_intp *dims = [a0, a1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        asum = 0
        count = 0
        for j in range(window - 1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            y[i,j] = NAN
        j = window - 1
        ai = a[i,j]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i,j] = asum / count
        else:
           y[i,j] = NAN
        for j in range(window, a1):
            ai = a[i,j]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i, j - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i,j] = asum / count
            else:
                y[i,j] = NAN

    return y                

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int32 along axis=0."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)


    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        for k in range(a2):
            asum = 0
            count = 0
            for i in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            i = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for i in range(window, a0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i - window,j,k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int64 along axis=0."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)


    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        for k in range(a2):
            asum = 0
            count = 0
            for i in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            i = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for i in range(window, a0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i - window,j,k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving nanmean of a 3d numpy array of dtype=np.float64 along axis=0."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)


    if (window < 1) or (window > a0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a0)

    for j in range(a1):
        for k in range(a2):
            asum = 0
            count = 0
            for i in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            i = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = asum / count
            else:
               y[i,j,k] = NAN
            for i in range(window, a0):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i - window,j,k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int32 along axis=1."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        for k in range(a2):
            asum = 0
            count = 0
            for j in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            j = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for j in range(window, a1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j - window, k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int64 along axis=1."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        for k in range(a2):
            asum = 0
            count = 0
            for j in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            j = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for j in range(window, a1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j - window, k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving nanmean of a 3d numpy array of dtype=np.float64 along axis=1."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)
    
    for i in range(a0):
        for k in range(a2):
            asum = 0
            count = 0
            for j in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            j = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = asum / count
            else:
               y[i,j,k] = NAN
            for j in range(window, a1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j - window, k]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int32 along axis=2."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)

    for i in range(a0):
        for j in range(a1):
            asum = 0
            count = 0
            for k in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            k = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for k in range(window, a2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j, k - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a, int window):
    "Moving nanmean of a 3d numpy array of dtype=np.int64 along axis=2."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)

    for i in range(a0):
        for j in range(a1):
            asum = 0
            count = 0
            for k in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            k = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = <np.float64_t> asum / count
            else:
               y[i,j,k] = NAN
            for k in range(window, a2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j, k - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = <np.float64_t> asum / count
                else:
                    y[i,j,k] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving nanmean of a 3d numpy array of dtype=np.float64 along axis=2."

    cdef Py_ssize_t i, j, k
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef double asum = 0, aold, ai
    cdef np.npy_intp *dims = [a0, a1, a2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)

    if (window < 1) or (window > a2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, a1)

    for i in range(a0):
        for j in range(a1):
            asum = 0
            count = 0
            for k in range(window - 1):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i,j,k] = NAN
            k = window - 1
            ai = a[i,j,k]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i,j,k] = asum / count
            else:
               y[i,j,k] = NAN
            for k in range(window, a2):
                ai = a[i,j,k]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i, j, k - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i,j,k] = asum / count
                else:
                    y[i,j,k] = NAN

    return y
