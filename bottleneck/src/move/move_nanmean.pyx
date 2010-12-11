"move_nanmean auto-generated from template"

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

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a,
                                  int window):
    "Moving mean of 1d array of dtype=int32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i0 in range(window - 1):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        y[i0] = NAN
    i0 = window - 1
    ai = a[i0]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i0] = <np.float64_t> asum / count
    else:
       y[i0] = NAN
    for i0 in range(window, n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i0 - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i0] = <np.float64_t> asum / count
        else:
            y[i0] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a,
                                  int window):
    "Moving mean of 1d array of dtype=int64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i0 in range(window - 1):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        y[i0] = NAN
    i0 = window - 1
    ai = a[i0]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i0] = <np.float64_t> asum / count
    else:
       y[i0] = NAN
    for i0 in range(window, n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i0 - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i0] = <np.float64_t> asum / count
        else:
            y[i0] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=int32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        asum = 0
        count = 0
        for i0 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i0 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = <np.float64_t> asum / count
        else:
           y[i0, i1] = NAN
        for i0 in range(window, n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0 - window, i1]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = <np.float64_t> asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=int32 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        asum = 0
        count = 0
        for i1 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i1 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = <np.float64_t> asum / count
        else:
           y[i0, i1] = NAN
        for i1 in range(window, n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0, i1 - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = <np.float64_t> asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=int64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        asum = 0
        count = 0
        for i0 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i0 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = <np.float64_t> asum / count
        else:
           y[i0, i1] = NAN
        for i0 in range(window, n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0 - window, i1]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = <np.float64_t> asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=int64 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        asum = 0
        count = 0
        for i1 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i1 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = <np.float64_t> asum / count
        else:
           y[i0, i1] = NAN
        for i1 in range(window, n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0, i1 - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = <np.float64_t> asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int32 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i0 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i0 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i0 in range(window, n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0 - window, i1, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int32 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i1 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i1 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i1 in range(window, n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1 - window, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int32 along axis=2 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n2)

    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            count = 0
            for i2 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i2 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i2 in range(window, n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1, i2 - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i0 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i0 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i0 in range(window, n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0 - window, i1, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int64 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i1 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i1 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i1 in range(window, n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1 - window, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=int64 along axis=2 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n2)

    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            count = 0
            for i2 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i2 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = <np.float64_t> asum / count
            else:
               y[i0, i1, i2] = NAN
            for i2 in range(window, n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1, i2 - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = <np.float64_t> asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a,
                                  int window):
    "Moving mean of 1d array of dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i0 in range(window - 1):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        y[i0] = NAN
    i0 = window - 1
    ai = a[i0]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[i0] = asum / count
    else:
       y[i0] = NAN
    for i0 in range(window, n0):
        ai = a[i0]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[i0 - window]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[i0] = asum / count
        else:
            y[i0] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        asum = 0
        count = 0
        for i0 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i0 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = asum / count
        else:
           y[i0, i1] = NAN
        for i0 in range(window, n0):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0 - window, i1]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                                  int window):
    "Moving mean of 2d array of dtype=float64 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        asum = 0
        count = 0
        for i1 in range(window - 1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            y[i0, i1] = NAN
        i1 = window - 1
        ai = a[i0, i1]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[i0, i1] = asum / count
        else:
           y[i0, i1] = NAN
        for i1 in range(window, n1):
            ai = a[i0, i1]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[i0, i1 - window]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[i0, i1] = asum / count
            else:
                y[i0, i1] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=float64 along axis=0 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n0)

    for i1 in range(n1):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i0 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i0 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = asum / count
            else:
               y[i0, i1, i2] = NAN
            for i0 in range(window, n0):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0 - window, i1, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=float64 along axis=1 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n1):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n1)

    for i0 in range(n0):
        for i2 in range(n2):
            asum = 0
            count = 0
            for i1 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i1 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = asum / count
            else:
               y[i0, i1, i2] = NAN
            for i1 in range(window, n1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1 - window, i2]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def move_nanmean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a,
                                  int window):
    "Moving mean of 3d array of dtype=float64 along axis=2 ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                              NPY_float64, 0)
    if (window < 1) or (window > n2):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, n2)

    for i0 in range(n0):
        for i1 in range(n1):
            asum = 0
            count = 0
            for i2 in range(window - 1):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                y[i0, i1, i2] = NAN
            i2 = window - 1
            ai = a[i0, i1, i2]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[i0, i1, i2] = asum / count
            else:
               y[i0, i1, i2] = NAN
            for i2 in range(window, n2):
                ai = a[i0, i1, i2]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[i0, i1, i2 - window]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[i0, i1, i2] = asum / count
                else:
                    y[i0, i1, i2] = NAN

    return y

cdef dict move_nanmean_dict = {}
move_nanmean_dict[(1, int32, 0)] = move_nanmean_1d_int32_axis0
move_nanmean_dict[(1, int64, 0)] = move_nanmean_1d_int64_axis0
move_nanmean_dict[(2, int32, 0)] = move_nanmean_2d_int32_axis0
move_nanmean_dict[(2, int32, 1)] = move_nanmean_2d_int32_axis1
move_nanmean_dict[(2, int64, 0)] = move_nanmean_2d_int64_axis0
move_nanmean_dict[(2, int64, 1)] = move_nanmean_2d_int64_axis1
move_nanmean_dict[(3, int32, 0)] = move_nanmean_3d_int32_axis0
move_nanmean_dict[(3, int32, 1)] = move_nanmean_3d_int32_axis1
move_nanmean_dict[(3, int32, 2)] = move_nanmean_3d_int32_axis2
move_nanmean_dict[(3, int64, 0)] = move_nanmean_3d_int64_axis0
move_nanmean_dict[(3, int64, 1)] = move_nanmean_3d_int64_axis1
move_nanmean_dict[(3, int64, 2)] = move_nanmean_3d_int64_axis2
move_nanmean_dict[(1, float64, 0)] = move_nanmean_1d_float64_axis0
move_nanmean_dict[(2, float64, 0)] = move_nanmean_2d_float64_axis0
move_nanmean_dict[(2, float64, 1)] = move_nanmean_2d_float64_axis1
move_nanmean_dict[(3, float64, 0)] = move_nanmean_3d_float64_axis0
move_nanmean_dict[(3, float64, 1)] = move_nanmean_3d_float64_axis1
move_nanmean_dict[(3, float64, 2)] = move_nanmean_3d_float64_axis2