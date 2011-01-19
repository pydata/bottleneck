"median auto-generated from template"
# Select smallest k elements code used for inner loop of median method:
# http://projects.scipy.org/numpy/attachment/ticket/1213/quickselect.pyx
# (C) 2009 Sturla Molden
# SciPy license 
#
# From the original C function (code in public domain) in:
#   Fast median search: an ANSI C implementation
#   Nicolas Devillard - ndevilla AT free DOT fr
#   July 1998
# which, in turn, took the algorithm from
#   Wirth, Niklaus
#   Algorithms + data structures = programs, p. 366
#   Englewood Cliffs: Prentice-Hall, 1976
#
# Adapted and expanded for Bottleneck:
# (C) 2010 Keith Goodman

def median(arr, axis=None):
    """
    Median of array elements along given axis.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is computed. The default (axis=None) is to
        compute the median of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, except that the specified axis
        has been removed. If `arr` is a 0d array, or if axis is None, a scalar
        is returned. `float64` return values are used for integer inputs. 
    
    See also
    --------
    bottleneck.nanmedian: Median along specified axis ignoring NaNs. 

    Notes
    -----
    This function returns the same output as NumPy's median except when the
    input contains NaN.

    Examples
    --------
    >>> a = np.array([[10, 7, 4], [3, 2, 1]])
    >>> a
    array([[10,  7,  4],
           [ 3,  2,  1]])
    >>> bn.median(a)
    3.5
    >>> bn.median(a, axis=0)
    array([ 6.5,  4.5,  2.5])
    >>> bn.median(a, axis=1)
    array([ 7.,  2.])
    
    """
    func, arr = median_selector(arr, axis)
    return func(arr)

def median_selector(arr, axis):
    """
    Return median function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.median() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the mean.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is to be computed. The default (axis=None)
        is to compute the mean of the flattened array.
    
    Returns
    -------
    func : function
        The median function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the
        median.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    
    Obtain the function needed to determine the median of `arr` along axis=0:

    >>> func, a = bn.func.median_selector(arr, axis=0)
    >>> func
    <built-in function median_1d_float64_axis0> 
    
    Use the returned function and array to determine the median:

    >>> func(a)
    2.0

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef tuple key
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = a.ravel()
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    try:
        func = median_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = median_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "Median of 1d array with dtype=int32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    k = n0 >> 1
    l = 0
    r = n0 - 1 
    with nogil:       
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n0 % 2 == 0:        
        amax = MINint32
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float64(0.5 * (a[k] + amax))
    else:
        return np.float64(a[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a):
    "Median of 1d array with dtype=int64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    k = n0 >> 1
    l = 0
    r = n0 - 1 
    with nogil:       
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n0 % 2 == 0:        
        amax = MINint64
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float64(0.5 * (a[k] + amax))
    else:
        return np.float64(a[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Median of 2d array with dtype=int32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k, i1]
            i = l
            j = r
            while 1:
                while a[i, i1] < x: i += 1
                while x < a[j, i1]: j -= 1
                if i <= j:
                    tmp = a[i, i1]
                    a[i, i1] = a[j, i1]
                    a[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MINint32
            for i in range(k):
                ai = a[i, i1]
                if ai >= amax:
                    amax = ai
            y[i1] = 0.5 * (a[k, i1] + amax)
        else:
            y[i1] = <np.float64_t> a[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Median of 2d array with dtype=int32 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[i0, k]
            i = l
            j = r
            while 1:
                while a[i0, i] < x: i += 1
                while x < a[i0, j]: j -= 1
                if i <= j:
                    tmp = a[i0, i]
                    a[i0, i] = a[i0, j]
                    a[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MINint32
            for i in range(k):
                ai = a[i0, i]
                if ai >= amax:
                    amax = ai
            y[i0] = 0.5 * (a[i0, k] + amax)
        else:
            y[i0] = <np.float64_t> a[i0, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Median of 2d array with dtype=int64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k, i1]
            i = l
            j = r
            while 1:
                while a[i, i1] < x: i += 1
                while x < a[j, i1]: j -= 1
                if i <= j:
                    tmp = a[i, i1]
                    a[i, i1] = a[j, i1]
                    a[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MINint64
            for i in range(k):
                ai = a[i, i1]
                if ai >= amax:
                    amax = ai
            y[i1] = 0.5 * (a[k, i1] + amax)
        else:
            y[i1] = <np.float64_t> a[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Median of 2d array with dtype=int64 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[i0, k]
            i = l
            j = r
            while 1:
                while a[i0, i] < x: i += 1
                while x < a[i0, j]: j -= 1
                if i <= j:
                    tmp = a[i0, i]
                    a[i0, i] = a[i0, j]
                    a[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MINint64
            for i in range(k):
                ai = a[i0, i]
                if ai >= amax:
                    amax = ai
            y[i0] = 0.5 * (a[i0, k] + amax)
        else:
            y[i0] = <np.float64_t> a[i0, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d array with dtype=int32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k, i1, i2]
                i = l
                j = r
                while 1:
                    while a[i, i1, i2] < x: i += 1
                    while x < a[j, i1, i2]: j -= 1
                    if i <= j:
                        tmp = a[i, i1, i2]
                        a[i, i1, i2] = a[j, i1, i2]
                        a[j, i1, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MINint32
                for i in range(k):
                    ai = a[i, i1, i2]
                    if ai >= amax:
                        amax = ai
                y[i1, i2] = 0.5 * (a[k, i1, i2] + amax)
            else:
                y[i1, i2] = <np.float64_t> a[k, i1, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d array with dtype=int32 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[i0, k, i2]
                i = l
                j = r
                while 1:
                    while a[i0, i, i2] < x: i += 1
                    while x < a[i0, j, i2]: j -= 1
                    if i <= j:
                        tmp = a[i0, i, i2]
                        a[i0, i, i2] = a[i0, j, i2]
                        a[i0, j, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MINint32
                for i in range(k):
                    ai = a[i0, i, i2]
                    if ai >= amax:
                        amax = ai
                y[i0, i2] = 0.5 * (a[i0, k, i2] + amax)
            else:
                y[i0, i2] = <np.float64_t> a[i0, k, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d array with dtype=int32 along axis=2."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[i0, i1, k]
                i = l
                j = r
                while 1:
                    while a[i0, i1, i] < x: i += 1
                    while x < a[i0, i1, j]: j -= 1
                    if i <= j:
                        tmp = a[i0, i1, i]
                        a[i0, i1, i] = a[i0, i1, j]
                        a[i0, i1, j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MINint32
                for i in range(k):
                    ai = a[i0, i1, i]
                    if ai >= amax:
                        amax = ai
                y[i0, i1] = 0.5 * (a[i0, i1, k] + amax)
            else:
                y[i0, i1] = <np.float64_t> a[i0, i1, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d array with dtype=int64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k, i1, i2]
                i = l
                j = r
                while 1:
                    while a[i, i1, i2] < x: i += 1
                    while x < a[j, i1, i2]: j -= 1
                    if i <= j:
                        tmp = a[i, i1, i2]
                        a[i, i1, i2] = a[j, i1, i2]
                        a[j, i1, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MINint64
                for i in range(k):
                    ai = a[i, i1, i2]
                    if ai >= amax:
                        amax = ai
                y[i1, i2] = 0.5 * (a[k, i1, i2] + amax)
            else:
                y[i1, i2] = <np.float64_t> a[k, i1, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d array with dtype=int64 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[i0, k, i2]
                i = l
                j = r
                while 1:
                    while a[i0, i, i2] < x: i += 1
                    while x < a[i0, j, i2]: j -= 1
                    if i <= j:
                        tmp = a[i0, i, i2]
                        a[i0, i, i2] = a[i0, j, i2]
                        a[i0, j, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MINint64
                for i in range(k):
                    ai = a[i0, i, i2]
                    if ai >= amax:
                        amax = ai
                y[i0, i2] = 0.5 * (a[i0, k, i2] + amax)
            else:
                y[i0, i2] = <np.float64_t> a[i0, k, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d array with dtype=int64 along axis=2."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.int64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[i0, i1, k]
                i = l
                j = r
                while 1:
                    while a[i0, i1, i] < x: i += 1
                    while x < a[i0, i1, j]: j -= 1
                    if i <= j:
                        tmp = a[i0, i1, i]
                        a[i0, i1, i] = a[i0, i1, j]
                        a[i0, i1, j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MINint64
                for i in range(k):
                    ai = a[i0, i1, i]
                    if ai >= amax:
                        amax = ai
                y[i0, i1] = 0.5 * (a[i0, i1, k] + amax)
            else:
                y[i0, i1] = <np.float64_t> a[i0, i1, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a):
    "Median of 1d array with dtype=float32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    k = n0 >> 1
    l = 0
    r = n0 - 1 
    with nogil:       
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n0 % 2 == 0:        
        amax = MINfloat32
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float32(0.5 * (a[k] + amax))
    else:
        return np.float32(a[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Median of 1d array with dtype=float64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    k = n0 >> 1
    l = 0
    r = n0 - 1 
    with nogil:       
        while l < r:
            x = a[k]
            i = l
            j = r
            while 1:
                while a[i] < x: i += 1
                while x < a[j]: j -= 1
                if i <= j:
                    tmp = a[i]
                    a[i] = a[j]
                    a[j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    if n0 % 2 == 0:        
        amax = MINfloat64
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float64(0.5 * (a[k] + amax))
    else:
        return np.float64(a[k])

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a):
    "Median of 2d array with dtype=float32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i1 in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k, i1]
            i = l
            j = r
            while 1:
                while a[i, i1] < x: i += 1
                while x < a[j, i1]: j -= 1
                if i <= j:
                    tmp = a[i, i1]
                    a[i, i1] = a[j, i1]
                    a[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MINfloat32
            for i in range(k):
                ai = a[i, i1]
                if ai >= amax:
                    amax = ai
            y[i1] = 0.5 * (a[k, i1] + amax)
        else:
            y[i1] = a[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a):
    "Median of 2d array with dtype=float32 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float32, 0)
    for i0 in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[i0, k]
            i = l
            j = r
            while 1:
                while a[i0, i] < x: i += 1
                while x < a[i0, j]: j -= 1
                if i <= j:
                    tmp = a[i0, i]
                    a[i0, i] = a[i0, j]
                    a[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MINfloat32
            for i in range(k):
                ai = a[i0, i]
                if ai >= amax:
                    amax = ai
            y[i0] = 0.5 * (a[i0, k] + amax)
        else:
            y[i0] = a[i0, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d array with dtype=float64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i1 in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k, i1]
            i = l
            j = r
            while 1:
                while a[i, i1] < x: i += 1
                while x < a[j, i1]: j -= 1
                if i <= j:
                    tmp = a[i, i1]
                    a[i, i1] = a[j, i1]
                    a[j, i1] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MINfloat64
            for i in range(k):
                ai = a[i, i1]
                if ai >= amax:
                    amax = ai
            y[i1] = 0.5 * (a[k, i1] + amax)
        else:
            y[i1] = a[k, i1]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d array with dtype=float64 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                              NPY_float64, 0)
    for i0 in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[i0, k]
            i = l
            j = r
            while 1:
                while a[i0, i] < x: i += 1
                while x < a[i0, j]: j -= 1
                if i <= j:
                    tmp = a[i0, i]
                    a[i0, i] = a[i0, j]
                    a[i0, j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MINfloat64
            for i in range(k):
                ai = a[i0, i]
                if ai >= amax:
                    amax = ai
            y[i0] = 0.5 * (a[i0, k] + amax)
        else:
            y[i0] = a[i0, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a):
    "Median of 3d array with dtype=float32 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k, i1, i2]
                i = l
                j = r
                while 1:
                    while a[i, i1, i2] < x: i += 1
                    while x < a[j, i1, i2]: j -= 1
                    if i <= j:
                        tmp = a[i, i1, i2]
                        a[i, i1, i2] = a[j, i1, i2]
                        a[j, i1, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MINfloat32
                for i in range(k):
                    ai = a[i, i1, i2]
                    if ai >= amax:
                        amax = ai
                y[i1, i2] = 0.5 * (a[k, i1, i2] + amax)
            else:
                y[i1, i2] = a[k, i1, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a):
    "Median of 3d array with dtype=float32 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[i0, k, i2]
                i = l
                j = r
                while 1:
                    while a[i0, i, i2] < x: i += 1
                    while x < a[i0, j, i2]: j -= 1
                    if i <= j:
                        tmp = a[i0, i, i2]
                        a[i0, i, i2] = a[i0, j, i2]
                        a[i0, j, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MINfloat32
                for i in range(k):
                    ai = a[i0, i, i2]
                    if ai >= amax:
                        amax = ai
                y[i0, i2] = 0.5 * (a[i0, k, i2] + amax)
            else:
                y[i0, i2] = a[i0, k, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a):
    "Median of 3d array with dtype=float32 along axis=2."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float32_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float32, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[i0, i1, k]
                i = l
                j = r
                while 1:
                    while a[i0, i1, i] < x: i += 1
                    while x < a[i0, i1, j]: j -= 1
                    if i <= j:
                        tmp = a[i0, i1, i]
                        a[i0, i1, i] = a[i0, i1, j]
                        a[i0, i1, j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MINfloat32
                for i in range(k):
                    ai = a[i0, i1, i]
                    if ai >= amax:
                        amax = ai
                y[i0, i1] = 0.5 * (a[i0, i1, k] + amax)
            else:
                y[i0, i1] = a[i0, i1, k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d array with dtype=float64 along axis=0."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i1 in range(n1):
        for i2 in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k, i1, i2]
                i = l
                j = r
                while 1:
                    while a[i, i1, i2] < x: i += 1
                    while x < a[j, i1, i2]: j -= 1
                    if i <= j:
                        tmp = a[i, i1, i2]
                        a[i, i1, i2] = a[j, i1, i2]
                        a[j, i1, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MINfloat64
                for i in range(k):
                    ai = a[i, i1, i2]
                    if ai >= amax:
                        amax = ai
                y[i1, i2] = 0.5 * (a[k, i1, i2] + amax)
            else:
                y[i1, i2] = a[k, i1, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d array with dtype=float64 along axis=1."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i2 in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[i0, k, i2]
                i = l
                j = r
                while 1:
                    while a[i0, i, i2] < x: i += 1
                    while x < a[i0, j, i2]: j -= 1
                    if i <= j:
                        tmp = a[i0, i, i2]
                        a[i0, i, i2] = a[i0, j, i2]
                        a[i0, j, i2] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MINfloat64
                for i in range(k):
                    ai = a[i0, i, i2]
                    if ai >= amax:
                        amax = ai
                y[i0, i2] = 0.5 * (a[i0, k, i2] + amax)
            else:
                y[i0, i2] = a[i0, k, i2]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d array with dtype=float64 along axis=2."
    cdef np.npy_intp i, j = 0, l, r, k 
    cdef np.float64_t x, tmp, amax, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                              NPY_float64, 0)
    for i0 in range(n0):
        for i1 in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[i0, i1, k]
                i = l
                j = r
                while 1:
                    while a[i0, i1, i] < x: i += 1
                    while x < a[i0, i1, j]: j -= 1
                    if i <= j:
                        tmp = a[i0, i1, i]
                        a[i0, i1, i] = a[i0, i1, j]
                        a[i0, i1, j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MINfloat64
                for i in range(k):
                    ai = a[i0, i1, i]
                    if ai >= amax:
                        amax = ai
                y[i0, i1] = 0.5 * (a[i0, i1, k] + amax)
            else:
                y[i0, i1] = a[i0, i1, k]         
    return y

cdef dict median_dict = {}
median_dict[(1, int32, 0)] = median_1d_int32_axis0
median_dict[(1, int64, 0)] = median_1d_int64_axis0
median_dict[(2, int32, 0)] = median_2d_int32_axis0
median_dict[(2, int32, 1)] = median_2d_int32_axis1
median_dict[(2, int64, 0)] = median_2d_int64_axis0
median_dict[(2, int64, 1)] = median_2d_int64_axis1
median_dict[(3, int32, 0)] = median_3d_int32_axis0
median_dict[(3, int32, 1)] = median_3d_int32_axis1
median_dict[(3, int32, 2)] = median_3d_int32_axis2
median_dict[(3, int64, 0)] = median_3d_int64_axis0
median_dict[(3, int64, 1)] = median_3d_int64_axis1
median_dict[(3, int64, 2)] = median_3d_int64_axis2
median_dict[(1, float32, 0)] = median_1d_float32_axis0
median_dict[(1, float64, 0)] = median_1d_float64_axis0
median_dict[(2, float32, 0)] = median_2d_float32_axis0
median_dict[(2, float32, 1)] = median_2d_float32_axis1
median_dict[(2, float64, 0)] = median_2d_float64_axis0
median_dict[(2, float64, 1)] = median_2d_float64_axis1
median_dict[(3, float32, 0)] = median_3d_float32_axis0
median_dict[(3, float32, 1)] = median_3d_float32_axis1
median_dict[(3, float32, 2)] = median_3d_float32_axis2
median_dict[(3, float64, 0)] = median_3d_float64_axis0
median_dict[(3, float64, 1)] = median_3d_float64_axis1
median_dict[(3, float64, 2)] = median_3d_float64_axis2

cdef dict median_slow_dict = {}
median_slow_dict[0] = median_slow_axis0
median_slow_dict[1] = median_slow_axis1
median_slow_dict[2] = median_slow_axis2
median_slow_dict[3] = median_slow_axis3
median_slow_dict[4] = median_slow_axis4
median_slow_dict[5] = median_slow_axis5
median_slow_dict[6] = median_slow_axis6
median_slow_dict[7] = median_slow_axis7
median_slow_dict[8] = median_slow_axis8
median_slow_dict[9] = median_slow_axis9
median_slow_dict[10] = median_slow_axis10
median_slow_dict[11] = median_slow_axis11
median_slow_dict[12] = median_slow_axis12
median_slow_dict[13] = median_slow_axis13
median_slow_dict[14] = median_slow_axis14
median_slow_dict[15] = median_slow_axis15
median_slow_dict[16] = median_slow_axis16
median_slow_dict[17] = median_slow_axis17
median_slow_dict[18] = median_slow_axis18
median_slow_dict[19] = median_slow_axis19
median_slow_dict[20] = median_slow_axis20
median_slow_dict[21] = median_slow_axis21
median_slow_dict[22] = median_slow_axis22
median_slow_dict[23] = median_slow_axis23
median_slow_dict[24] = median_slow_axis24
median_slow_dict[25] = median_slow_axis25
median_slow_dict[26] = median_slow_axis26
median_slow_dict[27] = median_slow_axis27
median_slow_dict[28] = median_slow_axis28
median_slow_dict[29] = median_slow_axis29
median_slow_dict[30] = median_slow_axis30
median_slow_dict[31] = median_slow_axis31
median_slow_dict[32] = median_slow_axis32
median_slow_dict[None] = median_slow_axisNone

def median_slow_axis0(arr):
    "Unaccelerated (slow) median along axis 0."
    return bn.slow.median(arr, axis=0)

def median_slow_axis1(arr):
    "Unaccelerated (slow) median along axis 1."
    return bn.slow.median(arr, axis=1)

def median_slow_axis2(arr):
    "Unaccelerated (slow) median along axis 2."
    return bn.slow.median(arr, axis=2)

def median_slow_axis3(arr):
    "Unaccelerated (slow) median along axis 3."
    return bn.slow.median(arr, axis=3)

def median_slow_axis4(arr):
    "Unaccelerated (slow) median along axis 4."
    return bn.slow.median(arr, axis=4)

def median_slow_axis5(arr):
    "Unaccelerated (slow) median along axis 5."
    return bn.slow.median(arr, axis=5)

def median_slow_axis6(arr):
    "Unaccelerated (slow) median along axis 6."
    return bn.slow.median(arr, axis=6)

def median_slow_axis7(arr):
    "Unaccelerated (slow) median along axis 7."
    return bn.slow.median(arr, axis=7)

def median_slow_axis8(arr):
    "Unaccelerated (slow) median along axis 8."
    return bn.slow.median(arr, axis=8)

def median_slow_axis9(arr):
    "Unaccelerated (slow) median along axis 9."
    return bn.slow.median(arr, axis=9)

def median_slow_axis10(arr):
    "Unaccelerated (slow) median along axis 10."
    return bn.slow.median(arr, axis=10)

def median_slow_axis11(arr):
    "Unaccelerated (slow) median along axis 11."
    return bn.slow.median(arr, axis=11)

def median_slow_axis12(arr):
    "Unaccelerated (slow) median along axis 12."
    return bn.slow.median(arr, axis=12)

def median_slow_axis13(arr):
    "Unaccelerated (slow) median along axis 13."
    return bn.slow.median(arr, axis=13)

def median_slow_axis14(arr):
    "Unaccelerated (slow) median along axis 14."
    return bn.slow.median(arr, axis=14)

def median_slow_axis15(arr):
    "Unaccelerated (slow) median along axis 15."
    return bn.slow.median(arr, axis=15)

def median_slow_axis16(arr):
    "Unaccelerated (slow) median along axis 16."
    return bn.slow.median(arr, axis=16)

def median_slow_axis17(arr):
    "Unaccelerated (slow) median along axis 17."
    return bn.slow.median(arr, axis=17)

def median_slow_axis18(arr):
    "Unaccelerated (slow) median along axis 18."
    return bn.slow.median(arr, axis=18)

def median_slow_axis19(arr):
    "Unaccelerated (slow) median along axis 19."
    return bn.slow.median(arr, axis=19)

def median_slow_axis20(arr):
    "Unaccelerated (slow) median along axis 20."
    return bn.slow.median(arr, axis=20)

def median_slow_axis21(arr):
    "Unaccelerated (slow) median along axis 21."
    return bn.slow.median(arr, axis=21)

def median_slow_axis22(arr):
    "Unaccelerated (slow) median along axis 22."
    return bn.slow.median(arr, axis=22)

def median_slow_axis23(arr):
    "Unaccelerated (slow) median along axis 23."
    return bn.slow.median(arr, axis=23)

def median_slow_axis24(arr):
    "Unaccelerated (slow) median along axis 24."
    return bn.slow.median(arr, axis=24)

def median_slow_axis25(arr):
    "Unaccelerated (slow) median along axis 25."
    return bn.slow.median(arr, axis=25)

def median_slow_axis26(arr):
    "Unaccelerated (slow) median along axis 26."
    return bn.slow.median(arr, axis=26)

def median_slow_axis27(arr):
    "Unaccelerated (slow) median along axis 27."
    return bn.slow.median(arr, axis=27)

def median_slow_axis28(arr):
    "Unaccelerated (slow) median along axis 28."
    return bn.slow.median(arr, axis=28)

def median_slow_axis29(arr):
    "Unaccelerated (slow) median along axis 29."
    return bn.slow.median(arr, axis=29)

def median_slow_axis30(arr):
    "Unaccelerated (slow) median along axis 30."
    return bn.slow.median(arr, axis=30)

def median_slow_axis31(arr):
    "Unaccelerated (slow) median along axis 31."
    return bn.slow.median(arr, axis=31)

def median_slow_axis32(arr):
    "Unaccelerated (slow) median along axis 32."
    return bn.slow.median(arr, axis=32)

def median_slow_axisNone(arr):
    "Unaccelerated (slow) median along axis None."
    return bn.slow.median(arr, axis=None)
