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
# Adapted for Bottleneck:
# (C) 2010 Keith Goodman
"median"

# key is (ndim, dtype, axis)
cdef dict median_dict = {}
median_dict[(1, f64, 0)] = median_1d_float64_axis0
median_dict[(2, f64, 0)] = median_2d_float64_axis0
median_dict[(2, f64, 1)] = median_2d_float64_axis1
median_dict[(3, f64, 0)] = median_3d_float64_axis0
median_dict[(3, f64, 1)] = median_3d_float64_axis1
median_dict[(3, f64, 2)] = median_3d_float64_axis2
median_dict[(1, i64, 0)] = median_1d_int64_axis0
median_dict[(2, i64, 0)] = median_2d_int64_axis0
median_dict[(2, i64, 1)] = median_2d_int64_axis1
median_dict[(3, i64, 0)] = median_3d_int64_axis0
median_dict[(3, i64, 1)] = median_3d_int64_axis1
median_dict[(3, i64, 2)] = median_3d_int64_axis2
median_dict[(1, i32, 0)] = median_1d_int32_axis0
median_dict[(2, i32, 0)] = median_2d_int32_axis0
median_dict[(2, i32, 1)] = median_2d_int32_axis1
median_dict[(3, i32, 0)] = median_3d_int32_axis0
median_dict[(3, i32, 1)] = median_3d_int32_axis1
median_dict[(3, i32, 2)] = median_3d_int32_axis2


def median(arr, axis=None):
    """
    Median of array elements along given axis.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the median is computed. The default (axis=None)is to
        compute the median of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, except that the specified axis
        has been removed. If `arr` is a 0d array, or if axis is None, a scalar
        is returned. `float64` return values are used for integer inputs. 

    Notes
    -----
    This function should give the same output as NumPy's median except for
    when the input contains NaN.

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
    cdef np.ndarray a = np.array(arr, copy=True)
    cdef tuple key
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    if axis != None:
        if axis < 0:
            axis += ndim
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
    else:
        a = a.ravel()
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    try:
        func = median_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError, "Unsupported ndim/dtype (%s/%s)." % tup
    return func, a

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a):
    "Median of 1d numpy array with dtype=np.int32 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n
    cdef np.int32_t x, tmp, amax, ai
    n = a.shape[0]
    k = n >> 1
    l = 0
    r = n - 1 
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
    if n % 2 == 0:        
        amax = MININT32
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
    "Median of 1d numpy array with dtype=np.int64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n
    cdef np.int64_t x, tmp, amax, ai
    n = a.shape[0]
    k = n >> 1
    l = 0
    r = n - 1 
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
    if n % 2 == 0:        
        amax = MININT64
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float64(0.5 * (a[k] + amax))
    else:
        return np.float64(a[k])         

@cython.boundscheck(False)
@cython.wraparound(False)
def median_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a):
    "Median of 1d numpy array with dtype=np.float64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n
    cdef np.float64_t x, tmp, amax, ai
    n = a.shape[0]
    k = n >> 1
    l = 0
    r = n - 1 
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
    if n % 2 == 0:        
        amax = np.NINF
        for i in range(k):
            ai = a[i]
            if ai >= amax:
                amax = ai
        return np.float64((a[k] + amax) / 2)
    else:
        return np.float64(a[k])         


# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.int32 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, jj 
    cdef np.int32_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for jj in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k,jj]
            i = l
            j = r
            while 1:
                while a[i,jj] < x: i += 1
                while x < a[j,jj]: j -= 1
                if i <= j:
                    tmp = a[i,jj]
                    a[i,jj] = a[j,jj]
                    a[j,jj] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MININT32
            for i in range(k):
                ai = a[i,jj]
                if ai >= amax:
                    amax = ai
            y[jj] = 0.5 * (a[k,jj] + amax)
        else:
            y[jj] = <np.float64_t> a[k,jj]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.int32 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, ii 
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for ii in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[ii, k]
            i = l
            j = r
            while 1:
                while a[ii,i] < x: i += 1
                while x < a[ii,j]: j -= 1
                if i <= j:
                    tmp = a[ii,i]
                    a[ii,i] = a[ii,j]
                    a[ii,j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MININT32
            for i in range(k):
                ai = a[ii,i]
                if ai >= amax:
                    amax = ai
            y[ii] = 0.5 * (a[ii,k] + amax) 
        else:
            y[ii] = <np.float64_t> a[ii,k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.int64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, jj 
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for jj in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k,jj]
            i = l
            j = r
            while 1:
                while a[i,jj] < x: i += 1
                while x < a[j,jj]: j -= 1
                if i <= j:
                    tmp = a[i,jj]
                    a[i,jj] = a[j,jj]
                    a[j,jj] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = MININT64
            for i in range(k):
                ai = a[i,jj]
                if ai >= amax:
                    amax = ai
            y[jj] = 0.5 * (a[k,jj] + amax)
        else:
            y[jj] = <np.float64_t> a[k,jj]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.int64 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, ii 
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for ii in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[ii, k]
            i = l
            j = r
            while 1:
                while a[ii,i] < x: i += 1
                while x < a[ii,j]: j -= 1
                if i <= j:
                    tmp = a[ii,i]
                    a[ii,i] = a[ii,j]
                    a[ii,j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = MININT64
            for i in range(k):
                ai = a[ii,i]
                if ai >= amax:
                    amax = ai
            y[ii] = 0.5 * (a[ii,k] + amax) 
        else:
            y[ii] = <np.float64_t> a[ii,k]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.float64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, jj 
    cdef np.float64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n1]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for jj in range(n1): 
        k = n0 >> 1
        l = 0
        r = n0 - 1
        while l < r:
            x = a[k,jj]
            i = l
            j = r
            while 1:
                while a[i,jj] < x: i += 1
                while x < a[j,jj]: j -= 1
                if i <= j:
                    tmp = a[i,jj]
                    a[i,jj] = a[j,jj]
                    a[j,jj] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n0 % 2 == 0:        
            amax = np.NINF
            for i in range(k):
                ai = a[i,jj]
                if ai >= amax:
                    amax = ai
            y[jj] = (a[k,jj] + amax) / 2 
        else:
            y[jj] = a[k,jj]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a):
    "Median of 2d numpy array with dtype=np.float64 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, ii 
    cdef np.float64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for ii in range(n0): 
        k = n1 >> 1
        l = 0
        r = n1 - 1
        while l < r:
            x = a[ii, k]
            i = l
            j = r
            while 1:
                while a[ii,i] < x: i += 1
                while x < a[ii,j]: j -= 1
                if i <= j:
                    tmp = a[ii,i]
                    a[ii,i] = a[ii,j]
                    a[ii,j] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        if n1 % 2 == 0:        
            amax = np.NINF
            for i in range(k):
                ai = a[ii,i]
                if ai >= amax:
                    amax = ai
            y[ii] = (a[ii,k] + amax) / 2 
        else:
            y[ii] = a[ii,k]         
    return y

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int32 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, jj, kk
    cdef np.int32_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for jj in range(n1):
        for kk in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k,jj,kk]
                i = l
                j = r
                while 1:
                    while a[i,jj,kk] < x: i += 1
                    while x < a[j,jj,kk]: j -= 1
                    if i <= j:
                        tmp = a[i,jj,kk]
                        a[i,jj,kk] = a[j,jj,kk]
                        a[j,jj,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MININT32
                for i in range(k):
                    ai = a[i,jj,kk]
                    if ai >= amax:
                        amax = ai
                y[jj,kk] = 0.5 * (a[k,jj,kk] + amax) 
            else:
                y[jj,kk] = <np.float64_t> a[k,jj,kk]       
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int32 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, kk
    cdef np.int32_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for kk in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[ii,k,kk]
                i = l
                j = r
                while 1:
                    while a[ii,i,kk] < x: i += 1
                    while x < a[ii,j,kk]: j -= 1
                    if i <= j:
                        tmp = a[ii,i,kk]
                        a[ii,i,kk] = a[ii,j,kk]
                        a[ii,j,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MININT32
                for i in range(k):
                    ai = a[ii,i,kk]
                    if ai >= amax:
                        amax = ai
                y[ii,kk] = 0.5 * (a[ii,k,kk] + amax) 
            else:
                y[ii,kk] = <np.float64_t> a[ii,k,kk]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int32 along axis=2."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, jj
    cdef np.int32_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for jj in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[ii,jj,k]
                i = l
                j = r
                while 1:
                    while a[ii,jj,i] < x: i += 1
                    while x < a[ii,jj,j]: j -= 1
                    if i <= j:
                        tmp = a[ii,jj,i]
                        a[ii,jj,i] = a[ii,jj,j]
                        a[ii,jj,j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MININT32
                for i in range(k):
                    ai = a[ii,jj,i]
                    if ai >= amax:
                        amax = ai
                y[ii,jj] = 0.5 * (a[ii,jj,k] + amax) 
            else:
                y[ii,jj] = <np.float64_t> a[ii,jj,k]        
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, jj, kk
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for jj in range(n1):
        for kk in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k,jj,kk]
                i = l
                j = r
                while 1:
                    while a[i,jj,kk] < x: i += 1
                    while x < a[j,jj,kk]: j -= 1
                    if i <= j:
                        tmp = a[i,jj,kk]
                        a[i,jj,kk] = a[j,jj,kk]
                        a[j,jj,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = MININT64
                for i in range(k):
                    ai = a[i,jj,kk]
                    if ai >= amax:
                        amax = ai
                y[jj,kk] = 0.5 * (a[k,jj,kk] + amax) 
            else:
                y[jj,kk] = <np.float64_t> a[k,jj,kk]       
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int64 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, kk
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for kk in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[ii,k,kk]
                i = l
                j = r
                while 1:
                    while a[ii,i,kk] < x: i += 1
                    while x < a[ii,j,kk]: j -= 1
                    if i <= j:
                        tmp = a[ii,i,kk]
                        a[ii,i,kk] = a[ii,j,kk]
                        a[ii,j,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = MININT64
                for i in range(k):
                    ai = a[ii,i,kk]
                    if ai >= amax:
                        amax = ai
                y[ii,kk] = 0.5 * (a[ii,k,kk] + amax) 
            else:
                y[ii,kk] = <np.float64_t> a[ii,k,kk]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.int64 along axis=2."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, jj
    cdef np.int64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for jj in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[ii,jj,k]
                i = l
                j = r
                while 1:
                    while a[ii,jj,i] < x: i += 1
                    while x < a[ii,jj,j]: j -= 1
                    if i <= j:
                        tmp = a[ii,jj,i]
                        a[ii,jj,i] = a[ii,jj,j]
                        a[ii,jj,j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = MININT64
                for i in range(k):
                    ai = a[ii,jj,i]
                    if ai >= amax:
                        amax = ai
                y[ii,jj] = 0.5 * (a[ii,jj,k] + amax) 
            else:
                y[ii,jj] = <np.float64_t> a[ii,jj,k]        
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.float64 along axis=0."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, jj, kk
    cdef np.float64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n1, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for jj in range(n1):
        for kk in range(n2):
            k = n0 >> 1
            l = 0
            r = n0 - 1
            while l < r:
                x = a[k,jj,kk]
                i = l
                j = r
                while 1:
                    while a[i,jj,kk] < x: i += 1
                    while x < a[j,jj,kk]: j -= 1
                    if i <= j:
                        tmp = a[i,jj,kk]
                        a[i,jj,kk] = a[j,jj,kk]
                        a[j,jj,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n0 % 2 == 0:        
                amax = np.NINF
                for i in range(k):
                    ai = a[i,jj,kk]
                    if ai >= amax:
                        amax = ai
                y[jj,kk] = (a[k,jj,kk] + amax) / 2 
            else:
                y[jj,kk] = a[k,jj,kk]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.float64 along axis=1."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, kk
    cdef np.float64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n2]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for kk in range(n2):
            k = n1 >> 1
            l = 0
            r = n1 - 1
            while l < r:
                x = a[ii,k,kk]
                i = l
                j = r
                while 1:
                    while a[ii,i,kk] < x: i += 1
                    while x < a[ii,j,kk]: j -= 1
                    if i <= j:
                        tmp = a[ii,i,kk]
                        a[ii,i,kk] = a[ii,j,kk]
                        a[ii,j,kk] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n1 % 2 == 0:        
                amax = np.NINF
                for i in range(k):
                    ai = a[ii,i,kk]
                    if ai >= amax:
                        amax = ai
                y[ii,kk] = (a[ii,k,kk] + amax) / 2 
            else:
                y[ii,kk] = a[ii,k,kk]         
    return y

@cython.boundscheck(False)
@cython.wraparound(False)
def median_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a):
    "Median of 3d numpy array with dtype=np.float64 along axis=2."
    cdef np.npy_intp i, j, l, r, k, n0, n1, n2, ii, jj
    cdef np.float64_t x, tmp, amax, ai
    n0 = a.shape[0]
    n1 = a.shape[1]
    n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for ii in range(n0):
        for jj in range(n1):
            k = n2 >> 1
            l = 0
            r = n2 - 1
            while l < r:
                x = a[ii,jj,k]
                i = l
                j = r
                while 1:
                    while a[ii,jj,i] < x: i += 1
                    while x < a[ii,jj,j]: j -= 1
                    if i <= j:
                        tmp = a[ii,jj,i]
                        a[ii,jj,i] = a[ii,jj,j]
                        a[ii,jj,j] = tmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
            if n2 % 2 == 0:        
                amax = np.NINF
                for i in range(k):
                    ai = a[ii,jj,i]
                    if ai >= amax:
                        amax = ai
                y[ii,jj] = (a[ii,jj,k] + amax) / 2 
            else:
                y[ii,jj] = a[ii,jj,k]         
    return y
