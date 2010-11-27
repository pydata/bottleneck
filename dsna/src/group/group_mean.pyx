"group_mean"

# key is (ndim, dtype, axis)
cdef dict group_mean_dict = {}

group_mean_dict[(1, f64, 0)] = group_mean_1d_float64_axis0
group_mean_dict[(2, f64, 0)] = group_mean_2d_float64_axis0
group_mean_dict[(2, f64, 1)] = group_mean_2d_float64_axis1
group_mean_dict[(3, f64, 0)] = group_mean_3d_float64_axis0
group_mean_dict[(3, f64, 1)] = group_mean_3d_float64_axis1
group_mean_dict[(3, f64, 2)] = group_mean_3d_float64_axis2

group_mean_dict[(1, i32, 0)] = group_mean_1d_int32_axis0
group_mean_dict[(2, i32, 0)] = group_mean_2d_int32_axis0
group_mean_dict[(2, i32, 1)] = group_mean_2d_int32_axis1
group_mean_dict[(3, i32, 0)] = group_mean_3d_int32_axis0
group_mean_dict[(3, i32, 1)] = group_mean_3d_int32_axis1
group_mean_dict[(3, i32, 2)] = group_mean_3d_int32_axis2

group_mean_dict[(1, i64, 0)] = group_mean_1d_int64_axis0
group_mean_dict[(2, i64, 0)] = group_mean_2d_int64_axis0
group_mean_dict[(2, i64, 1)] = group_mean_2d_int64_axis1
group_mean_dict[(3, i64, 0)] = group_mean_3d_int64_axis0
group_mean_dict[(3, i64, 1)] = group_mean_3d_int64_axis1
group_mean_dict[(3, i64, 2)] = group_mean_3d_int64_axis2


def group_mean(arr, label, order=None, int axis=0):
    """
    Group means of like-labeled array elements along given axis ignoring NaNs.

    
    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion to an array is
        attempted.
    label : {list, array_like}
        Group membership labels. For example, if the first and last values in
        an array belong to group 'a' and the middle two values belong to
        group 'b', then the label could be ['a', 'b', 'b', 'a'] or the
        equivalent array. Using a list for `label` is faster than using an
        array. The number of labels must match the number of array elements
        along the specified axis.
    order : array_like, optional
        A sequence of group labels that determine the output order of the
        group means. By default (order=None) the output is in sorted order
        of the unique elements in `label`. A list `order` is faster than an
        array `order`.
    axis : int, optional
        Axis along which the group mean is computed. The default is to compute
        the group mean along the first axis. An axis of None is not allowed.

    Returns
    -------
    gm : ndarray
        An array containing the group means. The shape is the same as the
        input array except along `axis` where the number of elements is equal
        to the number of unique labels. `float64` intermediate and return
        values are used.
    unique_label : list
        The unique group labels in sorted order. This is also the order of the
        group mean returned in `gm`.

    Notes
    -----
    No error is raised on overflow. (The sum is computed and then the result
    is divided by the number of non-NaN elements.)

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    Set up the problem:

    >>> from dsna import group_mean
    >>> arr = np.array([1, 2, 3, 9])
    >>> label = ['a', 'b', 'b', 'a']
    
    Find group mean:

    >>> group_mean(arr, label)
    (array([ 5. ,  2.5]), ['a', 'b'])
    >>> group_mean(arr, label, order=['b', 'a'])
    (array([ 2.5,  5. ]), ['b', 'a'])
    >>> group_mean(arr, label, order=['b'])
    (array([ 2.5]), ['b'])
    >>> group_mean(arr, label, order=['c'])
    KeyError: 'c'

    We can also change the type of the input:

    >>> group_mean(arr.tolist(), np.array(label))
    (array([ 5. ,  2.5]), ['a', 'b'])    

    """
    func, arr, label_dict, order = group_mean_selector(arr, label, order, axis)
    return func(arr, label_dict, order)

def group_mean_selector(arr, label, order=None, int axis=0):
    """
    Group mean function, array, and label mapper to use for specified problem.
    
    Under the hood dsna uses a separate Cython function for each combination
    of ndim, dtype, and axis. A lot of the overhead in ds.group_mean() is in
    checking that `axis` is within range, converting `arr` into an array (if
    it is not already an array), and selecting the function to use to
    calculate the group mean.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    label : {list, array_like}
        Group membership labels. For example, if the first and last values in
        an array belong to group 'a' and the middle two values belong to
        group 'b', then the label could be ['a', 'b', 'b', 'a'] or the
        equivalent array. Using a list for `label` is faster than using an
        array. The number of labels must match the number of array elements
        along the specified axis.
    order : array_like, optional
        A sequence of group labels that determine the output order of the
        group means. By default (order=None) the output is in sorted order
        of the unique elements in `label`. A list `order` is faster than an
        array `order`.
    axis : int, optional
        Axis along which the group mean is to be computed. The default is to
        compute the group mean along the first axis (=0).
    
    Returns
    -------
    func : function
        The group mean function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to find
        the group mean.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.
    label_dict : dict
        A dictionary mapping each unique group label (dict keys) to a list of
        index positions (dict values).
    order : list 
        A list of group labels that determine the output order of the
        group means. By default (order=None) the output is in sorted order
        of the unique elements in `label`.
    
    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> from dsna.group import group_mean_selector
    
    Obtain the function, etc. needed to determine the group mean of `arr`
    along axis=0:

    >>> func, a, label_dict, order = group_mean_selector(arr, label, axis=0)

    The output:

    >>> func
    <built-in function group_mean_1d_int64_axis0>
    >>> a
    array([1, 2, 3, 9])
    >>> label_dict
    {'a': [0, 3], 'b': [1, 2]}
    >>> order
    ['a', 'b']
    
    Use the returned items to determine the group mean:

    >>> func(a, label_dict, order)
    (array([ 5. ,  2.5]), ['a', 'b'])

    """
    cdef np.ndarray a = np.array(arr, copy=False)
    cdef int ndim = a.ndim
    cdef np.dtype dtype = a.dtype
    if axis < 0:
       axis += ndim
    if (axis < 0) or (axis >= ndim):
        raise ValueError("axis(=%d) out of bounds" % axis)
    cdef int narr = a.shape[axis], nlabel = len(label)
    if narr != nlabel:
        msg = "Number of labels (=%d) must equal number of elements (=%d) "
        msg += "along axis=%d of `arr`."
        raise ValueError(msg % (nlabel, narr, axis))
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = group_mean_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    label_dict, order = group_mapper(label, order)
    return func, a, label_dict, order

# One dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 1d, int32 numpy array."
    cdef Py_ssize_t i, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], count, norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(norder,
                                                       dtype=np.float64)
    for label in order:
        g += 1
        asum = 0
        count = 0
        idx = label_dict[label]
        for i in idx:
            asum += a[i]
            count += 1
        y[g] = asum / count
    return y, order        

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 1d, int64 numpy array."
    cdef Py_ssize_t i, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], count, norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(norder,
                                                       dtype=np.float64)
    for label in order:
        g += 1
        asum = 0
        count = 0
        idx = label_dict[label]
        for i in idx:
            asum += a[i]
            count += 1
        y[g] = asum / count
    return y, order       

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a,
                                dict label_dict, list order):
    "Group mean along axis=0 of a 1d, float64 numpy array."
    cdef Py_ssize_t i, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], count = 0, norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=1] y = np.empty(norder,
                                                       dtype=np.float64)
    for label in order:
        g += 1
        asum = 0
        count = 0
        idx = label_dict[label]
        for i in idx:
            ai = a[i]
            if ai == ai:
                asum += ai
                count += 1
        if count > 0:
            y[g] = asum / count
        else:
            y[g] = NAN
    return y, order      

# Two dimensional -----------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 3d, int32 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((norder, a1),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i,j]
                count += 1
            y[g,j] = asum / count
    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a,
                              dict label_dict, list order):
    "Group mean along axis=1 of a 2d, int32 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            asum = 0
            count = 0
            for j in idx:
                asum += a[i,j]
                count += 1
            y[i,g] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 2d, int64 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((norder, a1),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i,j]
                count += 1
            y[g,j] = asum / count
    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a,
                              dict label_dict, list order):
    "Group mean along axis=1 of a 2d, int64 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            asum = 0
            count = 0
            for j in idx:
                asum += a[i,j]
                count += 1
            y[i,g] = asum / count
    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                                dict label_dict, list order):
    "Group mean along axis=0 of a 2d, float64 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((norder, a1),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            asum = 0
            count = 0
            for i in idx:
                ai = a[i,j]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[g,j] = asum / count
            else:
                y[g,j] = NAN
    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                              dict label_dict, list order):
    "Group mean along axis=1 of a 2d, float64 numpy array."
    cdef Py_ssize_t i, j, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=2] y = np.empty((a0, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            asum = 0
            count = 0
            for j in idx:
                ai = a[i,j]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[i,g] = asum / count
            else:
                y[i,g] = NAN
    return y, order 

# Three dimensional ---------------------------------------------------------

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 3d, int32 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((norder, a1, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            for k in range(a2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i,j,k]
                    count += 1
                y[g,j,k] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=1 of a 3d, int32 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, norder, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for k in range(a2):
                asum = 0
                count = 0
                for j in idx:
                    asum += a[i,j,k]
                    count += 1
                y[i,g,k] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=2 of a 3d, int32 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, a1, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for j in range(a1):
                asum = 0
                count = 0
                for k in idx:
                    asum += a[i,j,k]
                    count += 1
                y[i,j,g] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=0 of a 3d, int64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((norder, a1, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            for k in range(a2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i,j,k]
                    count += 1
                y[i,g] = asum / count
    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=1 of a 3d, int64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, norder, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for k in range(a2):
                asum = 0
                count = 0
                for j in idx:
                    asum += a[i,j,k]
                    count += 1
                y[i,g,k] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a,
                              dict label_dict, list order):
    "Group mean along axis=2 of a 3d, int64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, a1, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for j in range(a1):
                asum = 0
                count = 0
                for k in idx:
                    asum += a[i,j,k]
                    count += 1
                y[i,j,g] = asum / count
    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a,
                                dict label_dict, list order):
    "Group mean along axis=0 of a 3d, float64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((norder, a1, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for j in range(a1):
            for k in range(a2):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i,j,k]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[g,j,k] = asum / count
                else:
                    y[g,j,k] = NAN

    return y, order 

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a,
                                dict label_dict, list order):
    "Group mean along axis=1 of a 3d, float64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, norder, a2),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for k in range(a2):
                asum = 0
                count = 0
                for j in idx:
                    ai = a[i,j,k]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i,g,k] = asum / count
                else:
                    y[i,g,k] = NAN

    return y, order

@cython.boundscheck(False)
@cython.wraparound(False)
def group_mean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a,
                                dict label_dict, list order):
    "Group mean along axis=2 of a 3d, float64 numpy array."
    cdef Py_ssize_t i, j, k, g = -1
    cdef list idx
    cdef int a0 = a.shape[0], a1 = a.shape[1], a2 = a.shape[2], count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef np.ndarray[np.float64_t, ndim=3] y = np.empty((a0, a1, norder),
                                                       dtype=np.float64)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i in range(a0):
            for j in range(a1):
                asum = 0
                count = 0
                for k in idx:
                    ai = a[i,j,k]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i,j,g] = asum / count
                else:
                    y[i,j,g] = NAN
    return y, order
