"group_nanmean auto-generated from template"

def group_nanmean(arr, label, order=None, int axis=0):
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

    >>> from bottleneck import group_nanmean
    >>> arr = np.array([1, 2, 3, 9])
    >>> label = ['a', 'b', 'b', 'a']
    
    Find group nanmean:

    >>> group_nanmean(arr, label)
    (array([ 5. ,  2.5]), ['a', 'b'])
    >>> group_nanmean(arr, label, order=['b', 'a'])
    (array([ 2.5,  5. ]), ['b', 'a'])
    >>> group_nanmean(arr, label, order=['b'])
    (array([ 2.5]), ['b'])
    >>> group_nanmean(arr, label, order=['c'])
    KeyError: 'c'

    We can also change the type of the input:

    >>> group_nanmean(arr.tolist(), np.array(label))
    (array([ 5. ,  2.5]), ['a', 'b'])    

    """
    func, arr, label_dict, order = group_nanmean_selector(arr, label, order,
                                                       axis)
    return func(arr, label_dict, order)

def group_nanmean_selector(arr, label, order=None, int axis=0):
    """
    Group nanmean function, array, and label mapper to use for specified
    problem.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.group_nanmean() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the group mean.

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
        The group nanmean function that matches the number of dimensions and
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
    >>> from bottleneck.group import group_nanmean_selector
    
    Obtain the function, etc. needed to determine the group mean of `arr`
    along axis=0:

    >>> func, a, label_dict, order = group_nanmean_selector(arr, label, axis=0)

    The output:

    >>> func
    <built-in function group_nanmean_1d_int64_axis0>
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
        func = group_nanmean_dict[key]
    except KeyError:
        tup = (str(ndim), str(dtype))
        raise TypeError("Unsupported ndim/dtype (%s/%s)." % tup)
    label_dict, order = group_mapper(label, order)
    return func, a, label_dict, order

def group_nanmean_1d_int32_axis0(np.ndarray[np.int32_t, ndim=1] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [norder]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        asum = 0
        count = 0
        idx = label_dict[label]
        for i in idx:
            asum += a[i]
            count += 1
        y[g] = <np.float64_t> asum / count
    return y, order 

def group_nanmean_1d_int64_axis0(np.ndarray[np.int64_t, ndim=1] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [norder]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        asum = 0
        count = 0
        idx = label_dict[label]
        for i in idx:
            asum += a[i]
            count += 1
        y[g] = <np.float64_t> asum / count
    return y, order 

def group_nanmean_2d_int32_axis0(np.ndarray[np.int32_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [norder, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i, i1]
                count += 1
            y[g, i1] = <np.float64_t> asum / count
    return y, order

def group_nanmean_2d_int32_axis1(np.ndarray[np.int32_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, norder]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i0, i]
                count += 1
            y[i0, g] = <np.float64_t> asum / count
    return y, order

def group_nanmean_2d_int64_axis0(np.ndarray[np.int64_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [norder, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i, i1]
                count += 1
            y[g, i1] = <np.float64_t> asum / count
    return y, order

def group_nanmean_2d_int64_axis1(np.ndarray[np.int64_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, norder]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            asum = 0
            count = 0
            for i in idx:
                asum += a[i0, i]
                count += 1
            y[i0, g] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int32_axis0(np.ndarray[np.int32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [norder, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i, i1, i2]
                    count += 1
                y[g, i1, i2] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int32_axis1(np.ndarray[np.int32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, norder, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i0, i, i2]
                    count += 1
                y[i0, g, i2] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int32_axis2(np.ndarray[np.int32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, norder]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i1 in range(n1):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i0, i1, i]
                    count += 1
                y[i0, i1, g] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int64_axis0(np.ndarray[np.int64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [norder, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i, i1, i2]
                    count += 1
                y[g, i1, i2] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int64_axis1(np.ndarray[np.int64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, norder, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i0, i, i2]
                    count += 1
                y[i0, g, i2] = <np.float64_t> asum / count
    return y, order

def group_nanmean_3d_int64_axis2(np.ndarray[np.int64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.int64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, norder]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i1 in range(n1):
                asum = 0
                count = 0
                for i in idx:
                    asum += a[i0, i1, i]
                    count += 1
                y[i0, i1, g] = <np.float64_t> asum / count
    return y, order

def group_nanmean_1d_float32_axis0(np.ndarray[np.float32_t, ndim=1] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [norder]
    cdef np.ndarray[np.float32_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float32, 0)
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

def group_nanmean_1d_float64_axis0(np.ndarray[np.float64_t, ndim=1] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0
    cdef int n0 = a.shape[0]
    cdef np.npy_intp *dims = [norder]
    cdef np.ndarray[np.float64_t, ndim=1] y = PyArray_EMPTY(1, dims,
                                                            NPY_float64, 0)
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

def group_nanmean_2d_float32_axis0(np.ndarray[np.float32_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [norder, n1]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float32, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            asum = 0
            count = 0
            for i in idx:
                ai = a[i, i1]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[g, i1] = asum / count
            else:
                y[g, i1] = NAN
    return y, order

def group_nanmean_2d_float32_axis1(np.ndarray[np.float32_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, norder]
    cdef np.ndarray[np.float32_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float32, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            asum = 0
            count = 0
            for i in idx:
                ai = a[i0, i]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[i0, g] = asum / count
            else:
                y[i0, g] = NAN
    return y, order

def group_nanmean_2d_float64_axis0(np.ndarray[np.float64_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [norder, n1]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            asum = 0
            count = 0
            for i in idx:
                ai = a[i, i1]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[g, i1] = asum / count
            else:
                y[g, i1] = NAN
    return y, order

def group_nanmean_2d_float64_axis1(np.ndarray[np.float64_t, ndim=2] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef np.npy_intp *dims = [n0, norder]
    cdef np.ndarray[np.float64_t, ndim=2] y = PyArray_EMPTY(2, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            asum = 0
            count = 0
            for i in idx:
                ai = a[i0, i]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:        
                y[i0, g] = asum / count
            else:
                y[i0, g] = NAN
    return y, order

def group_nanmean_3d_float32_axis0(np.ndarray[np.float32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [norder, n1, n2]
    cdef np.ndarray[np.float32_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float32, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i, i1, i2]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[g, i1, i2] = asum / count
                else:
                    y[g, i1, i2] = NAN

    return y, order 

def group_nanmean_3d_float32_axis1(np.ndarray[np.float32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, norder, n2]
    cdef np.ndarray[np.float32_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float32, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i0, i, i2]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i0, g, i2] = asum / count
                else:
                    y[i0, g, i2] = NAN

    return y, order 

def group_nanmean_3d_float32_axis2(np.ndarray[np.float32_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float32_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, norder]
    cdef np.ndarray[np.float32_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float32, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i1 in range(n1):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i0, i1, i]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i0, i1, g] = asum / count
                else:
                    y[i0, i1, g] = NAN

    return y, order 

def group_nanmean_3d_float64_axis0(np.ndarray[np.float64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [norder, n1, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i1 in range(n1):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i, i1, i2]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[g, i1, i2] = asum / count
                else:
                    y[g, i1, i2] = NAN

    return y, order 

def group_nanmean_3d_float64_axis1(np.ndarray[np.float64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, norder, n2]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i2 in range(n2):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i0, i, i2]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i0, g, i2] = asum / count
                else:
                    y[i0, g, i2] = NAN

    return y, order 

def group_nanmean_3d_float64_axis2(np.ndarray[np.float64_t, ndim=3] a,
                                           dict label_dict, list order):
    "Group mean along axis=0 of 2d, float64 array ignoring NaNs."
    cdef Py_ssize_t g = -1
    cdef list idx
    cdef int count
    cdef int norder = len(order)
    cdef np.float64_t asum = 0, ai
    cdef Py_ssize_t i0, i1, i2
    cdef int n0 = a.shape[0]
    cdef int n1 = a.shape[1]
    cdef int n2 = a.shape[2]
    cdef np.npy_intp *dims = [n0, n1, norder]
    cdef np.ndarray[np.float64_t, ndim=3] y = PyArray_EMPTY(3, dims,
                                                            NPY_float64, 0)
    for label in order:
        g += 1
        idx = label_dict[label]
        for i0 in range(n0):
            for i1 in range(n1):
                asum = 0
                count = 0
                for i in idx:
                    ai = a[i0, i1, i]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:        
                    y[i0, i1, g] = asum / count
                else:
                    y[i0, i1, g] = NAN

    return y, order 

cdef dict group_nanmean_dict = {}
group_nanmean_dict[(1, int32, 0)] = group_nanmean_1d_int32_axis0
group_nanmean_dict[(1, int64, 0)] = group_nanmean_1d_int64_axis0
group_nanmean_dict[(2, int32, 0)] = group_nanmean_2d_int32_axis0
group_nanmean_dict[(2, int32, 1)] = group_nanmean_2d_int32_axis1
group_nanmean_dict[(2, int64, 0)] = group_nanmean_2d_int64_axis0
group_nanmean_dict[(2, int64, 1)] = group_nanmean_2d_int64_axis1
group_nanmean_dict[(3, int32, 0)] = group_nanmean_3d_int32_axis0
group_nanmean_dict[(3, int32, 1)] = group_nanmean_3d_int32_axis1
group_nanmean_dict[(3, int32, 2)] = group_nanmean_3d_int32_axis2
group_nanmean_dict[(3, int64, 0)] = group_nanmean_3d_int64_axis0
group_nanmean_dict[(3, int64, 1)] = group_nanmean_3d_int64_axis1
group_nanmean_dict[(3, int64, 2)] = group_nanmean_3d_int64_axis2
group_nanmean_dict[(1, float32, 0)] = group_nanmean_1d_float32_axis0
group_nanmean_dict[(1, float64, 0)] = group_nanmean_1d_float64_axis0
group_nanmean_dict[(2, float32, 0)] = group_nanmean_2d_float32_axis0
group_nanmean_dict[(2, float32, 1)] = group_nanmean_2d_float32_axis1
group_nanmean_dict[(2, float64, 0)] = group_nanmean_2d_float64_axis0
group_nanmean_dict[(2, float64, 1)] = group_nanmean_2d_float64_axis1
group_nanmean_dict[(3, float32, 0)] = group_nanmean_3d_float32_axis0
group_nanmean_dict[(3, float32, 1)] = group_nanmean_3d_float32_axis1
group_nanmean_dict[(3, float32, 2)] = group_nanmean_3d_float32_axis2
group_nanmean_dict[(3, float64, 0)] = group_nanmean_3d_float64_axis0
group_nanmean_dict[(3, float64, 1)] = group_nanmean_3d_float64_axis1
group_nanmean_dict[(3, float64, 2)] = group_nanmean_3d_float64_axis2