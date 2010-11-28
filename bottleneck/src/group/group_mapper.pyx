"group label mapping"

# key is dtype
cdef dict group_mapper_dict = {}
group_mapper_dict[i32] = group_mapper_int32
group_mapper_dict[i64] = group_mapper_int64
group_mapper_dict[f64] = group_mapper_float64
group_mapper_dict['list'] = group_mapper_list


def group_mapper(label, order=None):
    """
    Map group membership labels to index positions
    
    Given a list (faster) or an array (slower) of group membership labels,
    returns a dictionary with the unique labels as keys and a list of index
    position for values.

    Parameters
    ----------
    label : {list, array_like}
        Input labels. Mapping is faster if `label` is a list. If `label` is
        not a list the an attempt to convert it to an array is made. Some
        array dtypes have been accelerated (int32, int64, float64) but are
        still slower than using a list.
    order : array_like, optional
        A sequence of group labels that determine the output order of the
        grouped values. By default (order=None) the output is in sorted order
        of the unique elements in `label`. A list `order` is faster than an
        array `order`.

    Returns
    -------
    label_dict : dict
        A dictionary mapping labels (dict keys) to a list of index positions
        of the group members (dict values). 
    order : list
        A list of group labels that determine the output order of the
        grouped values.

    Examples
    --------
    >>> from bottleneck.group import group_mapper
    >>> group_mapper([1, 2, 1, 2])
    {1: [0, 2], 2: [1, 3]}
    >>> group_mapper(['1', '2', '1', '2'])
    {'1': [0, 2], '2': [1, 3]}
    >>> group_mapper(np.array([1, 2, 3, 4]))
    {1L: [0], 2L: [1], 3L: [2], 4L: [3]}
    >>> group_mapper(np.array([1.0, 2.0, 3.0, 4.0]))
    {1.0: [0], 2.0: [1], 3.0: [2], 4.0: [3]}

    """
    func, lab = group_mapper_selector(label)
    label_dict, order_default = func(lab)
    if order is None:
        order = order_default
        order_default.sort()
    elif isinstance(order, list):
        pass
    elif isinstance(order, np.ndarray):
        order = order.tolist()
    else:
        raise TypeError("`order` must be a list, ndarray, or None.")
    return label_dict, order
    
def group_mapper_selector(label):
    """
    Return group mapper function and label that matches `label`.
    
    Parameters
    ----------
    label : {list, array_like}
        Group membership labels. For example, if the first and last values in
        an array belong to group 'a' and the middle two values belong to
        group 'b', then the label could be ['a', 'b', 'b', 'a'] or the
        equivalent array version np.array(['a', 'b', 'b', 'a']). Using a list
        for `label` is faster than using an array.
    
    Returns
    -------
    func : func
        The function to use in making a mapping dictionary from the `label`.
    lab : {list, ndarray}
        A possibly converted version of `label`. If `label` is a list or
        array, then no conversion is made. But if it is neither then an
        attempt is mode to convert it to an array.

    Examples
    --------
    >>> from bottleneck.group import group_mapper_selector
    >>> group_mapper_selector([1, 2, 1, 2])
    (<built-in function group_mapper_list>, [1, 2, 1, 2])
    >>> group_mapper_selector((1, 2, 1, 2))
    (<built-in function group_mapper_int64>, array([1, 2, 1, 2]))
    >>> group_mapper_selector(np.array([1, 2, 1, 2]))
    (<built-in function group_mapper_int64>, array([1, 2, 1, 2]))
    >>> group_mapper_selector(np.array([1.0, 2.0, 1.0, 2.0]))
    (<built-in function group_mapper_float64>, array([ 1.,  2.,  1.,  2.]))

    """
    cdef int ndim
    if isinstance(label, list):
        dtype = 'list'
        lab = label
        ndim = 1
    else:    
        lab = np.array(label, copy=False)
        dtype = lab.dtype
        ndim = lab.ndim
        if ndim != 1:
            raise ValueError("Input must be 1d.")
    try:
        func = group_mapper_dict[dtype]
    except KeyError:
        func = group_mapper_array
    return func, lab

def group_mapper_list(list label):
    "Map group membership labels (list) to index positions (dict)."
    cdef dict d = {}
    cdef list order = []
    cdef int count = 0
    for i in label:
        if i in d:
            d[i].append(count)
        else:
            d[i] = [count]
            order.append(i)
        count += 1
    return d, order

def group_mapper_int32(np.ndarray[np.int32_t, ndim=1] label):
    "Map group membership labels (array) to index positions (dict)."
    cdef Py_ssize_t i
    cdef dict d = {}
    cdef list order = []
    cdef int count = 0, n = label.size
    for i in range(n):
        li = label[i]
        if li in d:
            d[li].append(count)
        else:
            d[li] = [count]
            order.append(li)
        count += 1
    return d, order

def group_mapper_int64(np.ndarray[np.int64_t, ndim=1] label):
    "Map group membership labels (array) to index positions (dict)."
    cdef Py_ssize_t i
    cdef dict d = {}
    cdef list order = []
    cdef int count = 0, n = label.size
    for i in range(n):
        li = label[i]
        if li in d:
            d[li].append(count)
        else:
            d[li] = [count]
            order.append(li)
        count += 1
    return d, order

def group_mapper_float64(np.ndarray[np.float64_t, ndim=1] label):
    "Map group membership labels (array) to index positions (dict)."
    cdef Py_ssize_t i
    cdef dict d = {}
    cdef list order = []
    cdef int count = 0, n = label.size
    for i in range(n):
        li = label[i]
        if li in d:
            d[li].append(count)
        else:
            d[li] = [count]
            order.append(li)
        count += 1
    return d, order

def group_mapper_array(np.ndarray label):
    "Map group membership labels (array) to index positions (dict)."
    cdef Py_ssize_t i
    cdef dict d = {}
    cdef list order = []
    cdef int count = 0, n = label.size
    for i in range(n):
        li = label[i]
        if li in d:
            d[li].append(count)
        else:
            d[li] = [count]
            order.append(li)
        count += 1
    return d, order
