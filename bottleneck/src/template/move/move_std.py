"move_std template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_std"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]


# Float dtypes (no axis=None) -----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = False
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                                  int window, int ddof):
    "Moving std of NDIMd array of dtype=DTYPE along axis=AXIS."
    cdef int count = 0
    cdef double asum = 0, a2sum = 0, ai
"""

loop = {}
loop[1] = """\
    if (window < 1) or (window > nINDEX0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nINDEX0)

    for iINDEX0 in range(window - 1):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            a2sum += ai * ai
            count += 1
        y[INDEXALL] = NAN
    iINDEX0 = window - 1
    ai = a[INDEXALL]
    if ai == ai:
        asum += ai
        a2sum += ai * ai
        count += 1
    if count == window:
       y[INDEXALL] = sqrt((a2sum - asum * asum / count) / (count - ddof))
    else:
       y[INDEXALL] = NAN
    for iINDEX0 in range(window, nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            a2sum += ai * ai
            count += 1
        ai = a[INDEXREPLACE|iAXIS - window|]
        if ai == ai:
            asum -= ai
            a2sum -= ai * ai
            count -= 1
        if count == window:
            y[INDEXALL] = sqrt((a2sum - asum * asum / count) / (count - ddof))
        else:
            y[INDEXALL] = NAN

    return y
"""        
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    for iINDEX0 in range(nINDEX0):
        asum = 0
        a2sum = 0
        count = 0
        for iINDEX1 in range(window - 1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            y[INDEXALL] = NAN
        iINDEX1 = window - 1
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            a2sum += ai * ai
            count += 1
        if count == window:
           y[INDEXALL] = sqrt((a2sum - asum * asum / count) / (count - ddof))
        else:
           y[INDEXALL] = NAN
        for iINDEX1 in range(window, nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            ai = a[INDEXREPLACE|iAXIS - window|]
            if ai == ai:
                asum -= ai
                a2sum -= ai * ai
                count -= 1
            if count == window:
                y[INDEXALL] = sqrt((a2sum - asum * asum / count) \
                              / (count - ddof))
            else:
                y[INDEXALL] = NAN

    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            a2sum = 0
            count = 0
            for iINDEX2 in range(window - 1):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    a2sum += ai * ai
                    count += 1
                y[INDEXALL] = NAN
            iINDEX2 = window - 1
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            if count == window:
               y[INDEXALL] = sqrt((a2sum - asum * asum / count) \
                             / (count - ddof))
            else:
               y[INDEXALL] = NAN
            for iINDEX2 in range(window, nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    a2sum += ai * ai
                    count += 1
                ai = a[INDEXREPLACE|iAXIS - window|]
                if ai == ai:
                    asum -= ai
                    a2sum -= ai * ai
                    count -= 1
                if count == window:
                    y[INDEXALL] = sqrt((a2sum - asum * asum / count) \
                                  / (count - ddof))
                else:
                    y[INDEXALL] = NAN

    return y
"""

floats['loop'] = loop

# Int dtypes (no axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['force_output_dtype'] = 'float64'
ints['dtypes'] = INT_DTYPES
ints['top'] += "    cdef int winddof\n"

loop = {}
loop[1] = """\
    if (window < 1) or (window > nINDEX0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nINDEX0)

    winddof = window - ddof
    for iINDEX0 in range(window - 1):
        ai = a[INDEXALL]
        asum += ai
        a2sum += ai * ai
        y[INDEXALL] = NAN
    iINDEX0 = window - 1
    ai = a[INDEXALL]
    asum += ai
    a2sum += ai * ai
    y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)
    for iINDEX0 in range(window, nINDEX0):
        ai = a[INDEXALL]
        asum += ai
        a2sum += ai * ai
        ai = a[INDEXREPLACE|iAXIS - window|]
        asum -= ai
        a2sum -= ai * ai
        y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)

    return y
"""        
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)
    
    winddof = window - ddof
    for iINDEX0 in range(nINDEX0):
        asum = 0
        a2sum = 0
        for iINDEX1 in range(window - 1):
            ai = a[INDEXALL]
            asum += ai
            a2sum += ai * ai
            y[INDEXALL] = NAN
        iINDEX1 = window - 1
        ai = a[INDEXALL]
        asum += ai
        a2sum += ai * ai
        y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)
        for iINDEX1 in range(window, nINDEX1):
            ai = a[INDEXALL]
            asum += ai
            a2sum += ai * ai
            ai = a[INDEXREPLACE|iAXIS - window|]
            asum -= ai
            a2sum -= ai * ai
            y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)

    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    winddof = window - ddof
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            a2sum = 0
            for iINDEX2 in range(window - 1):
                ai = a[INDEXALL]
                asum += ai
                a2sum += ai * ai
                y[INDEXALL] = NAN
            iINDEX2 = window - 1
            ai = a[INDEXALL]
            asum += ai
            a2sum += ai * ai
            y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)
            for iINDEX2 in range(window, nINDEX2):
                ai = a[INDEXALL]
                asum += ai
                a2sum += ai * ai
                ai = a[INDEXREPLACE|iAXIS - window|]
                asum -= ai
                a2sum -= ai * ai
                y[INDEXALL] = sqrt((a2sum - asum * asum / window) / winddof)

    return y
"""

ints['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "move_std"
slow['signature'] = "arr, window, ddof"
slow['func'] = "bn.slow.move_std(arr, window, axis=AXIS, ddof=ddof)"    

# Template ------------------------------------------------------------------

move_std = {}
move_std['name'] = 'move_std'
move_std['is_reducing_function'] = False
move_std['cdef_output'] = True
move_std['slow'] = slow
move_std['templates'] = {}
move_std['templates']['float'] = floats
move_std['templates']['int'] = ints
move_std['pyx_file'] = 'move/move_std.pyx'

move_std['main'] = '''"move_std auto-generated from template"

def move_std(arr, int window, int axis=-1, int ddof=0):
    """
    Moving window standard deviation along the specified axis.

    Unlike bn.nanstd, which uses a more rubust two-pass algorithm, move_std
    uses a faster one-pass algorithm.

    An example of a one-pass algorithm:

        >>> np.sqrt((arr*arr).mean() - arr.mean()**2)
    
    An example of a two-pass algorithm:    
    
        >>> np.sqrt(((arr - arr.mean())**2).mean())

    Note in the two-pass algorithm the mean must be found (first pass) before
    the squared deviation (second pass) can be found.

    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving standard deviation. By
        default the moving standard deviation is taken over the last axis
        (axis=-1). An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving standard deviation of the input array along the specified
        axis. The output has the same shape as the input. 

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_std(arr, window=2)
    array([ nan,  1.5,  2.5,  3.5])

    """
    func, arr = move_std_selector(arr, axis)
    return func(arr, window, ddof)

def move_std_selector(arr, int axis):
    """
    Return move_std function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_std() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving standard deviation.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the moving standard deviation is to be computed.
        The default (axis=0) is to compute the moving standard deviation
        along the first axis.
    
    Returns
    -------
    func : function
        The moving standard deviation function that matches the number of
        dimensions, dtype, and the axis along which you wish to find the
        standard deviation.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    
    Obtain the function needed to determine the sum of `arr` along axis=0:
    
    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_std_selector(arr, axis)
    >>> func
    <built-in function move_std_1d_float64_axis0>    
    
    Use the returned function and array to determine the moving std:

    >>> func(a, window)
    array([ nan,  1.5,  2.5,  3.5])

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:    
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis < 0:
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = move_std_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError, "axis(=%d) out of bounds" % axis
        try:
            func = move_std_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
