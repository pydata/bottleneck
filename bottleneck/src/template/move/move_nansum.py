"move_nansum template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_nansum"]

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
                                  int window):
    "Moving sum of NDIMd array of dtype=DTYPE along axis=AXIS, ignoring NaNs."
    cdef Py_ssize_t count = 0
    cdef double asum = 0, ai, aold
"""

loop = {}
loop[1] = """\
    if (window < 1) or (window > nINDEX0):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nINDEX0))

    for iINDEX0 in range(window - 1):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            count += 1
        y[INDEXALL] = NAN
    iINDEX0 = window - 1
    ai = a[INDEXALL]
    if ai == ai:
        asum += ai
        count += 1
    if count > 0:
       y[INDEXALL] = asum
    else:
       y[INDEXALL] = NAN
    for iINDEX0 in range(window, nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            count += 1
        aold = a[INDEXREPLACE|iAXIS - window|]
        if aold == aold:
            asum -= aold
            count -= 1
        if count > 0:
            y[INDEXALL] = asum
        else:
            y[INDEXALL] = NAN

    return y
"""
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))

    for iINDEX0 in range(nINDEX0):
        asum = 0
        count = 0
        for iINDEX1 in range(window - 1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
            y[INDEXALL] = NAN
        iINDEX1 = window - 1
        ai = a[INDEXALL]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
           y[INDEXALL] = asum
        else:
           y[INDEXALL] = NAN
        for iINDEX1 in range(window, nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
            aold = a[INDEXREPLACE|iAXIS - window|]
            if aold == aold:
                asum -= aold
                count -= 1
            if count > 0:
                y[INDEXALL] = asum
            else:
                y[INDEXALL] = NAN

    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))

    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            asum = 0
            count = 0
            for iINDEX2 in range(window - 1):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    count += 1
                y[INDEXALL] = NAN
            iINDEX2 = window - 1
            ai = a[INDEXALL]
            if ai == ai:
                asum += ai
                count += 1
            if count > 0:
               y[INDEXALL] = asum
            else:
               y[INDEXALL] = NAN
            for iINDEX2 in range(window, nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = a[INDEXREPLACE|iAXIS - window|]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count > 0:
                    y[INDEXALL] = asum
                else:
                    y[INDEXALL] = NAN

    return y
"""

floats['loop'] = loop

# Int dtypes (no axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['reuse_non_nan_func'] = True
ints['dtypes'] = INT_DTYPES

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "move_nansum"
slow['signature'] = "arr, window"
slow['func'] = "bn.slow.move_nansum(arr, window, axis=AXIS)"

# Template ------------------------------------------------------------------

move_nansum = {}
move_nansum['name'] = 'move_nansum'
move_nansum['is_reducing_function'] = False
move_nansum['cdef_output'] = True
move_nansum['slow'] = slow
move_nansum['templates'] = {}
move_nansum['templates']['float'] = floats
move_nansum['templates']['int'] = ints
move_nansum['pyx_file'] = 'move/move_nansum.pyx'

move_nansum['main'] = '''"move_nansum auto-generated from template"

def move_nansum(arr, int window, int axis=-1):
    """
    Moving window sum along the specified axis, ignoring NaNs.

    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving sum. By default the moving
        sum is taken over the last axis (axis=-1). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving sum of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_nansum(arr, window=2)
    array([ nan,  3.,  5.,  7.])

    """
    func, arr = move_nansum_selector(arr, axis)
    return func(arr, window)

def move_nansum_selector(arr, int axis):
    """
    Return move_nansum function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_nansum() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the moving sum is to be computed.

    Returns
    -------
    func : function
        The moving nansum function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the sum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])

    Obtain the function needed to determine the nansum of `arr` along axis=0:

    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_nansum_selector(arr, axis)
    >>> func
    <function move_nansum_1d_float64_axis0>

    Use the returned function and array to determine the moving nansum:

    >>> func(a, window)
    array([ nan,  3.,  5.,  7.])

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
    if axis < 0:
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = move_nansum_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = move_nansum_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
