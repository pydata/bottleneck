"move_median template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_median"]

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
    "Moving median of NDIMd array of dtype=DTYPE along axis=AXIS."
    cdef mm_handle *mm
"""

loop = {}
loop[1] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))
    elif (window == 1):
        if issubclass(a.dtype.type, np.inexact):
            return PyArray_Copy(a)
        else:
            return a.astype(np.float64)
    for iINDEX0 in range(window-1):
        y[INDEXALL] = np.nan
    mm = mm_new(window)
    for iINDEX0 in range(window):
        mm_insert_init(mm, a[INDEXALL])
    y[INDEXREPLACE|window-1|] = mm_get_median(mm)
    for iINDEX0 in range(window, nINDEX0):
        mm_update(mm, a[INDEXALL])
        y[INDEXALL] = mm_get_median(mm)
    mm_free(mm)
    return y
"""
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))
    elif (window == 1):
        if issubclass(a.dtype.type, np.inexact):
            return PyArray_Copy(a)
        else:
            return a.astype(np.float64)
    mm = mm_new(window)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(window-1):
            y[INDEXALL] = np.nan
        for iINDEX1 in range(window):
            mm_insert_init(mm, a[INDEXALL])
        y[INDEXREPLACE|window-1|] = mm_get_median(mm)
        for iINDEX1 in range(window, nINDEX1):
            mm_update(mm, a[INDEXALL])
            y[INDEXALL] = mm_get_median(mm)
        mm.n_s = 0
        mm.n_l = 0
    mm_free(mm)
    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))
    elif (window == 1):
        if issubclass(a.dtype.type, np.inexact):
            return PyArray_Copy(a)
        else:
            return a.astype(np.float64)
    mm = mm_new(window)
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(window-1):
                y[INDEXALL] = np.nan
            for iINDEX2 in range(window):
                mm_insert_init(mm, a[INDEXALL])
            y[INDEXREPLACE|window-1|] = mm_get_median(mm)
            for iINDEX2 in range(window, nINDEX2):
                mm_update(mm, a[INDEXALL])
                y[INDEXALL] = mm_get_median(mm)
            mm.n_s = 0
            mm.n_l = 0
    mm_free(mm)
    return y
"""

floats['loop'] = loop

# Int dtypes (no axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['force_output_dtype'] = 'float64'
ints['dtypes'] = INT_DTYPES

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "move_median"
slow['signature'] = "arr, window"
slow['func'] = "bn.slow.move_median(arr, window, axis=AXIS)"

# Template ------------------------------------------------------------------

move_median = {}
move_median['name'] = 'move_median'
move_median['is_reducing_function'] = False
move_median['cdef_output'] = True
move_median['slow'] = slow
move_median['templates'] = {}
move_median['templates']['float'] = floats
move_median['templates']['int'] = ints
move_median['pyx_file'] = 'move/move_median.pyx'

move_median['main'] = '''"move_median auto-generated from template"

cdef extern from "csrc/move_median.c":
    struct _mm_node:
        np.npy_uint32   small
        np.npy_uint64   idx
        np.npy_float64  val
        _mm_node         *next
    ctypedef _mm_node mm_node
    struct _mm_handle:
        int              odd
        np.npy_uint64    n_s
        np.npy_uint64    n_l
        mm_node          **s_heap
        mm_node          **l_heap
        mm_node          **nodes
        mm_node           *node_data
        mm_node           *first
        mm_node           *last
        np.npy_uint64 s_first_leaf
        np.npy_uint64 l_first_leaf
    ctypedef _mm_handle mm_handle
    mm_handle *mm_new(np.npy_uint64 size)
    void mm_insert_init(mm_handle *mm, np.npy_float64 val)
    void mm_update(mm_handle *mm, np.npy_float64 val)
    np.npy_float64 mm_get_median(mm_handle *mm)
    void mm_free(mm_handle *mm)

def move_median(arr, int window, int axis=-1):
    """
    Moving window median along the specified axis.

    This functions is not protected against NaN. Therefore, you may get
    unexpected results if the input contains NaN.

    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving median. By default the moving
        median is taken over the last axis (axis=-1). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving median of the input array along the specified axis. The
        output has the same shape as the input.

    Notes
    -----
    Unexpected results may occur if the input array contains NaN.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_median(arr, window=2)
    array([ nan,  1.5,  2.5,  3.5])

    """
    func, arr = move_median_selector(arr, axis)
    return func(arr, window)

def move_median_selector(arr, int axis):
    """
    Return move_median function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_median() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving median.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the moving median is to be computed.

    Returns
    -------
    func : function
        The moving median function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the median.
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
    >>> func, a = bn.move.move_median_selector(arr, axis)
    >>> func
    <function move_median_1d_float64_axis0>

    Use the returned function and array to determine the moving median:

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
        func = move_median_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = move_median_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
