"move_min template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_min"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loop ----------------------------------------------------------------------

loop = {}
loop[1] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
    end = ring + window
    last = ring

    minpair = ring
    ai = a[INDEXREPLACE|0|]
    if ai == ai:
        minpair.value = ai
    else:
        minpair.value = MAXfloat64
    minpair.death = window

    count = 0
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai == ai:
            count += 1
        else:
            ai = MAXfloat64
        if iINDEX0 >= window:
            aold = a[INDEXREPLACE|iINDEX0 - window|]
            if aold == aold:
                count -= 1
        if minpair.death == iINDEX0:
            minpair += 1
            if minpair >= end:
                minpair = ring
        if ai <= minpair.value:
            minpair.value = ai
            minpair.death = iINDEX0 + window
            last = minpair
        else:
            while last.value >= ai:
                if last == ring:
                    last = end
                last -= 1
            last += 1
            if last == end:
                last = ring
            last.value = ai
            last.death = iINDEX0 + window
        if count == window:
            y[INDEXALL] = minpair.value
        else:
            y[INDEXALL] = NAN
    for iINDEX0 in range(window - 1):
        y[INDEXALL] = NAN

    stdlib.free(ring)
    return y
"""
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))

    for iINDEX0 in range(nINDEX0):

        end = ring + window
        last = ring

        minpair = ring
        ai = a[INDEXREPLACE|0|]
        if ai == ai:
            minpair.value = ai
        else:
            minpair.value = MAXfloat64
        minpair.death = window

        count = 0
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai == ai:
                count += 1
            else:
                ai = MAXfloat64
            if iINDEX1 >= window:
                aold = a[INDEXREPLACE|iINDEX1 - window|]
                if aold == aold:
                    count -= 1
            if minpair.death == iINDEX1:
                minpair += 1
                if minpair >= end:
                    minpair = ring
            if ai <= minpair.value:
                minpair.value = ai
                minpair.death = iINDEX1 + window
                last = minpair
            else:
                while last.value >= ai:
                    if last == ring:
                        last = end
                    last -= 1
                last += 1
                if last == end:
                    last = ring
                last.value = ai
                last.death = iINDEX1 + window
            if count == window:
                y[INDEXALL] = minpair.value
            else:
                y[INDEXALL] = NAN
        for iINDEX1 in range(window - 1):
            y[INDEXALL] = NAN

    stdlib.free(ring)
    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError(MOVE_WINDOW_ERR_MSG % (window, nAXIS))

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))

    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            end = ring + window
            last = ring

            minpair = ring
            ai = a[INDEXREPLACE|0|]
            if ai == ai:
                minpair.value = ai
            else:
                minpair.value = MAXfloat64
            minpair.death = window

            count = 0
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai == ai:
                    count += 1
                else:
                    ai = MAXfloat64
                if iINDEX2 >= window:
                    aold = a[INDEXREPLACE|iINDEX2 - window|]
                    if aold == aold:
                        count -= 1
                if minpair.death == iINDEX2:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                if ai <= minpair.value:
                    minpair.value = ai
                    minpair.death = iINDEX2 + window
                    last = minpair
                else:
                    while last.value >= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = iINDEX2 + window
                if count == window:
                    y[INDEXALL] = minpair.value
                else:
                    y[INDEXALL] = NAN
            for iINDEX2 in range(window - 1):
                y[INDEXALL] = NAN

    stdlib.free(ring)
    return y
"""

# Float dtypes (no axis=None) -----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'float64'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a, int window):
    "Moving min of NDIMd array of dtype=DTYPE along axis=AXIS."
    cdef np.float64_t ai, aold
    cdef Py_ssize_t count
    cdef pairs* ring
    cdef pairs* minpair
    cdef pairs* end
    cdef pairs* last
"""

floats['loop'] = loop

# Int dtypes (no axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['force_output_dtype'] = 'float64'
ints['dtypes'] = INT_DTYPES
ints['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "move_min"
slow['signature'] = "arr, window"
slow['func'] = "bn.slow.move_min(arr, window, axis=AXIS)"

# Template ------------------------------------------------------------------

move_min = {}
move_min['name'] = 'move_min'
move_min['is_reducing_function'] = False
move_min['cdef_output'] = True
move_min['slow'] = slow
move_min['templates'] = {}
move_min['templates']['float'] = floats
move_min['templates']['int'] = ints
move_min['pyx_file'] = 'move/move_min.pyx'

move_min['main'] = '''"move_min auto-generated from template"

# The minimum on a sliding window algorithm by Richard Harter
# http://home.tiac.net/~cri/2001/slidingmin.html
# Original C code:
# Copyright Richard Harter 2009
# Released under a Simplified BSD license
#
# Adapted and expanded for Bottleneck:
# Copyright 2010 Keith Goodman
# Released under the Bottleneck license

def move_min(arr, int window, int axis=-1):
    """
    Moving window minimum along the specified axis.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to find the moving minimum. By default the moving
        minimum is taken over the last axis (axis=-1). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving minimum of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 4.0, 3.0])
    >>> bn.move_min(arr, window=2)
    array([ nan,  1.,  2.,  3.])

    """
    func, arr = move_min_selector(arr, axis)
    return func(arr, window)

def move_min_selector(arr, int axis):
    """
    Return move_min function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_min() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving minimum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the moving minimum is to be computed.

    Returns
    -------
    func : function
        The moving minimum function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the minimum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 4.0, 3.0])

    Obtain the function needed to determine the sum of `arr` along axis=0:

    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_min_selector(arr, axis)
    >>> func
    <function move_min_1d_float64_axis0>

    Use the returned function and array to determine the moving minimum:

    >>> func(a, window)
    array([ nan,  1.,  2.,  3.])

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
        func = move_min_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = move_min_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
