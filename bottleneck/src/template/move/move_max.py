"move_max template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_max"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loop ----------------------------------------------------------------------

loop = {}
loop[1] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
    end = ring + window
    last = ring
    minpair = ring
    minpair.value = a[INDEXREPLACE|0|]
    minpair.death = window
    y[INDEXREPLACE|0|] = a[INDEXREPLACE|0|]

    for iINDEX0 in range(nINDEX0):
        if minpair.death == iINDEX0:
            minpair += 1
            if minpair >= end:
                minpair = ring
        ai = a[INDEXALL]
        if ai >= minpair.value:
            minpair.value = ai
            minpair.death = iINDEX0 + window
            last = minpair
        else:
            while last.value <= ai:
                if last == ring:
                    last = end
                last -= 1
            last += 1
            if last == end:
                last = ring
            last.value = ai
            last.death = iINDEX0 + window
        y[INDEXALL] = minpair.value
    for iINDEX0 in range(window - 1):
        y[INDEXALL] = NAN
    
    stdlib.free(ring)
    return y
"""        
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))

    for iINDEX0 in range(nINDEX0):
    
        end = ring + window
        last = ring
        minpair = ring
        minpair.value = a[INDEXREPLACE|0|]
        minpair.death = window
        y[INDEXREPLACE|0|] = a[INDEXREPLACE|0|]

        for iINDEX1 in range(nINDEX1):
            if minpair.death == iINDEX1:
                minpair += 1
                if minpair >= end:
                    minpair = ring
            ai = a[INDEXALL]
            if ai >= minpair.value:
                minpair.value = ai
                minpair.death = iINDEX1 + window
                last = minpair
            else:
                while last.value <= ai:
                    if last == ring:
                        last = end
                    last -= 1
                last += 1
                if last == end:
                    last = ring
                last.value = ai
                last.death = iINDEX1 + window
            y[INDEXALL] = minpair.value
        for iINDEX1 in range(window - 1):
            y[INDEXALL] = NAN
    
    stdlib.free(ring)
    return y
"""
loop[3] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

    ring = <pairs*>stdlib.malloc(window * sizeof(pairs))

    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):    
            end = ring + window
            last = ring
            minpair = ring
            minpair.value = a[INDEXREPLACE|0|]
            minpair.death = window
            y[INDEXREPLACE|0|] = a[INDEXREPLACE|0|]

            for iINDEX2 in range(nINDEX2):
                if minpair.death == iINDEX2:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                ai = a[INDEXALL]
                if ai >= minpair.value:
                    minpair.value = ai
                    minpair.death = iINDEX2 + window
                    last = minpair
                else:
                    while last.value <= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = iINDEX2 + window
                y[INDEXALL] = minpair.value
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

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a, int window):
    "Moving max of NDIMd array of dtype=DTYPE along axis=AXIS."
    cdef np.float64_t ai
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
slow['name'] = "move_max"
slow['signature'] = "arr, window"
slow['func'] = "bn.slow.move_max(arr, window, axis=AXIS)"

# Template ------------------------------------------------------------------

move_max = {}
move_max['name'] = 'move_max'
move_max['is_reducing_function'] = False
move_max['cdef_output'] = True
move_max['slow'] = slow
move_max['templates'] = {}
move_max['templates']['float'] = floats
move_max['templates']['int'] = ints
move_max['pyx_file'] = 'move/move_max.pyx'

move_max['main'] = '''"move_max auto-generated from template"

# The minimum on a sliding window algorithm by Richard Harter
# http://home.tiac.net/~cri/2001/slidingmin.html
# Original C code:
# Copyright Richard Harter 2009
# Released under a Simplified BSD license 
#
# Adapted and expanded for Bottleneck:
# Copyright 2010 Keith Goodman
# Released under the Bottleneck license

def move_max(arr, int window, int axis=0):
    """
    Moving window maximum along the specified axis.
    
    float64 output is returned for all input data types.  
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to find the moving maximum. By default the moving
        maximum is taken over the first axis (axis=0). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving maximum of the input array along the specified axis. The
        output has the same shape as the input. 

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 4.0, 3.0])
    >>> bn.move_max(arr, window=2)
    array([ nan,  2.,  4.,  4.])

    """
    func, arr = move_max_selector(arr, window, axis)
    return func(arr, window)

def move_max_selector(arr, int window, int axis):
    """
    Return move_max function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_max() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving maximum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the moving maximum is to be computed. The default
        (axis=0) is to compute the moving maximum along the first axis.
    
    Returns
    -------
    func : function
        The moving maximum function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the maximum.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarra; otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 4.0, 3.0])
    
    Obtain the function needed to determine the sum of `arr` along axis=0:
    
    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_max_selector(arr, window=2, axis=0)
    >>> func
    <built-in function move_max_1d_float64_axis0>    
    
    Use the returned function and array to determine the moving maximum:

    >>> func(a, window)
    array([ nan,  1.,  2.,  3.])

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
        func = move_max_dict[key]
    except KeyError:
        try:
            func = move_max_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(dtype), str(axis))
            raise TypeError, "Unsupported ndim/dtype/axis (%s/%s/%s)." % tup
    return func, a
'''   
