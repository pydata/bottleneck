"move_nanmean template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["move_nanmean"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loops ---------------------------------------------------------------------

loop = {}
loop[1] = """\
    if (window < 1) or (window > nINDEX0):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nINDEX0)

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
       y[INDEXALL] = CASTasum / count
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
            y[INDEXALL] = CASTasum / count
        else:
            y[INDEXALL] = NAN

    return y
"""        
loop[2] = """\
    if (window < 1) or (window > nAXIS):
        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)

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
           y[INDEXALL] = CASTasum / count
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
                y[INDEXALL] = CASTasum / count
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
               y[INDEXALL] = CASTasum / count
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
                    y[INDEXALL] = CASTasum / count
                else:
                    y[INDEXALL] = NAN

    return y
"""

# Float dtypes (no axis=None) -----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
                                  int window):
    "Moving mean of NDIMd array of dtype=DTYPE along axis=AXIS ignoring NaNs."
    cdef int count = 0
    cdef double asum = 0, aold, ai
"""

floats['loop'] = {}
floats['loop'][1] = loop[1].replace('CAST', '')
floats['loop'][2] = loop[2].replace('CAST', '')
floats['loop'][3] = loop[3].replace('CAST', '')

# Int dtypes (no axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['force_output_dtype'] = 'float64'
ints['dtypes'] = INT_DTYPES 

ints['loop'] = {}
ints['loop'][1] = loop[1].replace('CAST', '<np.float64_t> ')
ints['loop'][2] = loop[2].replace('CAST', '<np.float64_t> ')
ints['loop'][3] = loop[3].replace('CAST', '<np.float64_t> ')

# The loop code below for integers should be faster than using the
# loop code for floats (which checks for NaNs). But it runs slower.
# Can anyone spot why?

#loop = {}
#loop[1] = """\
#    if (window < 1) or (window > nINDEX0):
#        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nINDEX0)
#
#    for iINDEX0 in range(window - 1):
#        asum += a[INDEXALL]
#        y[INDEXALL] = NAN
#    iINDEX0 = window - 1
#    asum += a[INDEXALL]
#    y[INDEXALL] = <np.float64_t> asum / window
#    for iINDEX0 in range(window, nINDEX0):
#        asum += a[INDEXALL]
#        aold = a[INDEXREPLACE|iAXIS - window|]
#        asum -= aold
#        y[INDEXALL] = <np.float64_t> asum / window 
#
#    return y
#"""        
#loop[2] = """\
#    if (window < 1) or (window > nAXIS):
#        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)
#
#    for iINDEX0 in range(nINDEX0):
#        asum = 0
#        for iINDEX1 in range(window - 1):
#            asum += a[INDEXALL]
#            y[INDEXALL] = NAN
#        iINDEX1 = window - 1
#        asum += a[INDEXALL]
#        y[INDEXALL] = <np.float64_t> asum / window
#        for iINDEX1 in range(window, nINDEX1):
#            asum += a[INDEXALL]
#            aold = a[INDEXREPLACE|iAXIS - window|]
#            asum -= aold
#            y[INDEXALL] = <np.float64_t> asum / window
#
#    return y
#"""
#loop[3] = """\
#    if (window < 1) or (window > nAXIS):
#        raise ValueError, MOVE_WINDOW_ERR_MSG % (window, nAXIS)
#
#    for iINDEX0 in range(nINDEX0):
#        for iINDEX1 in range(nINDEX1):
#            asum = 0
#            for iINDEX2 in range(window - 1):
#                asum += a[INDEXALL]
#                y[INDEXALL] = NAN
#            iINDEX2 = window - 1
#            asum += a[INDEXALL]
#            y[INDEXALL] = <np.float64_t> asum / window
#            for iINDEX2 in range(window, nINDEX2):
#                asum += a[INDEXALL]
#                aold = a[INDEXREPLACE|iAXIS - window|]
#                asum -= aold
#                y[INDEXALL] = <np.float64_t> asum / window 
#
#    return y
#"""

# Template ------------------------------------------------------------------

move_nanmean = {}
move_nanmean['name'] = 'move_nanmean'
move_nanmean['is_reducing_function'] = False
move_nanmean['cdef_output'] = True
move_nanmean['templates'] = {}
move_nanmean['templates']['float'] = floats
move_nanmean['templates']['int'] = ints
move_nanmean['pyx_file'] = '../move/move_nanmean.pyx'

move_nanmean['main'] = '''"move_nanmean auto-generated from template"

def move_nanmean(arr, int window, int axis=0):
    """
    Moving window mean along the specified axis, ignoring NaNs.
    
    Parameters
    ----------
    arr : ndarray
        Input array.
    window : int
        The number of elements in the moving window.
    axis : int, optional
        The axis over which to perform the moving mean. By default the moving
        mean is taken over the first axis (axis=0). An axis of None is not
        allowed.

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis. The output
        has the same shape as the input. 

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_nanmean(arr, window=2)
    array([ nan,  1.5,  2.5,  3.5])

    """
    func, arr = move_nanmean_selector(arr, window, axis)
    return func(arr, window)

def move_nanmean_selector(arr, int window, int axis):
    """
    Return move_nanmean function and array that matches `arr` and `axis`.
    
    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.move_nanmean() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to calculate the moving sum.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the moving mean is to be computed. The default
        (axis=0) is to compute the moving mean along the first axis.
    
    Returns
    -------
    func : function
        The moving nanmean function that matches the number of dimensions,
        dtype, and the axis along which you wish to find the mean.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarra; otherwise a view is
        returned.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    
    Obtain the function needed to determine the sum of `arr` along axis=0:
    
    >>> window, axis = 2, 0
    >>> func, a = bn.move.move_nanmean_selector(arr, window=2, axis=0)
    <built-in function move_nanmean_1d_float64_axis0>    
    
    Use the returned function and array to determine the sum:

    >>> func(a, window)
    array([ nan,  1.5,  2.5,  3.5])

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
        func = move_nanmean_dict[key]
    except KeyError:
        tup = (str(ndim), str(axis))
        raise TypeError, "Unsupported ndim/axis (%s/%s)." % tup
    return func, a
'''   
