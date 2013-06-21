"anynan template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["anynan"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'bool'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Check for NaNs in NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef int f = 1
    cdef np.DTYPE_t ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        f = 1
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai != ai:
                y[INDEXPOP] = 1
                f = 0
                break
        if f == 1:
            y[INDEXPOP] = 0
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            f = 1
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai != ai:
                    y[INDEXPOP] = 1
                    f = 0
                    break
            if f == 1:
                y[INDEXPOP] = 0
    return y
"""

floats['loop'] = loop

# Float dtypes (axis=None) --------------------------------------------------

floats_None = deepcopy(floats)
floats_None['axisNone'] = True

loop = {}
loop[1] = """\
    for iINDEX0 in range(nINDEX0):
        ai = a[INDEXALL]
        if ai != ai:
            return np.bool_(True)
    return np.bool_(False)
"""
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            if ai != ai:
                return np.bool_(True)
    return np.bool_(False)
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                if ai != ai:
                    return np.bool_(True)
    return np.bool_(False)
"""

floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        y[INDEXPOP] = 0
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            y[INDEXPOP] = 0
    return y
"""

ints['loop'] = loop

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(ints)
ints_None['axisNone'] = True

loop = {}
loop[1] = """\
    return np.bool_(False)
"""
loop[2] = """\
    return np.bool_(False)
"""
loop[3] = """\
    return np.bool_(False)
"""

ints_None['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "anynan"
slow['signature'] = "arr"
slow['func'] = "bn.slow.anynan(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

anynan = {}
anynan['name'] = 'anynan'
anynan['is_reducing_function'] = True
anynan['cdef_output'] = True
anynan['slow'] = slow
anynan['templates'] = {}
anynan['templates']['float'] = floats
anynan['templates']['float_None'] = floats_None
anynan['templates']['int'] = ints
anynan['templates']['int_None'] = ints_None
anynan['pyx_file'] = 'func/anynan.pyx'

anynan['main'] = '''"anynan auto-generated from template"

def anynan(arr, axis=None):
    """
    Test whether any array element along a given axis is NaN.

    Returns single boolean unless `axis` is not ``None``.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which NaNs are searched.  The default (`axis` = ``None``)
        is to search for NaNs over a flattened input array. `axis` may be
        negative, in which case it counts from the last to the first axis.

    Returns
    -------
    y : bool or ndarray
        A new boolean or `ndarray` is returned.

    See also
    --------
    bottleneck.allnan: Test if all array elements along given axis are NaN

    Examples
    --------
    >>> bn.anynan(1)
    False
    >>> bn.anynan(np.nan)
    True
    >>> bn.anynan([1, np.nan])
    True
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.anynan(a)
    True
    >>> bn.anynan(a, axis=0)
    array([False,  True], dtype=bool)

    """
    func, arr = anynan_selector(arr, axis)
    return func(arr)

def anynan_selector(arr, axis):
    """
    Return anynan function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in bn.anynan()
    is in checking that `axis` is within range, converting `arr` into an
    array (if it is not already an array), and selecting the function to use.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which NaNs are searched.

    Returns
    -------
    func : function
        The anynan function that matches the number of dimensions and
        dtype of the input array and the axis.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 3.0])

    Obtain the function needed to determine if there are any NaN in `arr`:

    >>> func, a = bn.func.anynan_selector(arr, axis=0)
    >>> func
    <function anynan_1d_float64_axisNone>

    Use the returned function and array to determine if there are any
    NaNs:

    >>> func(a)
    False

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if (axis is not None) and (axis < 0):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = anynan_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = anynan_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
