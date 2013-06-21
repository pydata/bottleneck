"ss template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["ss"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]


# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = False
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a):
    "Sum of squares of NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef np.DTYPE_t ssum = 0, ai
"""

loop = {}
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        ssum = 0
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            ssum += ai * ai
        y[INDEXPOP] = ssum
    return y
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ssum = 0
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                ssum += ai * ai
            y[INDEXPOP] = ssum
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
        ssum += ai * ai
    return np.DTYPE(ssum)
"""
loop[2] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            ai = a[INDEXALL]
            ssum += ai * ai
    return np.DTYPE(ssum)
"""
loop[3] = """\
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            for iINDEX2 in range(nINDEX2):
                ai = a[INDEXALL]
                ssum += ai * ai
    return np.DTYPE(ssum)
"""
floats_None['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

# Int dtypes (axis=None) ----------------------------------------------------

ints_None = deepcopy(floats_None)
ints_None['dtypes'] = INT_DTYPES
ints_None['axisNone'] = True

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "ss"
slow['signature'] = "arr"
slow['func'] = "bn.slow.ss(arr, axis=AXIS)"

# Template ------------------------------------------------------------------

ss = {}
ss['name'] = 'ss'
ss['is_reducing_function'] = True
ss['cdef_output'] = True
ss['slow'] = slow
ss['templates'] = {}
ss['templates']['float'] = floats
ss['templates']['float_None'] = floats_None
ss['templates']['int'] = ints
ss['templates']['int_None'] = ints_None
ss['pyx_file'] = 'func/ss.pyx'

ss['main'] = '''"ss auto-generated from template"

def ss(arr, axis=0):
    """
    Sum of the square of each element along specified axis.

    Parameters
    ----------
    arr : array_like
        Array whose sum of squares is desired. If `arr` is not an array, a
        conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum if squared is computed. The default (axis=0)
        is to sum the squares along the first dimension.

    Returns
    -------
    y : ndarray
        The sum of a**2 along the given axis.

    See also
    --------
    bottleneck.nn: Nearest neighbor.

    Examples
    --------
    >>> a = np.array([1., 2., 5.])
    >>> bn.ss(a)
    30.0

    And calculating along an axis:

    >>> b = np.array([[1., 2., 5.], [2., 5., 6.]])
    >>> bn.ss(b, axis=1)
    array([ 30., 65.])

    """
    func, arr = ss_selector(arr, axis)
    return func(arr)

def ss_selector(arr, axis):
    """
    Return ss function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.ss() is in checking that `axis` is within range, converting `arr`
    into an array (if it is not already an array), and selecting the function
    to use to calculate the sum of squares.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which the sum of squares is to be computed.

    Returns
    -------
    func : function
        The ss function that matches the number of dimensions and dtype
        of the input array and the axis along which you wish to find the sum
        if squares.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, 2.0, 5.0])

    Obtain the function needed to determine the sum of squares of `arr` along
    axis=0:

    >>> func, a = bn.func.ss_selector(arr, axis=0)
    >>> func
    <function ss_1d_float64_axisNone>

    Use the returned function and array to determine the sum of squares:

    >>> func(a)
    30.0

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
    if (axis is not None) and (axis < 0):
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = ss_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = ss_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
