"nn template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["nn"]

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
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=2] a,
                              np.ndarray[np.DTYPE_t, ndim=1] a0):
    "Nearest neighbor of 1d `a0` in 2d `a` with dtype=DTYPE, axis=AXIS."
    cdef:
        np.float64_t xsum = 0, d, xsummin=np.inf, dist
        Py_ssize_t imin = -1, n, a0size
"""

loop = {}
loop[2] = """\
    a0size = PyArray_SIZE(a0)
    if nAXIS != a0size:
        raise ValueError("`a0` must match size of `a` along specified axis")
    for iINDEX0 in range(nINDEX0):
        xsum = 0
        for iINDEX1 in range(nINDEX1):
            d = a[INDEXALL] - a0[iINDEX1]
            xsum += d * d
        if xsum < xsummin:
            xsummin = xsum
            imin = iINDEX0
    if imin == -1:
        dist = NAN
        imin = 0
    else:
        dist = sqrt(xsummin)
    return dist, imin
"""
floats['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "nn"
slow['signature'] = "arr, arr0"
slow['func'] = "bn.slow.nn(arr, arr0, axis=AXIS)"

# Template ------------------------------------------------------------------

nn = {}
nn['name'] = 'nn'
nn['is_reducing_function'] = False
nn['cdef_output'] = False
nn['slow'] = slow
nn['templates'] = {}
nn['templates']['float'] = floats
nn['templates']['int'] = ints
nn['pyx_file'] = 'func/nn.pyx'

nn['main'] = '''"nn auto-generated from template"

def nn(arr, arr0, int axis=1):
    """
    Distance of nearest neighbor (and its index) along specified axis.

    The Euclidian distance between `arr0` and its nearest neighbor in
    `arr` is returned along with the index of the nearest neighbor in
    `arr`.

    The squared distance used to determine the nearest neighbor of `arr0`
    is equivalent to np.sum((arr - arr0) ** 2), axis) where `arr` is 2d
    and `arr0` is 1d and `arr0` must be reshaped if `axis` is 1.

    If all distances are NaN then the distance returned is NaN and the
    index is zero.

    Parameters
    ----------
    arr : array_like
        A 2d array. If `arr` is not an array, a conversion is attempted.
    arr0 : array_like
        A 1d array. If `arr0` is not an array, a conversion is attempted.
    axis : int, optional
        Axis along which the distance is computed. The default (axis=1)
        is to compute the distance along rows.

    Returns
    -------
    dist : np.float64
        The Euclidian distance between `arr0` and the nearest neighbor
        in `arr`. If all distances are NaN then the distance returned
        is NaN.
    idx : int
        Index of nearest neighbor in `arr`. If all distances are NaN
        then the index returned is zero.

    See also
    --------
    bottleneck.ss: Sum of squares along specified axis.

    Notes
    -----
    A brute force algorithm is used to find the nearest neighbor.

    Depending on the shapes of `arr` and `arr0`, SciPy's cKDTree may
    be faster than bn.nn(). So benchmark if speed is important.

    The relative speed also depends on how many times you will use
    the same array `arr` to find nearest neighbors with different
    `arr0`. That is because it takes time to set up SciPy's cKDTree.

    Examples
    --------
    Create the input arrays:

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr0 = np.array([2, 4])

    Find nearest neighbor of `arr0` in `arr` along axis 1:

    >>> dist, idx = bn.nn(arr, arr0, axis=1)
    >>> dist
    1.0
    >>> idx
    1

    Find nearest neighbor of `arr0` in `arr` along axis 0:

    >>> dist, idx = bn.nn(arr, arr0, axis=0)
    >>> dist
    0.0
    >>> idx
    1

    """
    func, a, a0 = nn_selector(arr, arr0, axis)
    return func(a, a0)

def nn_selector(arr, arr0, int axis):
    """
    Return nn function and arrays.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of dtype, and axis. A lot of the overhead in bn.nn() is in
    checking that `axis` is within range, converting `arr` into an array
    (if it is not already an array), and selecting the function to use to
    find the nearest neighbor.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : array_like
        A 2d array. If `arr` is not an array, a conversion is attempted.
    arr0 : array_like
        A 1d array. If `arr0` is not an array, a conversion is attempted.
    axis : int, optional
        Axis along which the distance is computed. The default (axis=1)
        is to compute the distance along rows.

    Returns
    -------
    func : function
        The nn function that is appropriate to use with the given input.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.
    a0 : ndarray
        If the input array `arr0` is not a ndarray, then `a0` will contain the
        result of converting `arr0` into a ndarray.

    Examples
    --------
    Create the input arrays:

    >>> arr = np.array([[1, 2], [3, 4]])
    >>> arr0 = np.array([2, 4])

    Obtain the function needed to find the nearest neighbor of `arr0`
    in `arr0` along axis 0:

    >>> func, a, a0 = bn.func.nn_selector(arr, arr0, axis=0)
    >>> func
    <function nn_2d_int64_axis0>

    Use the returned function and arrays to determine the nearest
    neighbor:

    >>> dist, idx = func(a, a0)
    >>> dist
    0.0
    >>> idx
    1


    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef np.ndarray a0
    if type(arr0) is np.ndarray:
        a0 = arr0
    else:
        a0 = np.array(arr0, copy=False)
    cdef int dtype = PyArray_TYPE(a)
    cdef int dtype0 = PyArray_TYPE(a0)
    if dtype != dtype0:
        raise ValueError("`arr` and `arr0` must be of the same dtype.")
    cdef int ndim = PyArray_NDIM(a)
    if ndim != 2:
        raise ValueError("`arr` must be 2d")
    cdef int ndim0 = PyArray_NDIM(a0)
    if ndim0 != 1:
        raise ValueError("`arr0` must be 1d")
    if axis < 0:
        axis += ndim
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = nn_dict[key]
    except KeyError:
        if axis is not None:
            if (axis < 0) or (axis >= ndim):
                raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = nn_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a, a0
'''
