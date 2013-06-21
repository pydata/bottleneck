"argpartsort template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["argpartsort"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]

# loops ---------------------------------------------------------------------

loop = {}
loop[1] = """\
    for i0 in range(n0):
        y[INDEXALL] = i0
    if nAXIS == 0:
        return y
    if (n < 1) or (n > nAXIS):
        raise ValueError(PARTSORT_ERR_MSG % (n, nAXIS))
    l = 0
    r = nAXIS - 1
    with nogil:
        while l < r:
            x = b[k]
            i = l
            j = r
            while 1:
                while b[i] < x: i += 1
                while x < b[j]: j -= 1
                if i <= j:
                    tmp = b[i]
                    b[i] = b[j]
                    b[j] = tmp
                    itmp = y[i]
                    y[i] = y[j]
                    y[j] = itmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    return y
"""
loop[2] = """\
    for i0 in range(n0):
        for i1 in range(n1):
            y[INDEXALL] = iINDEX1
    if nAXIS == 0:
        return y
    if (n < 1) or (n > nAXIS):
        raise ValueError(PARTSORT_ERR_MSG % (n, nAXIS))
    for iINDEX0 in range(nINDEX0):
        l = 0
        r = nAXIS - 1
        while l < r:
            x = b[INDEXREPLACE|k|]
            i = l
            j = r
            while 1:
                while b[INDEXREPLACE|i|] < x: i += 1
                while x < b[INDEXREPLACE|j|]: j -= 1
                if i <= j:
                    tmp = b[INDEXREPLACE|i|]
                    b[INDEXREPLACE|i|] = b[INDEXREPLACE|j|]
                    b[INDEXREPLACE|j|] = tmp
                    itmp = y[INDEXREPLACE|i|]
                    y[INDEXREPLACE|i|] = y[INDEXREPLACE|j|]
                    y[INDEXREPLACE|j|] = itmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
    return y
"""
loop[3] = """\
    for i0 in range(n0):
        for i1 in range(n1):
            for i2 in range(n2):
                y[INDEXALL] = iINDEX2
    if nAXIS == 0:
        return y
    if (n < 1) or (n > nAXIS):
        raise ValueError(PARTSORT_ERR_MSG % (n, nAXIS))
    for iINDEX0 in range(nINDEX0):
        for iINDEX1 in range(nINDEX1):
            l = 0
            r = nAXIS - 1
            while l < r:
                x = b[INDEXREPLACE|k|]
                i = l
                j = r
                while 1:
                    while b[INDEXREPLACE|i|] < x: i += 1
                    while x < b[INDEXREPLACE|j|]: j -= 1
                    if i <= j:
                        tmp = b[INDEXREPLACE|i|]
                        b[INDEXREPLACE|i|] = b[INDEXREPLACE|j|]
                        b[INDEXREPLACE|j|] = tmp
                        itmp = y[INDEXREPLACE|i|]
                        y[INDEXREPLACE|i|] = y[INDEXREPLACE|j|]
                        y[INDEXREPLACE|j|] = itmp
                        i += 1
                        j -= 1
                    if i > j: break
                if j < k: l = i
                if k < i: r = j
    return y
"""

# Float dtypes (not axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = False
floats['force_output_dtype'] = 'intp'
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a, int n):
    "Partial sort of NDIMd array with dtype=DTYPE along axis=AXIS."
    cdef np.npy_intp i, j = 0, l, r, k = n-1, itmp
    cdef np.DTYPE_t x, tmp
    cdef np.ndarray[np.DTYPE_t, ndim=NDIM] b = PyArray_Copy(a)
"""

floats['loop'] = loop

# Int dtypes (not axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "argpartsort"
slow['signature'] = "arr, n"
slow['func'] = "bn.slow.argpartsort(arr, n, axis=AXIS)"

# Template ------------------------------------------------------------------

argpartsort = {}
argpartsort['name'] = 'argpartsort'
argpartsort['is_reducing_function'] = False
argpartsort['cdef_output'] = True
argpartsort['slow'] = slow
argpartsort['templates'] = {}
argpartsort['templates']['float'] = floats
argpartsort['templates']['int'] = ints
argpartsort['pyx_file'] = 'func/argpartsort.pyx'

argpartsort['main'] = '''"argpartsort auto-generated from template"
# Select smallest k elements code used for inner loop of argpartsort method:
# http://projects.scipy.org/numpy/attachment/ticket/1213/quickselect.pyx
# (C) 2009 Sturla Molden
# SciPy license
#
# From the original C function (code in public domain) in:
#   Fast median search: an ANSI C implementation
#   Nicolas Devillard - ndevilla AT free DOT fr
#   July 1998
# which, in turn, took the algorithm from
#   Wirth, Niklaus
#   Algorithms + data structures = programs, p. 366
#   Englewood Cliffs: Prentice-Hall, 1976
#
# Adapted and expanded for Bottleneck:
# (C) 2011 Keith Goodman

def argpartsort(arr, n, axis=-1):
    """
    Return indices that would partially sort an array.

    A partially sorted array is one in which the `n` smallest values appear
    (in any order) in the first `n` elements. The remaining largest elements
    are also unordered. Due to the algorithm used (Wirth's method), the nth
    smallest element is in its sorted position (at index `n-1`).

    Shuffling the input array may change the output. The only guarantee is
    that the first `n` elements will be the `n` smallest and the remaining
    element will appear in the remainder of the output.

    This functions is not protected against NaN. Therefore, you may get
    unexpected results if the input contains NaN.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    n : int
        The indices of the `n` smallest elements will appear in the first `n`
        elements of the output array along the given `axis`.
    axis : {int, None}, optional
        Axis along which the partial sort is performed. The default (axis=-1)
        is to sort along the last axis.

    Returns
    -------
    y : ndarray
        An array the same shape as the input array containing the indices
        that partially sort `arr` such that the `n` smallest elements will
        appear (unordered) in the first `n` elements.

    See Also
    --------
    bottleneck.partsort: Partial sorting of array elements along given axis.

    Notes
    -----
    Unexpected results may occur if the input array contains NaN.

    Examples
    --------
    Create a numpy array:

    >>> a = np.array([1, 0, 3, 4, 2])

    Find the indices that partially sort that array so that the first 3
    elements are the smallest 3 elements:

    >>> index = bn.argpartsort(a, n=3)
    >>> index
    array([0, 1, 4, 3, 2])

    Let's use the indices to partially sort the array (note, as in this
    example, that the smallest 3 elements may not be in order):

    >>> a[index]
    array([1, 0, 2, 4, 3])

    """
    func, arr = argpartsort_selector(arr, axis)
    return func(arr, n)

def argpartsort_selector(arr, axis):
    """
    Return argpartsort function and array that matches `arr` and `axis`.

    Under the hood Bottleneck uses a separate Cython function for each
    combination of ndim, dtype, and axis. A lot of the overhead in
    bn.argpartsort() is in checking that `axis` is within range, converting
    `arr` into an array (if it is not already an array), and selecting the
    function to use to partially sort.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using the this function.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}
        Axis along which to partially sort.

    Returns
    -------
    func : function
        The argpartsort function that matches the number of dimensions and
        dtype of the input array and the axis along which you wish to partially
        sort.
    a : ndarray
        If the input array `arr` is not a ndarray, then `a` will contain the
        result of converting `arr` into a ndarray.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1, 0, 3, 4, 2])

    Obtain the function needed to find the indices of a partial sort of `arr`
    along axis=0:

    >>> func, a = bn.func.argpartsort_selector(arr, axis=0)
    >>> func
    <function argpartsort_1d_int64_axis0>

    Use the returned function and array to find the indices of the partial
    sort:

    >>> func(a, n=3)
    array([0, 1, 4, 3, 2])

    """
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)
    cdef tuple key
    cdef int ndim = PyArray_NDIM(a)
    cdef int dtype = PyArray_TYPE(a)
    if axis is not None:
        if axis < 0:
            axis += ndim
    else:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis = 0
        ndim = 1
    key = (ndim, dtype, axis)
    try:
        func = argpartsort_dict[key]
    except KeyError:
        if (axis < 0) or (axis >= ndim):
            raise ValueError("axis(=%d) out of bounds" % axis)
        try:
            func = argpartsort_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(a.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func, a
'''
