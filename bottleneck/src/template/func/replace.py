"replace template"

from copy import deepcopy
import bottleneck as bn

__all__ = ["replace"]

FLOAT_DTYPES = [x for x in bn.dtypes if 'float' in x]
INT_DTYPES = [x for x in bn.dtypes if 'int' in x]


# Float dtypes (axis=None) ----------------------------------------------

floats = {}
floats['dtypes'] = FLOAT_DTYPES
floats['axisNone'] = True
floats['force_output_dtype'] = False
floats['reuse_non_nan_func'] = False

floats['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
    double old, double new):
    "replace (inplace) specified elements of NDIMd array of dtype=DTYPE."
    cdef np.DTYPE_t ai
"""

loop = {}
loop[1] = """\
    if old == old:
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai == old:
                a[INDEXALL] = new
    else:
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai != ai:
                a[INDEXALL] = new
"""
loop[2] = """\
    if old == old:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai == old:
                    a[INDEXALL] = new
    else:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai != ai:
                    a[INDEXALL] = new
"""
loop[3] = """\
    if old == old:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    if ai == old:
                        a[INDEXALL] = new
    else:
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    if ai != ai:
                        a[INDEXALL] = new
"""
floats['loop'] = loop

# Int dtypes (axis=None) ------------------------------------------------

ints = deepcopy(floats)
ints['dtypes'] = INT_DTYPES
ints['axisNone'] = True

ints['top'] = """
@cython.boundscheck(False)
@cython.wraparound(False)
def NAME_NDIMd_DTYPE_axisAXIS(np.ndarray[np.DTYPE_t, ndim=NDIM] a,
    double old, double new):
    "replace (inplace) specified elements of NDIMd array of dtype=DTYPE."
    cdef np.DTYPE_t asum = 0, ai
    cdef np.DTYPE_t oldint, newint
"""

loop = {}
loop[1] = """\
    if old == old:
        oldint = <np.DTYPE_t>old
        newint = <np.DTYPE_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        for iINDEX0 in range(nINDEX0):
            ai = a[INDEXALL]
            if ai == oldint:
                a[INDEXALL] = newint
"""
loop[2] = """\
    if old == old:
        oldint = <np.DTYPE_t>old
        newint = <np.DTYPE_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                ai = a[INDEXALL]
                if ai == oldint:
                    a[INDEXALL] = newint
"""
loop[3] = """\
    if old == old:
        oldint = <np.DTYPE_t>old
        newint = <np.DTYPE_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        for iINDEX0 in range(nINDEX0):
            for iINDEX1 in range(nINDEX1):
                for iINDEX2 in range(nINDEX2):
                    ai = a[INDEXALL]
                    if ai == oldint:
                        a[INDEXALL] = newint
"""
ints['loop'] = loop

# Slow, unaccelerated ndim/dtype --------------------------------------------

slow = {}
slow['name'] = "replace"
slow['signature'] = "arr, old, new"
slow['func'] = "bn.slow.replace(arr, old, new)"

# Template ------------------------------------------------------------------

replace = {}
replace['name'] = 'replace'
replace['is_reducing_function'] = False
replace['cdef_output'] = False
replace['slow'] = slow
replace['templates'] = {}
replace['templates']['float_None'] = floats
replace['templates']['int_None'] = ints
replace['pyx_file'] = 'func/replace.pyx'

replace['main'] = '''"replace auto-generated from template"

def replace(arr, old, new):
    """
    Replace (inplace) given scalar values of an array with new values.

    The equivalent numpy function:

        arr[arr==old] = new

    Or in the case where old=np.nan:

        arr[np.isnan(old)] = new

    Parameters
    ----------
    arr : numpy.ndarray
        The input array, which is also the output array since this functions
        works inplace.
    old : scalar
        All elements in `arr` with this value will be replaced by `new`.
    new : scalar
        All elements in `arr` with a value of `old` will be replaced by `new`.

    Returns
    -------
    None, the operation is inplace.

    Examples
    --------
    Replace zero with 3 (note that the input array is modified):

    >>> a = np.array([1, 2, 0])
    >>> bn.replace(a, 0, 3)
    >>> a
    array([1, 2, 3])

    Replace np.nan with 0:

    >>> a = np.array([1, 2, np.nan])
    >>> bn.replace(a, np.nan, 0)
    >>> a
    array([ 1.,  2.,  0.])

    """
    func = replace_selector(arr)
    return func(arr, old, new)

def replace_selector(arr):
    """
    Return replace function and array that matches `arr`.

    Under the hood Bottleneck uses a separate replace() Cython function for
    each combination of ndim and dtype. A lot of the overhead in bn.replace()
    is inselecting the low level function to use.

    You can get rid of the overhead by doing all this before you, for example,
    enter an inner loop, by using this function.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array.

    Returns
    -------
    func : function
        The replace() function that matches the number of dimensions and dtype
        of the input array.

    Examples
    --------
    Create a numpy array:

    >>> arr = np.array([1.0, np.nan, 3.0])

    Obtain the function needed to replace values in `arr`:

    >>> func = bn.func.replace_selector(arr)
    >>> func
    <function replace_1d_float64_axisNone>

    Use the returned function to replace NaN with zero:

    >>> func(arr, np.nan, 0)
    >>> arr
    array([ 1.,  0.,  3.])

    """
    axis = None
    if type(arr) is not np.ndarray:
        # replace works in place so input must be an array, not (e.g.) a list
        raise TypeError("`arr` must be a numpy array.")
    cdef int ndim = PyArray_NDIM(arr)
    cdef int dtype = PyArray_TYPE(arr)
    cdef tuple key = (ndim, dtype, axis)
    try:
        func = replace_dict[key]
    except KeyError:
        try:
            func = replace_slow_dict[axis]
        except KeyError:
            tup = (str(ndim), str(arr.dtype), str(axis))
            raise TypeError("Unsupported ndim/dtype/axis (%s/%s/%s)." % tup)
    return func
'''
