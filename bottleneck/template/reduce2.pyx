#cython: embedsignature=True

# Select smallest k elements code used for inner loop of median method:
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
# (C) 2010, 2015 Keith Goodman

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t, intp_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INTP as NPY_intp
from numpy cimport NPY_BOOL

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM
from numpy cimport NPY_ANYORDER
from numpy cimport PyArray_ISBYTESWAPPED

from numpy cimport PyArray_FillWithScalar
from numpy cimport PyArray_Copy
from numpy cimport PyArray_Ravel
from numpy cimport PyArray_EMPTY
from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.reduce as slow

cdef double NAN = <double> np.nan
cdef extern from "math.h":
    double sqrt(double x) nogil

cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float32_t MINfloat32 = -np.inf
cdef np.float64_t MINfloat64 = -np.inf


# nansum --------------------------------------------------------------------

def nansum(arr, axis=None):
    """
    Sum of array elements along given axis treating NaNs as zero.

    The data type (dtype) of the output is the same as the input. On 64-bit
    operating systems, 32-bit input is NOT upcast to 64-bit accumulator and
    return values.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose sum is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the sum is computed. The default (axis=None) is to
        compute the sum of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.

    Notes
    -----
    No error is raised on overflow.

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> bn.nansum(1)
    1
    >>> bn.nansum([1])
    1
    >>> bn.nansum([1, np.nan])
    1.0
    >>> a = np.array([[1, 1], [1, np.nan]])
    >>> bn.nansum(a)
    3.0
    >>> bn.nansum(a, axis=0)
    array([ 2.,  1.])

    When positive infinity and negative infinity are present:

    >>> bn.nansum([1, np.nan, np.inf])
    inf
    >>> bn.nansum([1, np.nan, np.NINF])
    -inf
    >>> bn.nansum([1, np.nan, np.inf, np.NINF])
    nan

    """
    try:
        return reducer(arr, axis,
                       nansum_float64,
                       nansum_float32,
                       nansum_int64,
                       nansum_int32,
                       nansum_float64,
                       nansum_float64,
                       nansum_0d)
    except TypeError:
        return slow.nansum(arr, axis)


cdef inline DTYPE0_t nansum_DTYPE0(char *p,
                                   Py_ssize_t stride,
                                   Py_ssize_t length,
                                   int int_input) nogil:
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, asum=0
    for i in range(length):
        ai = (<DTYPE0_t*>(p + i * stride))[0]
        if DTYPE0 == 'float64':
            if ai == ai:
                asum += ai
        if DTYPE0 == 'float32':
            if ai == ai:
                asum += ai
        if DTYPE0 == 'int64':
            asum += ai
        if DTYPE0 == 'int32':
            asum += ai
    return asum


cdef nansum_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return out
    else:
        return 0.0


# nanmean --------------------------------------------------------------------

def nanmean(arr, axis=None):
    """
    Mean of array elements along given axis ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    arr : array_like
        Array containing numbers whose mean is desired. If `arr` is not an
        array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the means are computed. The default (axis=None) is to
        compute the mean of the flattened array.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs.

    See also
    --------
    bottleneck.nanmedian: Median along specified axis, ignoring NaNs.

    Notes
    -----
    No error is raised on overflow. (The sum is computed and then the result
    is divided by the number of non-NaN elements.)

    If positive or negative infinity are present the result is positive or
    negative infinity. But if both positive and negative infinity are present,
    the result is Not A Number (NaN).

    Examples
    --------
    >>> bn.nanmean(1)
    1.0
    >>> bn.nanmean([1])
    1.0
    >>> bn.nanmean([1, np.nan])
    1.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanmean(a)
    2.0
    >>> bn.nanmean(a, axis=0)
    array([ 1.,  4.])

    When positive infinity and negative infinity are present:

    >>> bn.nanmean([1, np.nan, np.inf])
    inf
    >>> bn.nanmean([1, np.nan, np.NINF])
    -inf
    >>> bn.nanmean([1, np.nan, np.inf, np.NINF])
    nan
    """
    cdef int is_int_to_float = 1
    try:
        return reducer(arr, axis,
                       nanmean_float64,
                       nanmean_float32,
                       nansum_int64,
                       nansum_int32,
                       nanmean_int64,
                       nanmean_int32,
                       nanmean_0d,
                       is_int_to_float)
    except TypeError:
        return slow.nanmean(arr, axis)


@cython.cdivision(True)
cdef inline DTYPE0_t nanmean_DTYPE0(char *p,
                                    Py_ssize_t stride,
                                    Py_ssize_t length,
                                    int int_input) nogil:
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t ai, asum = 0
    for i in range(length):
        ai = (<DTYPE0_t*>(p + i * stride))[0]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        return asum / count
    else:
        return NAN


@cython.cdivision(True)
cdef inline DTYPE1_t nanmean_DTYPE0(char *p,
                                    Py_ssize_t stride,
                                    Py_ssize_t length,
                                    int int_input) nogil:
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, count=0
    cdef DTYPE0_t ai
    cdef DTYPE1_t asum=0
    for i in range(length):
        ai = (<DTYPE0_t*>(p + i * stride))[0]
        asum += ai
        count += 1
    if count > 0:
        return asum / count
    else:
        return NAN


cdef nanmean_0d(ndarray a, int int_input):
    return <double>a[()]


# reducer -------------------------------------------------------------------

ctypedef float64_t (*one_float64_t)(char *, Py_ssize_t, Py_ssize_t, int) nogil
ctypedef float32_t (*one_float32_t)(char *, Py_ssize_t, Py_ssize_t, int) nogil
ctypedef int64_t (*one_int64_t)(char *, Py_ssize_t, Py_ssize_t, int) nogil
ctypedef int32_t (*one_int32_t)(char *, Py_ssize_t, Py_ssize_t, int) nogil
ctypedef object (*f0d_t)(ndarray, int)


cdef ndarray one_DTYPE0(np.flatiter ita, Py_ssize_t stride, Py_ssize_t length,
                        int a_ndim, np.npy_intp* y_dims, int int_input,
                        one_DTYPE0_t func):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef DTYPE0_t yi
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    cdef char *p
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                # TODO not every functions uses 0; move this to reducer
                # and each function can pass in a python object like 0,
                # NaN, True etc
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = 0
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                p = <char*>pid(ita)
                yi = func(p, stride, length, int_input)
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = yi
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


cdef reducer(arr, axis,
             one_float64_t func_f64,
             one_float32_t func_f32,
             one_int64_t func_i64,
             one_int32_t func_i32,
             one_float64_t func_i64f64,
             one_float64_t func_i32f64,
             f0d_t f0d,
             int is_int_to_float=0,
             int int_input=0,
             int copy=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # check for byte swapped input array
    cdef bint is_swapped = PyArray_ISBYTESWAPPED(a)
    if is_swapped:
        raise TypeError

    # input array
    if copy == 1:
        a = PyArray_Copy(a)

    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i, j
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # output array, if needed
    cdef ndarray y
    cdef np.npy_intp *adim
    cdef np.npy_intp *y_dims = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # defend against 0d beings
    if a_ndim == 0:
        if axis is None or axis == 0 or axis == -1:
            return f0d(a, int_input)
        else:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # does user want to reduce over all axes?
    cdef int reduce_all = 0
    cdef int axis_int
    cdef int axis_reduce
    if axis is None:
        reduce_all = 1
        axis_reduce = -1
        if a_ndim != 1:
            a = PyArray_Ravel(a, NPY_ANYORDER)
            a_ndim = 1
    else:
        axis_int = <int>axis
        if axis_int < 0:
            axis_int += a_ndim
            if axis_int < 0:
                raise ValueError("axis(=%d) out of bounds" % axis)
        elif axis_int >= a_ndim:
            raise ValueError("axis(=%d) out of bounds" % axis)
        if a_ndim == 1 and axis_int == 0:
            reduce_all = 1
        axis_reduce = axis_int

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis_reduce)
    stride = a.strides[axis_reduce]
    length = a.shape[axis_reduce]

    cdef char *p
    if reduce_all == 1:
        # reduce over all axes
        p = <char*>pid(ita)
        if dtype == NPY_float64:
            return func_f64(p, stride, length, int_input)
        elif dtype == NPY_float32:
            return func_f32(p, stride, length, int_input)
        elif dtype == NPY_int64:
            if is_int_to_float == 1:
                return func_i64f64(p, stride, length, int_input)
            else:
                return func_i64(p, stride, length, int_input)
        elif dtype == NPY_int32:
            if is_int_to_float == 1:
                return func_i32f64(p, stride, length, int_input)
            else:
                return func_i32(p, stride, length, int_input)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
    else:
        # reduce over a single axis; a_ndim > 1
        if a_ndim > 11:
            raise ValueError("arr.ndim must be less than 12")
        adim = np.PyArray_DIMS(a)
        j = 0
        for i in range(a_ndim):
            if i != axis_reduce:
                y_dims[j] = adim[i]
                j += 1
        if dtype == NPY_float64:
            y = one_float64(ita, stride, length, a_ndim, y_dims, int_input,
                            func_f64)
        elif dtype == NPY_float32:
            y = one_float32(ita, stride, length, a_ndim, y_dims, int_input,
                            func_f32)
        elif dtype == NPY_int64:
            y = one_int64(ita, stride, length, a_ndim, y_dims, int_input,
                          func_i64)
        elif dtype == NPY_int32:
            y = one_int32(ita, stride, length, a_ndim, y_dims, int_input,
                          func_i32)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
        return y
