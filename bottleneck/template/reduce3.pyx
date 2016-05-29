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
from numpy cimport NPY_CORDER
from numpy cimport PyArray_ISBYTESWAPPED
from numpy cimport PyArray_STRIDES

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
                       nansum_all_float64,
                       nansum_all_float32,
                       nansum_all_int64,
                       nansum_all_int32,
                       nansum_one_float64,
                       nansum_one_float32,
                       nansum_one_int64,
                       nansum_one_int32,
                       nansum_0d)
    except TypeError:
        return slow.nansum(arr, axis)


cdef object nansum_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]

    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0

    cdef Py_ssize_t remainder = length % 4
    cdef Py_ssize_t stride2 = 2 * stride
    cdef Py_ssize_t stride3 = 3 * stride
    cdef Py_ssize_t stride4 = 4 * stride
    cdef Py_ssize_t width = (length - (length % 4)) * stride
    cdef DTYPE0_t x[4]
    cdef char *p
    cdef char *pstop

    with nogil:
        if length >= 8:
            while PyArray_ITER_NOTDONE(ita):
                p = <char*>pid(ita)
                pstop = p + width
                x[0] = x[1] = x[2] = x[3] = 0
                while 1:
                    if p >= pstop:
                        break
                    ai = (<DTYPE0_t*>p)[0]
                    if ai == ai: x[0] += ai
                    ai = (<DTYPE0_t*>(p + stride))[0]
                    if ai == ai: x[1] += ai
                    ai = (<DTYPE0_t*>(p + stride2))[0]
                    if ai == ai: x[2] += ai
                    ai = (<DTYPE0_t*>(p + stride3))[0]
                    if ai == ai: x[3] += ai
                    p += stride4
                for i in range(remainder):
                    ai = (<DTYPE0_t*>p)[0]
                    if ai == ai: x[0] += ai
                    p += stride
                asum += (x[0] + x[1]) + (x[2] + x[3])
                PyArray_ITER_NEXT(ita)
        else:
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>(<char*>pid(ita) + i * stride))[0]
                    if ai == ai: asum += ai
                PyArray_ITER_NEXT(ita)

    return asum


cdef object nansum_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]

    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0

    cdef Py_ssize_t remainder = length % 4
    cdef Py_ssize_t stride2 = 2 * stride
    cdef Py_ssize_t stride3 = 3 * stride
    cdef Py_ssize_t stride4 = 4 * stride
    cdef Py_ssize_t width = (length - (length % 4)) * stride
    cdef DTYPE0_t x[4]
    cdef char *p
    cdef char *pstop

    with nogil:
        if length >= 8:
            while PyArray_ITER_NOTDONE(ita):
                p = <char*>pid(ita)
                pstop = p + width
                x[0] = x[1] = x[2] = x[3] = 0
                while 1:
                    if p >= pstop:
                        break
                    x[0] += (<DTYPE0_t*>p)[0]
                    x[1] += (<DTYPE0_t*>(p + stride))[0]
                    x[2] += (<DTYPE0_t*>(p + stride2))[0]
                    x[3] += (<DTYPE0_t*>(p + stride3))[0]
                    p += stride4
                for i in range(remainder):
                    x[0] += (<DTYPE0_t*>p)[0]
                    p += stride
                asum += (x[0] + x[1]) + (x[2] + x[3])
                PyArray_ITER_NEXT(ita)
        else:
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    asum += (<DTYPE0_t*>(<char*>pid(ita) + i * stride))[0]
                PyArray_ITER_NEXT(ita)

    return asum


cdef ndarray nansum_one_DTYPE0(np.flatiter ita, Py_ssize_t stride, Py_ssize_t fstride,
                               Py_ssize_t length, int a_ndim,
                               np.npy_intp* y_dims, int int_input, int fast, Py_ssize_t length_fast):
    # bn.dtypes = [['float64'], ['float32']]

    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)

    cdef Py_ssize_t remainder
    cdef Py_ssize_t stride1
    cdef Py_ssize_t stride2
    cdef Py_ssize_t stride3
    cdef Py_ssize_t stride4
    cdef Py_ssize_t width
    cdef Py_ssize_t stride_step
    cdef DTYPE0_t x[4]
    cdef DTYPE0_t *p
    cdef DTYPE0_t *pstop

    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = 0
                PyArray_ITER_NEXT(ity)
        elif length >= 8:
            if fast == 1:
                remainder = length % 4
                stride1 = stride / sizeof(DTYPE0_t)
                stride2 = 2 * stride1
                stride3 = 3 * stride1
                stride4 = 4 * stride1
                width = (length - (length % 4)) * stride1
                while PyArray_ITER_NOTDONE(ita):
                    p = <DTYPE0_t*>pid(ita)
                    pstop = p + width
                    x[0] = x[1] = x[2] = x[3] = 0
                    while 1:
                        if p >= pstop:
                            break
                        ai = p[0]
                        if ai == ai: x[0] += ai
                        ai = (p + stride1)[0]
                        if ai == ai: x[1] += ai
                        ai = (p + stride2)[0]
                        if ai == ai: x[2] += ai
                        ai = (p + stride3)[0]
                        if ai == ai: x[3] += ai
                        p += stride4
                    for i in range(remainder):
                        ai = p[0]
                        if ai == ai: x[0] += ai
                        p += stride1
                    asum = (x[0] + x[1]) + (x[2] + x[3])
                    (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                    PyArray_ITER_NEXT(ita)
                    PyArray_ITER_NEXT(ity)
            else:
                remainder = length_fast % 4
                stride_step = stride / sizeof(DTYPE0_t)
                stride1 = fstride / sizeof(DTYPE0_t)
                stride2 = 2 * stride1
                stride3 = 3 * stride1
                stride4 = 4 * stride1
                width = length * stride_step
                while PyArray_ITER_NOTDONE(ita):
                    p = <DTYPE0_t*>pid(ita)
                    pstop = p + width
                    x[0] = x[1] = x[2] = x[3] = 0
                    while 1:
                        if p >= pstop:
                            break
                        ai = p[0]
                        if ai == ai: x[0] += ai
                        ai = (p + stride1)[0]
                        if ai == ai: x[1] += ai
                        ai = (p + stride2)[0]
                        if ai == ai: x[2] += ai
                        ai = (p + stride3)[0]
                        if ai == ai: x[3] += ai
                        p += stride_step
                    p = <DTYPE0_t*>pid(ita)
                    while 1:
                        if p >= pstop:
                            break
                        for i in range(remainder):
                            ai = p[0]
                            if ai == ai: x[i] += ai
                        p += stride_step
                    (<DTYPE0_t*>((<char*>pid(ity))))[0] = x[0]
                    (<DTYPE0_t*>((<char*>pid(ity) + stride1)))[0] = x[1]
                    (<DTYPE0_t*>((<char*>pid(ity) + stride2)))[0] = x[2]
                    (<DTYPE0_t*>((<char*>pid(ity) + stride3)))[0] = x[3]
                    PyArray_ITER_NEXT(ita)
                    PyArray_ITER_NEXT(ity)

        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>(<char*>pid(ita) + i * stride))[0]
                    if ai == ai: asum += ai
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)

    return y


cdef ndarray nansum_one_DTYPE0(np.flatiter ita, Py_ssize_t stride, Py_ssize_t fstride,
                               Py_ssize_t length, int a_ndim,
                               np.npy_intp* y_dims, int int_input, int fast, Py_ssize_t length_fast):
    # bn.dtypes = [['int64'], ['int32']]

    cdef Py_ssize_t i
    cdef DTYPE0_t asum
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)

    cdef Py_ssize_t remainder = length % 4
    cdef Py_ssize_t stride2 = 2 * stride
    cdef Py_ssize_t stride3 = 3 * stride
    cdef Py_ssize_t stride4 = 4 * stride
    cdef Py_ssize_t width = (length - (length % 4)) * stride
    cdef DTYPE0_t x[4]
    cdef char *p
    cdef char *pstop

    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = 0
                PyArray_ITER_NEXT(ity)
        elif length >= 8:
            while PyArray_ITER_NOTDONE(ita):
                p = <char*>pid(ita)
                pstop = p + width
                x[0] = x[1] = x[2] = x[3] = 0
                while 1:
                    if p >= pstop:
                        break
                    x[0] += (<DTYPE0_t*>p)[0]
                    x[1] += (<DTYPE0_t*>(p + stride))[0]
                    x[2] += (<DTYPE0_t*>(p + stride2))[0]
                    x[3] += (<DTYPE0_t*>(p + stride3))[0]
                    p += stride4
                for i in range(remainder):
                    x[0] += (<DTYPE0_t*>p)[0]
                    p += stride
                asum = (x[0] + x[1]) + (x[2] + x[3])
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    asum += (<DTYPE0_t*>(<char*>pid(ita) + i * stride))[0]
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)

    return y


cdef nansum_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return out
    else:
        return 0.0


# reducer -------------------------------------------------------------------

# pointer to functions that reduce along ALL axes
ctypedef object (*fall_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)

# pointer to functions that reduce along ONE axis
ctypedef ndarray (*fone_t)(np.flatiter, Py_ssize_t, Py_ssize_t, Py_ssize_t, int,
                           np.npy_intp*, int, int, Py_ssize_t)

# pointer to functions that handle 0d arrays
ctypedef object (*f0d_t)(ndarray, int)


cdef reducer(arr, axis,
             fall_t fall_float64,
             fall_t fall_float32,
             fall_t fall_int64,
             fall_t fall_int32,
             fone_t fone_float64,
             fone_t fone_float32,
             fone_t fone_int64,
             fone_t fone_int32,
             f0d_t f0d,
             int int_input=0,
             int ravel=0,
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
    if ravel == 1:
        a = PyArray_Ravel(a, NPY_CORDER)
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
    cdef np.npy_intp *strides = PyArray_STRIDES(a)
    cdef np.npy_intp fstride=0
    cdef fast = 0
    cdef int axis_fast = axis_reduce
    for i in range(a_ndim):
        if strides[i] > fstride:
            fstride = strides[i]
            axis_fast = i
            length_fast = a.shape[i]

    if reduce_all == 1:
        # reduce over all axes
        if dtype == NPY_float64:
            return fall_float64(ita, stride, length, int_input)
        elif dtype == NPY_float32:
            return fall_float32(ita, stride, length, int_input)
        elif dtype == NPY_int64:
            return fall_int64(ita, stride, length, int_input)
        elif dtype == NPY_int32:
            return fall_int32(ita, stride, length, int_input)
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
        if axis_fast == axis_reduce:
            fast = 1
        if dtype == NPY_float64:
            y = fone_float64(ita, stride, fstride, length, a_ndim, y_dims, int_input, fast, length_fast)
        elif dtype == NPY_float32:
            y = fone_float32(ita, stride, fstride, length, a_ndim, y_dims, int_input, fast, length_fast)
        elif dtype == NPY_int64:
            y = fone_int64(ita, stride, fstride, length, a_ndim, y_dims, int_input, fast, length_fast)
        elif dtype == NPY_int32:
            y = fone_int32(ita, stride, fstride, length, a_ndim, y_dims, int_input, fast, length_fast)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
        return y
