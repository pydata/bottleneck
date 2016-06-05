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
from numpy cimport npy_intp
from numpy cimport NPY_BOOL

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_DATA
from numpy cimport PyArray_STRIDE
from numpy cimport PyArray_STRIDES
from numpy cimport PyArray_SIZE
from numpy cimport flatiter

from numpy cimport NPY_CONTIGUOUS
from numpy cimport NPY_FORTRAN
from numpy cimport PyArray_CHKFLAGS
from numpy cimport PyArray_Check
from numpy cimport PyArray_TYPE
from numpy cimport PyArray_DIMS
from numpy cimport PyArray_NDIM
from numpy cimport PyArray_SIZE
from numpy cimport PyArray_NBYTES
from numpy cimport PyArray_ITEMSIZE
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
                       nansum_all_float64,
                       nansum_all_float32,
                       nansum_all_int64,
                       nansum_all_int32,
                       nansum_all12_float64,
                       nansum_all12_float32,
                       nansum_all12_int64,
                       nansum_all12_int32,
                       nansum_one_float64,
                       nansum_one_float32,
                       nansum_one_int64,
                       nansum_one_int32,
                       nansum_0d)
    except TypeError:
        return slow.nansum(arr, axis)


cdef object nansum_all12_DTYPE0(char *p,
                                npy_intp stride,
                                npy_intp length,
                                int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0
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


cdef object nansum_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
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
            PyArray_ITER_NEXT(ita)
    return asum


cdef ndarray nansum_one_DTYPE0(flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
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
ctypedef object (*fall12_t)(char *, npy_intp, npy_intp, int)
ctypedef object (*fall_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)

# pointer to functions that reduce along ONE axis
ctypedef ndarray (*fone_t)(flatiter, Py_ssize_t, Py_ssize_t, int,
                           npy_intp*, int)

# pointer to functions that handle 0d arrays
ctypedef object (*f0d_t)(ndarray, int)


cdef reducer(arr, axis,
             fall_t fall_float64,
             fall_t fall_float32,
             fall_t fall_int64,
             fall_t fall_int32,
             fall12_t fall12_float64,
             fall12_t fall12_float32,
             fall12_t fall12_int64,
             fall12_t fall12_int32,
             fone_t fone_float64,
             fone_t fone_float32,
             fone_t fone_int64,
             fone_t fone_int32,
             f0d_t f0d,
             int int_input=0,
             int copy=0):

    # convert to array if necessary
    cdef ndarray a
    if PyArray_Check(arr):
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

    cdef flatiter ita
    cdef Py_ssize_t stride, length, i, j
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

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

    cdef npy_intp *shape = PyArray_DIMS(a)
    cdef npy_intp *strides = PyArray_STRIDES(a)
    cdef npy_intp size
    cdef npy_intp stride_min
    cdef char *p

    if reduce_all == 1:
        # reduce over all axes

        if (a_ndim==1 or PyArray_CHKFLAGS(a, NPY_CONTIGUOUS) or
            PyArray_CHKFLAGS(a, NPY_FORTRAN)):
            stride_min = strides[0]
            for i in range(1, a_ndim):
                if strides[i] < stride_min:
                    stride_min = strides[i]
            size = PyArray_SIZE(a)
            p = <char *>PyArray_DATA(a)
            if dtype == NPY_float64:
                return fall12_float64(p, stride_min, size, int_input)
            elif dtype == NPY_float32:
                return fall12_float32(p, stride_min, size, int_input)
            elif dtype == NPY_int64:
                return fall12_int64(p, stride_min, size, int_input)
            elif dtype == NPY_int32:
                return fall12_int32(p, stride_min, size, int_input)
            else:
                raise TypeError
        else:
            ita = PyArray_IterAllButAxis(a, &axis_reduce)
            stride = strides[axis_reduce]
            length = shape[axis_reduce]
            if dtype == NPY_float64:
                return fall_float64(ita, stride, length, int_input)
            elif dtype == NPY_float32:
                return fall_float32(ita, stride, length, int_input)
            elif dtype == NPY_int64:
                return fall_int64(ita, stride, length, int_input)
            elif dtype == NPY_int32:
                return fall_int32(ita, stride, length, int_input)
            else:
                raise TypeError

    # if we have reached this point then we are reducing an array with
    # ndim > 1 over a single axis

    # output array
    cdef ndarray y
    cdef npy_intp *y_dims = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis_reduce)
    stride = strides[axis_reduce]
    length = shape[axis_reduce]

    # reduce over a single axis; a_ndim > 1
    if a_ndim > 11:
        raise ValueError("arr.ndim must be less than 12")
    j = 0
    for i in range(a_ndim):
        if i != axis_reduce:
            y_dims[j] = shape[i]
            j += 1
    if dtype == NPY_float64:
        y = fone_float64(ita, stride, length, a_ndim, y_dims, int_input)
    elif dtype == NPY_float32:
        y = fone_float32(ita, stride, length, a_ndim, y_dims, int_input)
    elif dtype == NPY_int64:
        y = fone_int64(ita, stride, length, a_ndim, y_dims, int_input)
    elif dtype == NPY_int32:
        y = fone_int32(ita, stride, length, a_ndim, y_dims, int_input)
    else:
        raise TypeError
    return y
