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
from numpy cimport flatiter

from numpy cimport PyArray_DATA
from numpy cimport PyArray_SIZE
from numpy cimport PyArray_DIMS
from numpy cimport PyArray_NDIM
from numpy cimport PyArray_TYPE
from numpy cimport PyArray_STRIDE
from numpy cimport PyArray_STRIDES

from numpy cimport NPY_CONTIGUOUS
from numpy cimport NPY_FORTRAN
from numpy cimport NPY_ANYORDER
from numpy cimport PyArray_CHKFLAGS
from numpy cimport PyArray_ISBYTESWAPPED

from numpy cimport PyArray_FROM_O
from numpy cimport PyArray_Copy
from numpy cimport PyArray_EMPTY

from numpy cimport PyArray_Ravel
from numpy cimport PyArray_Check
from numpy cimport PyArray_FillWithScalar

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
                       nansum_all_ss_float64,
                       nansum_all_ss_float32,
                       nansum_all_ss_int64,
                       nansum_all_ss_int32,
                       nansum_one_float64,
                       nansum_one_float32,
                       nansum_one_int64,
                       nansum_one_int32,
                       nansum_0d)
    except TypeError:
        return slow.nansum(arr, axis)


cdef object nansum_all_ss_DTYPE0(char *p,
                                 npy_intp stride,
                                 npy_intp length,
                                 int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0
    with nogil:
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
    try:
        return reducer(arr, axis,
                       nanmean_all_float64,
                       nanmean_all_float32,
                       nanmean_all_int64,
                       nanmean_all_int32,
                       nanmean_all_ss_float64,
                       nanmean_all_ss_float32,
                       nanmean_all_ss_int64,
                       nanmean_all_ss_int32,
                       nanmean_one_float64,
                       nanmean_one_float32,
                       nanmean_one_int64,
                       nanmean_one_int32,
                       nanmean_0d)
    except TypeError:
        return slow.nanmean(arr, axis)


@cython.cdivision(True)
cdef object nanmean_all_ss_DTYPE0(char *p,
                                  npy_intp stride,
                                  npy_intp length,
                                  int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0
    with nogil:
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
cdef object nanmean_all_ss_DTYPE0(char *p,
                                  npy_intp stride,
                                  npy_intp length,
                                  int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0
    with nogil:
        for i in range(length):
            asum += (<DTYPE0_t*>(p + i * stride))[0]
    if length == 0:
        return NAN
    else:
        return asum / length


@cython.cdivision(True)
cdef object nanmean_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                               Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            PyArray_ITER_NEXT(ita)
    if count > 0:
        return asum / count
    else:
        return NAN


@cython.cdivision(True)
cdef object nanmean_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                               Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0
    cdef DTYPE0_t ai
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                asum += ai
            size += length
            PyArray_ITER_NEXT(ita)
    if size == 0:
        return NAN
    else:
        return asum / size


@cython.cdivision(True)
cdef ndarray nanmean_one_DTYPE0(np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum = 0, ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                count = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > 0:
                    asum = asum / count
                else:
                    asum = NAN
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanmean_one_DTYPE0(np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                    asum += ai
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum / length
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


cdef nanmean_0d(ndarray a, int int_input):
    return <double>a[()]


# nanstd --------------------------------------------------------------------

def nanstd(arr, axis=None, int ddof=0):
    """
    Standard deviation along the specified axis, ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Instead of a faster one-pass algorithm, a more stable two-pass algorithm
    is used.

    An example of a one-pass algorithm:

        >>> np.sqrt((arr*arr).mean() - arr.mean()**2)

    An example of a two-pass algorithm:

        >>> np.sqrt(((arr - arr.mean())**2).mean())

    Note in the two-pass algorithm the mean must be found (first pass) before
    the squared deviation (second pass) can be found.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the standard deviation is computed. The default
        (axis=None) is to compute the standard deviation of the flattened
        array.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non-NaN elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis removed.
        If `arr` is a 0-d array, or if axis is None, a scalar is returned.
        `float64` intermediate and return values are used for integer inputs.
        If ddof is >= the number of non-NaN elements in a slice or the slice
        contains only NaNs, then the result for that slice is NaN.

    See also
    --------
    bottleneck.nanvar: Variance along specified axis ignoring NaNs

    Notes
    -----
    If positive or negative infinity are present the result is Not A Number
    (NaN).

    Examples
    --------
    >>> bn.nanstd(1)
    0.0
    >>> bn.nanstd([1])
    0.0
    >>> bn.nanstd([1, np.nan])
    0.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanstd(a)
    1.4142135623730951
    >>> bn.nanstd(a, axis=0)
    array([ 0.,  0.])

    When positive infinity or negative infinity are present NaN is returned:

    >>> bn.nanstd([1, np.nan, np.inf])
    nan

    """
    try:
        return reducer(arr, axis,
                       nanstd_all_float64,
                       nanstd_all_float32,
                       nanstd_all_int64,
                       nanstd_all_int32,
                       nanstd_all_ss_float64,
                       nanstd_all_ss_float32,
                       nanstd_all_ss_int64,
                       nanstd_all_ss_int32,
                       nanstd_one_float64,
                       nanstd_one_float32,
                       nanstd_one_int64,
                       nanstd_one_int32,
                       nanstd_0d,
                       ddof)
    except TypeError:
        return slow.nanstd(arr, axis, ddof=ddof)


@cython.cdivision(True)
cdef object nanstd_all_ss_DTYPE0(char *p,
                                 npy_intp stride,
                                 npy_intp length,
                                 int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0
    cdef DTYPE0_t amean
    cdef DTYPE0_t out
    with nogil:
        for i in range(length):
            ai = (<DTYPE0_t*>(p + i * stride))[0]
            if ai == ai:
                asum += ai
                count += 1
        if count > ddof:
            amean = asum / count
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>(p + i * stride))[0]
                if ai == ai:
                    ai -= amean
                    asum += ai * ai
            out = sqrt(asum / (count - ddof))
        else:
            out = NAN
    return out


@cython.cdivision(True)
cdef object nanstd_all_ss_DTYPE0(char *p,
                                 npy_intp stride,
                                 npy_intp length,
                                 int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE1_t aj
    cdef DTYPE1_t asum = 0
    cdef DTYPE1_t amean
    cdef DTYPE1_t out
    with nogil:
        for i in range(length):
            ai = (<DTYPE0_t*>(p + i * stride))[0]
            if ai == ai:
                asum += ai
        if length > ddof:
            amean = asum / length
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>(p + i * stride))[0]
                if ai == ai:
                    aj = ai - amean
                    asum += aj * aj
            out = sqrt(asum / (length - ddof))
        else:
            out = NAN
    return out


@cython.cdivision(True)
cdef object nanstd_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, amean, ai, out
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            PyArray_ITER_NEXT(ita)
        if count > ddof:
            amean = asum / count
            asum = 0
            PyArray_ITER_RESET(ita)
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        ai -= amean
                        asum += ai * ai
                PyArray_ITER_NEXT(ita)
            out = sqrt(asum / (count - ddof))
        else:
            out = NAN
    return out


@cython.cdivision(True)
cdef object nanstd_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0, amean, aj, out
    cdef DTYPE0_t ai
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                asum += ai
            size += length
            PyArray_ITER_NEXT(ita)
        if size > ddof:
            amean = asum / size
            asum = 0
            PyArray_ITER_RESET(ita)
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    aj = ai - amean
                    asum += aj * aj
                PyArray_ITER_NEXT(ita)
            out =  sqrt(asum / (size - ddof))
        else:
            out =  NAN
    return out


@cython.cdivision(True)
cdef ndarray nanstd_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai, amean
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                count = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > ddof:
                    amean = asum / count
                    asum = 0
                    for i in range(length):
                        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                        if ai == ai:
                            ai -= amean
                            asum += ai * ai
                    asum = sqrt(asum / (count - ddof))
                else:
                    asum = NAN
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanstd_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    asum += ai
                if length > ddof:
                    amean = asum / length
                    asum = 0
                    for i in range(length):
                        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                        aj = ai - amean
                        asum += aj * aj
                    asum = sqrt(asum / (length - ddof))
                else:
                    asum = NAN
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


cdef nanstd_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        if out == np.inf or out == -np.inf:
            return NAN
        else:
            return 0.0
    else:
        return NAN


# nanvar --------------------------------------------------------------------

def nanvar(arr, axis=None, int ddof=0):
    """
    Variance along the specified axis, ignoring NaNs.

    `float64` intermediate and return values are used for integer inputs.

    Instead of a faster one-pass algorithm, a more stable two-pass algorithm
    is used.

    An example of a one-pass algorithm:

        >>> (arr*arr).mean() - arr.mean()**2

    An example of a two-pass algorithm:

        >>> ((arr - arr.mean())**2).mean()

    Note in the two-pass algorithm the mean must be found (first pass) before
    the squared deviation (second pass) can be found.

    Parameters
    ----------
    arr : array_like
        Input array. If `arr` is not an array, a conversion is attempted.
    axis : {int, None}, optional
        Axis along which the variance is computed. The default (axis=None) is
        to compute the variance of the flattened array.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of non_NaN elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        An array with the same shape as `arr`, with the specified axis
        removed. If `arr` is a 0-d array, or if axis is None, a scalar is
        returned. `float64` intermediate and return values are used for
        integer inputs. If ddof is >= the number of non-NaN elements in a
        slice or the slice contains only NaNs, then the result for that slice
        is NaN.

    See also
    --------
    bottleneck.nanstd: Standard deviation along specified axis ignoring NaNs.

    Notes
    -----
    If positive or negative infinity are present the result is Not A Number
    (NaN).

    Examples
    --------
    >>> bn.nanvar(1)
    0.0
    >>> bn.nanvar([1])
    0.0
    >>> bn.nanvar([1, np.nan])
    0.0
    >>> a = np.array([[1, 4], [1, np.nan]])
    >>> bn.nanvar(a)
    2.0
    >>> bn.nanvar(a, axis=0)
    array([ 0.,  0.])

    When positive infinity or negative infinity are present NaN is returned:

    >>> bn.nanvar([1, np.nan, np.inf])
    nan

    """
    try:
        return reducer(arr, axis,
                       nanvar_all_float64,
                       nanvar_all_float32,
                       nanvar_all_int64,
                       nanvar_all_int32,
                       nanvar_all_ss_float64,
                       nanvar_all_ss_float32,
                       nanvar_all_ss_int64,
                       nanvar_all_ss_int32,
                       nanvar_one_float64,
                       nanvar_one_float32,
                       nanvar_one_int64,
                       nanvar_one_int32,
                       nanvar_0d,
                       ddof)
    except TypeError:
        return slow.nanvar(arr, axis, ddof=ddof)


@cython.cdivision(True)
cdef object nanvar_all_ss_DTYPE0(char *p,
                                 npy_intp stride,
                                 npy_intp length,
                                 int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i
    cdef Py_ssize_t count = 0
    cdef DTYPE0_t ai
    cdef DTYPE0_t asum = 0
    cdef DTYPE0_t amean
    cdef DTYPE0_t out
    with nogil:
        for i in range(length):
            ai = (<DTYPE0_t*>(p + i * stride))[0]
            if ai == ai:
                asum += ai
                count += 1
        if count > ddof:
            amean = asum / count
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>(p + i * stride))[0]
                if ai == ai:
                    ai -= amean
                    asum += ai * ai
            out = asum / (count - ddof)
        else:
            out = NAN
    return out


@cython.cdivision(True)
cdef object nanvar_all_ss_DTYPE0(char *p,
                                 npy_intp stride,
                                 npy_intp length,
                                 int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE1_t aj
    cdef DTYPE1_t asum = 0
    cdef DTYPE1_t amean
    cdef DTYPE1_t out
    with nogil:
        for i in range(length):
            ai = (<DTYPE0_t*>(p + i * stride))[0]
            if ai == ai:
                asum += ai
        if length > ddof:
            amean = asum / length
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>(p + i * stride))[0]
                if ai == ai:
                    aj = ai - amean
                    asum += aj * aj
            out = asum / (length - ddof)
        else:
            out = NAN
    return out


@cython.cdivision(True)
cdef object nanvar_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, amean, ai, out
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            PyArray_ITER_NEXT(ita)
        if count > ddof:
            amean = asum / count
            asum = 0
            PyArray_ITER_RESET(ita)
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        ai -= amean
                        asum += ai * ai
                PyArray_ITER_NEXT(ita)
            out =  asum / (count - ddof)
        else:
            out =  NAN
    return out


@cython.cdivision(True)
cdef object nanvar_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0, amean, aj, out
    cdef DTYPE0_t ai
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                asum += ai
            size += length
            PyArray_ITER_NEXT(ita)
        if size > ddof:
            amean = asum / size
            asum = 0
            PyArray_ITER_RESET(ita)
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    aj = ai - amean
                    asum += aj * aj
                PyArray_ITER_NEXT(ita)
            out =  asum / (size - ddof)
        else:
            out =  NAN
    return out


@cython.cdivision(True)
cdef ndarray nanvar_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai, amean
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                count = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        asum += ai
                        count += 1
                if count > ddof:
                    amean = asum / count
                    asum = 0
                    for i in range(length):
                        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                        if ai == ai:
                            ai -= amean
                            asum += ai * ai
                    asum = asum / (count - ddof)
                else:
                    asum = NAN
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanvar_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    with nogil:
        if length == 0:
            while PyArray_ITER_NOTDONE(ity):
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
                PyArray_ITER_NEXT(ity)
        else:
            while PyArray_ITER_NOTDONE(ita):
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    asum += ai
                if length > ddof:
                    amean = asum / length
                    asum = 0
                    for i in range(length):
                        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                        aj = ai - amean
                        asum += aj * aj
                    asum = asum / (length - ddof)
                else:
                    asum = NAN
                (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum
                PyArray_ITER_NEXT(ita)
                PyArray_ITER_NEXT(ity)
    return y


cdef nanvar_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        if out == np.inf or out == -np.inf:
            return NAN
        else:
            return 0.0
    else:
        return NAN


# reducer -------------------------------------------------------------------

# pointer to functions that reduce along ALL axes
ctypedef object (*fall_ss_t)(char *, npy_intp, npy_intp, int)
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
             fall_ss_t fall_ss_float64,
             fall_ss_t fall_ss_float32,
             fall_ss_t fall_ss_int64,
             fall_ss_t fall_ss_int32,
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
        a = PyArray_FROM_O(arr)

    # check for byte swapped input array
    if PyArray_ISBYTESWAPPED(a):
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
                return fall_ss_float64(p, stride_min, size, int_input)
            elif dtype == NPY_float32:
                return fall_ss_float32(p, stride_min, size, int_input)
            elif dtype == NPY_int64:
                return fall_ss_int64(p, stride_min, size, int_input)
            elif dtype == NPY_int32:
                return fall_ss_int32(p, stride_min, size, int_input)
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
