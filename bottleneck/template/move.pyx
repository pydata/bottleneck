#cython: embedsignature=True

# The minimum on a sliding window algorithm by Richard Harter
# http://www.richardhartersworld.com/cri/2001/slidingmin.html
# Original C code:
# Copyright Richard Harter 2009
# Released under a Simplified BSD license
#
# Adapted and expanded for Bottleneck:
# Copyright 2010, 2014 Keith Goodman
# Released under the Bottleneck license

from libc cimport stdlib

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport npy_intp

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_IterAllButAxis
from numpy cimport flatiter

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM
from numpy cimport PyArray_DIMS
from numpy cimport PyArray_STRIDE

from numpy cimport PyArray_ISBYTESWAPPED
from numpy cimport PyArray_Check

from numpy cimport PyArray_Copy
from numpy cimport PyArray_EMPTY
from numpy cimport PyArray_FROM_O

from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.move as slow

cdef double NAN = <double> np.nan
cdef extern from "math.h":
    double sqrt(double x) nogil

# Used by move_min and move_max
cdef struct pairs:
    double value
    int death

cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

cdef np.float32_t MINfloat32 = -np.inf
cdef np.float64_t MINfloat64 = -np.inf


# move_sum -----------------------------------------------------------------

def move_sum(arr, int window, min_count=None, int axis=-1):
    """
    Moving window sum along the specified axis, optionally ignoring NaNs.

    This function cannot handle input arrays that contain Inf. When the
    window contains Inf, the output will correctly be Inf. However, when Inf
    moves out of the window, the remaining output values in the slice will
    incorrectly be NaN.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving sum of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_sum(arr, window=2)
    array([ nan,   3.,   5.,  nan,  nan])
    >>> bn.move_sum(arr, window=2, min_count=1)
    array([ 1.,  3.,  5.,  3.,  5.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_sum_float64,
                     move_sum_float32,
                     move_sum_int64,
                     move_sum_int32)
    except TypeError:
        return slow.move_sum(arr, window, min_count, axis)


cdef ndarray move_sum_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            count = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                if count >= min_count:
                    yi = asum
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count >= min_count:
                    yi = asum
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


cdef ndarray move_sum_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum, aold, yi
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                yi = <DTYPE1_t>asum
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                asum -= aold
                yi = <DTYPE1_t>asum
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


# move_mean -----------------------------------------------------------------

def move_mean(arr, int window, min_count=None, int axis=-1):
    """
    Moving window mean along the specified axis, optionally ignoring NaNs.

    This function cannot handle input arrays that contain Inf. When the
    window contains Inf, the output will correctly be Inf. However, when Inf
    moves out of the window, the remaining output values in the slice will
    incorrectly be NaN.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving mean of the input array along the specified axis. The output
        has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_mean(arr, window=2)
    array([ nan,  1.5,  2.5,  nan,  nan])
    >>> bn.move_mean(arr, window=2, min_count=1)
    array([ 1. ,  1.5,  2.5,  3. ,  5. ])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_mean_float64,
                     move_mean_float32,
                     move_mean_int64,
                     move_mean_int32)
    except TypeError:
        return slow.move_mean(arr, window, min_count, axis)


@cython.cdivision(True)
cdef ndarray move_mean_DTYPE0(ndarray a, int window, int min_count, int axis,
                              flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int a_ndim,
                              npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            count = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                if count >= min_count:
                    yi = asum / count
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                if aold == aold:
                    asum -= aold
                    count -= 1
                if count >= min_count:
                    yi = asum / count
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_mean_DTYPE0(ndarray a, int window, int min_count, int axis,
                              flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int a_ndim,
                              npy_intp* y_dims, int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum, aold, yi
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                yi = asum / (i + 1)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                asum -= aold
                yi = asum / window
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


# move_std -----------------------------------------------------------------

def move_std(arr, int window, min_count=None, int axis=-1, int ddof=0):
    """
    Moving window standard deviation along the specified axis, optionally
    ignoring NaNs.

    This function cannot handle input arrays that contain Inf. When Inf
    enters the moving window, the outout becomes NaN and will continue to
    be NaN for the remainer of the slice.

    Unlike bn.nanstd, which uses a two-pass algorithm, move_nanstd uses a
    one-pass algorithm called Welford's method. The algorithm is slow but
    numerically stable for cases where the mean is large compared to the
    standard deviation.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        The moving standard deviation of the input array along the specified
        axis. The output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_std(arr, window=2)
    array([ nan,  0.5,  0.5,  nan,  nan])
    >>> bn.move_std(arr, window=2, min_count=1)
    array([ 0. ,  0.5,  0.5,  0. ,  0. ])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_std_float64,
                     move_std_float32,
                     move_std_int64,
                     move_std_int32,
                     ddof)
    except TypeError:
        return slow.move_std(arr, window, min_count, axis, ddof)


@cython.cdivision(True)
cdef ndarray move_std_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t delta, amean, assqdm, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            amean = 0
            assqdm = 0
            count = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                    delta = ai - amean
                    amean += delta / count
                    assqdm += delta * (ai - amean)
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                    delta = ai - amean
                    amean += delta / count
                    assqdm += delta * (ai - amean)
                if count >= min_count:
                    if assqdm < 0:
                        assqdm = 0
                    yi = sqrt(assqdm / (count - ddof))
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                if ai == ai:
                    if aold == aold:
                        delta = ai - aold
                        aold -= amean
                        amean += delta / count
                        ai -= amean
                        assqdm += (ai + aold) * delta
                    else:
                        count += 1
                        delta = ai - amean
                        amean += delta / count
                        assqdm += delta * (ai - amean)
                else:
                    if aold == aold:
                        count -= 1
                        if count > 0:
                            delta = aold - amean
                            amean -= delta / count
                            assqdm -= delta * (aold - amean)
                        else:
                            amean = 0
                            assqdm = 0
                if count >= min_count:
                    if assqdm < 0:
                        assqdm = 0
                    yi = sqrt(assqdm / (count - ddof))
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_std_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef int winddof
    cdef DTYPE1_t delta, amean, assqdm, yi, ai, aold
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        winddof = window - ddof
        while PyArray_ITER_NOTDONE(ita):
            amean = 0
            assqdm = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                delta = ai - amean
                amean += delta / (i + 1)
                assqdm += delta * (ai - amean)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                delta = ai - amean
                amean += delta / (i + 1)
                assqdm += delta * (ai - amean)
                yi = sqrt(assqdm / (i + 1 - ddof))
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                delta = ai - aold
                aold -= amean
                amean += delta / window
                ai -= amean
                assqdm += (ai + aold) * delta
                if assqdm < 0:
                    assqdm = 0
                yi = sqrt(assqdm / winddof)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


# move_var-----------------------------------------------------------------

def move_var(arr, int window, min_count=None, int axis=-1, int ddof=0):
    """
    Moving window variance along the specified axis, optionally ignoring NaNs.

    This function cannot handle input arrays that contain Inf. When Inf
    enters the moving window, the outout becomes NaN and will continue to
    be NaN for the remainer of the slice.

    Unlike bn.nanvar, which uses a two-pass algorithm, move_nanvar uses a
    one-pass algorithm called Welford's method. The algorithm is slow but
    numerically stable for cases where the mean is large compared to the
    standard deviation.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero.

    Returns
    -------
    y : ndarray
        The moving variance of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_var(arr, window=2)
    array([ nan,  0.25,  0.25,  nan,  nan])
    >>> bn.move_var(arr, window=2, min_count=1)
    array([ 0. ,  0.25,  0.25,  0. ,  0. ])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_var_float64,
                     move_var_float32,
                     move_var_int64,
                     move_var_int32,
                     ddof)
    except TypeError:
        return slow.move_var(arr, window, min_count, axis, ddof)


@cython.cdivision(True)
cdef ndarray move_var_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t delta, amean, assqdm, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            amean = 0
            assqdm = 0
            count = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                    delta = ai - amean
                    amean += delta / count
                    assqdm += delta * (ai - amean)
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                    delta = ai - amean
                    amean += delta / count
                    assqdm += delta * (ai - amean)
                if count >= min_count:
                    if assqdm < 0:
                        assqdm = 0
                    yi = assqdm / (count - ddof)
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                if ai == ai:
                    if aold == aold:
                        delta = ai - aold
                        aold -= amean
                        amean += delta / count
                        ai -= amean
                        assqdm += (ai + aold) * delta
                    else:
                        count += 1
                        delta = ai - amean
                        amean += delta / count
                        assqdm += delta * (ai - amean)
                else:
                    if aold == aold:
                        count -= 1
                        if count > 0:
                            delta = aold - amean
                            amean -= delta / count
                            assqdm -= delta * (aold - amean)
                        else:
                            amean = 0
                            assqdm = 0
                if count >= min_count:
                    if assqdm < 0:
                        assqdm = 0
                    yi = assqdm / (count - ddof)
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_var_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef int winddof
    cdef DTYPE1_t delta, amean, assqdm, yi, ai, aold
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    with nogil:
        winddof = window - ddof
        while PyArray_ITER_NOTDONE(ita):
            amean = 0
            assqdm = 0
            for i in range(min_count - 1):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                delta = ai - amean
                amean += delta / (i + 1)
                assqdm += delta * (ai - amean)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
            for i in range(min_count - 1, window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                delta = ai - amean
                amean += delta / (i + 1)
                assqdm += delta * (ai - amean)
                yi = assqdm / (i + 1 - ddof)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                delta = ai - aold
                aold -= amean
                amean += delta / window
                ai -= amean
                assqdm += (ai + aold) * delta
                if assqdm < 0:
                    assqdm = 0
                yi = assqdm / winddof
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


# move_min ---------------------------------------------------------------

def move_min(arr, int window, min_count=None, int axis=-1):
    """
    Moving window minimum along the specified axis, optionally ignoring NaNs.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving minimum of the input array along the specified axis. The
        output has the same shape as the input. The dtype of the output is
        always float64.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_min(arr, window=2)
    array([ nan,   1.,   2.,  nan,  nan])
    >>> bn.move_min(arr, window=2, min_count=1)
    array([ 1.,  1.,  2.,  3.,  5.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_min_float64,
                     move_min_float32,
                     move_min_int64,
                     move_min_int32)
    except TypeError:
        return slow.move_min(arr, window, min_count, axis)


cdef ndarray move_min_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64'], ['float32']]
    cdef DTYPE0_t ai, aold, yi
    cdef Py_ssize_t i, count
    cdef pairs* ring
    cdef pairs* minpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            minpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            if ai == ai:
                minpair.value = ai
            else:
                minpair.value = MAXDTYPE0
            minpair.death = window

            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                else:
                    ai = MAXDTYPE0
                if i >= window:
                    aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                    if aold == aold:
                        count -= 1
                if minpair.death == i:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                if ai <= minpair.value:
                    minpair.value = ai
                    minpair.death = i + window
                    last = minpair
                else:
                    while last.value >= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if count >= min_count:
                    yi = minpair.value
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


cdef ndarray move_min_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    cdef Py_ssize_t i
    cdef pairs* ring
    cdef pairs* minpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            minpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            minpair.value = ai
            minpair.death = window

            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if minpair.death == i:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                if ai <= minpair.value:
                    minpair.value = ai
                    minpair.death = i + window
                    last = minpair
                else:
                    while last.value >= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if i + 1 >= min_count:
                    yi = minpair.value
                else:
                    yi = NAN
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


# move_max ---------------------------------------------------------------

def move_max(arr, int window, min_count=None, int axis=-1):
    """
    Moving window maximum along the specified axis, optionally ignoring NaNs.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving maximum of the input array along the specified axis. The
        output has the same shape as the input. The dtype of the output is
        always float64.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
    >>> bn.move_max(arr, window=2)
    array([ nan,   2.,   3.,  nan,  nan])
    >>> bn.move_max(arr, window=2, min_count=1)
    array([ 1.,  2.,  3.,  3.,  5.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_max_float64,
                     move_max_float32,
                     move_max_int64,
                     move_max_int32)
    except TypeError:
        return slow.move_max(arr, window, min_count, axis)


cdef ndarray move_max_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64'], ['float32']]
    cdef DTYPE0_t ai, aold, yi
    cdef Py_ssize_t i, count
    cdef pairs* ring
    cdef pairs* maxpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            maxpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            if ai == ai:
                maxpair.value = ai
            else:
                maxpair.value = MINDTYPE0
            maxpair.death = window

            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                else:
                    ai = MINDTYPE0
                if i >= window:
                    aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                    if aold == aold:
                        count -= 1
                if maxpair.death == i:
                    maxpair += 1
                    if maxpair >= end:
                        maxpair = ring
                if ai >= maxpair.value:
                    maxpair.value = ai
                    maxpair.death = i + window
                    last = maxpair
                else:
                    while last.value <= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if count >= min_count:
                    yi = maxpair.value
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


cdef ndarray move_max_DTYPE0(ndarray a, int window, int min_count, int axis,
                             flatiter ita, Py_ssize_t stride,
                             Py_ssize_t length, int a_ndim,
                             npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    cdef Py_ssize_t i
    cdef pairs* ring
    cdef pairs* maxpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            maxpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            maxpair.value = ai
            maxpair.death = window

            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if maxpair.death == i:
                    maxpair += 1
                    if maxpair >= end:
                        maxpair = ring
                if ai >= maxpair.value:
                    maxpair.value = ai
                    maxpair.death = i + window
                    last = maxpair
                else:
                    while last.value <= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if i + 1 >= min_count:
                    yi = maxpair.value
                else:
                    yi = NAN
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


# move_argmin ---------------------------------------------------------------

def move_argmin(arr, int window, min_count=None, int axis=-1):
    """
    Moving window index of minimum along the specified axis, optionally
    ignoring NaNs.

    Index 0 is at the rightmost edge of the window. For example, if the array
    is monotonically decreasing (increasing) along the specified axis then
    the output array will contain zeros (window-1).

    If there is a tie in input values within a window, then the rightmost
    index is returned.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving index of minimum values of the input array along the
        specified axis. The output has the same shape as the input. The dtype
        of the output is always float64.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> bn.move_argmin(arr, window=2)
    array([ nan,   1.,   1.,   1.,   1.])

    >>> arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    >>> bn.move_argmin(arr, window=2)
    array([ nan,   0.,   0.,   0.,   0.])

    >>> arr = np.array([2.0, 3.0, 4.0, 1.0, 7.0, 5.0, 6.0])
    >>> bn.move_argmin(arr, window=3)
    array([ nan,  nan,   2.,   0.,   1.,   2.,   1.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_argmin_float64,
                     move_argmin_float32,
                     move_argmin_int64,
                     move_argmin_int32)
    except TypeError:
        return slow.move_argmin(arr, window, min_count, axis)


cdef ndarray move_argmin_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64'], ['float32']]
    cdef DTYPE0_t ai, aold, yi
    cdef Py_ssize_t i, count
    cdef pairs* ring
    cdef pairs* minpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            minpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            if ai == ai:
                minpair.value = ai
            else:
                minpair.value = MAXDTYPE0
            minpair.death = window

            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                else:
                    ai = MAXDTYPE0
                if i >= window:
                    aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                    if aold == aold:
                        count -= 1
                if minpair.death == i:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                if ai <= minpair.value:
                    minpair.value = ai
                    minpair.death = i + window
                    last = minpair
                else:
                    while last.value >= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if count >= min_count:
                    yi = i - minpair.death + window
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


cdef ndarray move_argmin_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    cdef Py_ssize_t i
    cdef pairs* ring
    cdef pairs* minpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            minpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            minpair.value = ai
            minpair.death = window

            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if minpair.death == i:
                    minpair += 1
                    if minpair >= end:
                        minpair = ring
                if ai <= minpair.value:
                    minpair.value = ai
                    minpair.death = i + window
                    last = minpair
                else:
                    while last.value >= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if i + 1 >= min_count:
                    yi = i - minpair.death + window
                else:
                    yi = NAN
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


# move_argmax ---------------------------------------------------------------

def move_argmax(arr, int window, min_count=None, int axis=-1):
    """
    Moving window index of maximum along the specified axis, optionally
    ignoring NaNs.

    Index 0 is at the rightmost edge of the window. For example, if the array
    is monotonically increasing (decreasing) along the specified axis then
    the output array will contain zeros (window-1).

    If there is a tie in input values within a window, then the rightmost
    index is returned.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving index of maximum values of the input array along the
        specified axis. The output has the same shape as the input. The dtype
        of the output is always float64.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> bn.move_argmax(arr, window=2)
    array([ nan,   0.,   0.,   0.,   0.])

    >>> arr = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    >>> bn.move_argmax(arr, window=2)
    array([ nan,   1.,   1.,   1.,   1.])

    >>> arr = np.array([2.0, 3.0, 4.0, 1.0, 7.0, 5.0, 6.0])
    >>> bn.move_argmax(arr, window=3)
    array([ nan,  nan,   0.,   1.,   0.,   1.,   2.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_argmax_float64,
                     move_argmax_float32,
                     move_argmax_int64,
                     move_argmax_int32)
    except TypeError:
        return slow.move_argmax(arr, window, min_count, axis)


cdef ndarray move_argmax_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64'], ['float32']]
    cdef DTYPE0_t ai, aold, yi
    cdef Py_ssize_t i, count
    cdef pairs* ring
    cdef pairs* maxpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            maxpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            if ai == ai:
                maxpair.value = ai
            else:
                maxpair.value = MINDTYPE0
            maxpair.death = window

            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    count += 1
                else:
                    ai = MINDTYPE0
                if i >= window:
                    aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
                    if aold == aold:
                        count -= 1
                if maxpair.death == i:
                    maxpair += 1
                    if maxpair >= end:
                        maxpair = ring
                if ai >= maxpair.value:
                    maxpair.value = ai
                    maxpair.death = i + window
                    last = maxpair
                else:
                    while last.value <= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if count >= min_count:
                    yi = i - maxpair.death + window
                else:
                    yi = NAN
                (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


cdef ndarray move_argmax_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    cdef Py_ssize_t i
    cdef pairs* ring
    cdef pairs* maxpair
    cdef pairs* end
    cdef pairs* last
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]

    with nogil:
        ring = <pairs*>stdlib.malloc(window * sizeof(pairs))
        while PyArray_ITER_NOTDONE(ita):

            end = ring + window
            last = ring

            maxpair = ring
            ai = (<DTYPE0_t*>((<char*>pid(ita))))[0]
            maxpair.value = ai
            maxpair.death = window

            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if maxpair.death == i:
                    maxpair += 1
                    if maxpair >= end:
                        maxpair = ring
                if ai >= maxpair.value:
                    maxpair.value = ai
                    maxpair.death = i + window
                    last = maxpair
                else:
                    while last.value <= ai:
                        if last == ring:
                            last = end
                        last -= 1
                    last += 1
                    if last == end:
                        last = ring
                    last.value = ai
                    last.death = i + window
                if i + 1 >= min_count:
                    yi = i - maxpair.death + window
                else:
                    yi = NAN
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)

        stdlib.free(ring)
    return y


# move_median ---------------------------------------------------------------

def move_median(arr, int window, min_count=None, axis=-1):
    """
    Moving window median along the specified axis, optionally ignoring NaNs.

    float64 output is returned for all input data types.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving median of the input array along the specified axis. The
        output has the same shape as the input.

    Examples
    --------
    >>> arr = np.array([1.0, 2.0, 3.0, 4.0])
    >>> bn.move_median(arr, window=2)
    array([ nan,  1.5,  2.5,  3.5])
    >>> bn.move_median(arr, window=2, min_count=1)
    array([ 1. ,  1.5,  2.5,  3.5])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_median_float64,
                     move_median_float32,
                     move_median_int64,
                     move_median_int32)
    except TypeError:
        return slow.move_median(arr, window, min_count, axis)


cdef extern from "csrc/move_median.c":
    struct _mm_node:
        np.npy_uint32   small
        np.npy_uint64   idx
        np.npy_float64  ai
        _mm_node       *next
    ctypedef _mm_node mm_node
    struct _mm_handle:
        np.npy_uint64    window
        int              odd
        np.npy_uint64    min_count
        np.npy_uint64    n_s
        np.npy_uint64    n_l
        np.npy_uint64    n_n
        mm_node        **s_heap
        mm_node        **l_heap
        mm_node        **n_array
        mm_node        **nodes
        mm_node         *node_data
        mm_node         *first
        mm_node         *last
        np.npy_uint64    s_first_leaf
        np.npy_uint64    l_first_leaf
    ctypedef _mm_handle mm_handle
    mm_handle *mm_new(np.npy_uint64 window, np.npy_uint64 min_count) nogil
    np.npy_float64 mm_update_init(mm_handle *mm, np.npy_float64 val) nogil
    np.npy_float64 mm_update(mm_handle* mm, np.npy_float64 val) nogil
    mm_handle *mm_new_nan(np.npy_uint64 window, np.npy_uint64 min_count) nogil
    np.npy_float64 mm_update_init_nan(mm_handle *mm, np.npy_float64 val) nogil
    np.npy_float64 mm_update_nan(mm_handle* mm, np.npy_float64 val) nogil
    void mm_reset(mm_handle* mm) nogil
    void mm_free(mm_handle *mm) nogil


@cython.cdivision(True)
cdef ndarray move_median_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64', 'float64'], ['float32', 'float32']]
    cdef mm_handle *mm
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    if window == 1:
        return PyArray_Copy(a)
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    mm = mm_new_nan(window, min_count)
    if mm is NULL:
        raise MemoryError()
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                yi = mm_update_init_nan(mm, ai)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                yi = mm_update_nan(mm, ai)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
            mm_reset(mm)
        mm_free(mm)
    return y


@cython.cdivision(True)
cdef ndarray move_median_DTYPE0(ndarray a, int window, int min_count, int axis,
                                flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int a_ndim,
                                npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef mm_handle *mm
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    if window == 1:
        return a.astype(np.float64)
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    mm = mm_new(window, min_count)
    if mm is NULL:
        raise MemoryError()
    with nogil:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(window):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                yi = mm_update_init(mm, ai)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            for i in range(window, length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                yi = mm_update(mm, ai)
                (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
            mm_reset(mm)
        mm_free(mm)
    return y


# move_rank-----------------------------------------------------------------

def move_rank(arr, int window, min_count=None, int axis=-1):
    """
    Moving window ranking along the specified axis, optionally ignoring NaNs.

    The output is normalized to be between -1 and 1. For example, with a
    window width of 3 (and with no ties), the possible output values are
    -1, 0, 1.

    Ties are broken by averaging the rankings. See the examples below.

    The runtime depends almost linearly on `window`. The more NaNs there are
    in the input array, the shorter the runtime.

    Parameters
    ----------
    arr : ndarray
        Input array. If `arr` is not an array, a conversion is attempted.
    window : int
        The number of elements in the moving window.
    min_count: {int, None}, optional
        If the number of non-NaN values in a window is less than `min_count`,
        then a value of NaN is assigned to the window. By default `min_count`
        is None, which is equivalent to setting `min_count` equal to `window`.
    axis : int, optional
        The axis over which the window is moved. By default the last axis
        (axis=-1) is used. An axis of None is not allowed.

    Returns
    -------
    y : ndarray
        The moving ranking along the specified axis. The output has the same
        shape as the input. For integer input arrays, the dtype of the output
        is float64.

    Examples
    --------
    With window=3 and no ties, there are 3 possible output values, i.e.
    [-1., 0., 1.]:

    >>> arr = np.array([1, 2, 3, 9, 8, 7, 5, 6, 4])
    >>> bn.move_rank(arr, window=3)
        array([ nan,  nan,   1.,   1.,   0.,  -1.,  -1.,   0.,  -1.])

    Ties are broken by averaging the rankings of the tied elements:

    >>> arr = np.array([1, 2, 3, 3, 3, 4])
    >>> bn.move_rank(arr, window=3)
        array([ nan,  nan,  1. ,  0.5,  0. ,  1. ])

    In an increasing sequence, the moving window ranking is always equal to 1:

    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> bn.move_rank(arr, window=2)
        array([ nan,   1.,   1.,   1.,   1.])

    """
    try:
        return mover(arr, window, min_count, axis,
                     move_rank_float64,
                     move_rank_float32,
                     move_rank_int64,
                     move_rank_int32,
                     0)
    except TypeError:
        return slow.move_rank(arr, window, min_count, axis)


@cython.cdivision(True)
cdef ndarray move_rank_DTYPE0(ndarray a, int window, int min_count, int axis,
                              flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int a_ndim,
                              npy_intp* y_dims, int ignore):
    # bn.dtypes = [['float64', 'float64'], ['float32', 'float32']]
    cdef Py_ssize_t i, j
    cdef DTYPE0_t ai, aj
    cdef DTYPE1_t g, e, n, r
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    cdef char* pita
    while PyArray_ITER_NOTDONE(ita):
        for i in range(min_count - 1):
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        for i in range(min_count - 1, window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                g = 0
                e = 1
                n = 1
                r = 0
                for j in range(i):
                    aj = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                    if aj == aj:
                        n += 1
                        if ai > aj:
                            g += 1
                        elif ai == aj:
                            e += 1
                if n < min_count:
                    r = NAN
                elif n == 1:
                    r = 0.0
                else:
                    r = (g + g + e - 1.0) / 2.0
                    r = r / (n - 1.0)
                    r = 2.0 * (r - 0.5)
            else:
                r = NAN
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = r
        for i in range(window - 1, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                g = 0
                e = 1
                n = 1
                r = 0
                for j in range(i - window + 1, i):
                    aj = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                    if aj == aj:
                        n += 1
                        if ai > aj:
                            g += 1
                        elif ai == aj:
                            e += 1
                if n < min_count:
                    r = NAN
                elif n == 1:
                    r = 0.0
                else:
                    r = (g + g + e - 1.0) / 2.0
                    r = r / (n - 1.0)
                    r = 2.0 * (r - 0.5)
            else:
                r = NAN
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = r
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_rank_DTYPE0(ndarray a, int window, int min_count, int axis,
                              flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int a_ndim,
                              npy_intp* y_dims, int ignore):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, j
    cdef DTYPE0_t ai, aj
    cdef DTYPE1_t g, e, n, r
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    cdef char* pita
    while PyArray_ITER_NOTDONE(ita):
        for i in range(min_count - 1):
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        for i in range(min_count - 1, window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            g = 0
            e = 1
            n = 1
            r = 0
            for j in range(i):
                aj = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                n += 1
                if ai > aj:
                    g += 1
                elif ai == aj:
                    e += 1
            if n < min_count:
                r = NAN
            elif n == 1:
                r = 0.0
            else:
                r = (g + g + e - 1.0) / 2.0
                r = r / (n - 1.0)
                r = 2.0 * (r - 0.5)
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = r
        for i in range(window - 1, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            g = 0
            e = 1
            n = 1
            r = 0
            for j in range(i - window + 1, i):
                aj = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                n += 1
                if ai > aj:
                    g += 1
                elif ai == aj:
                    e += 1
            if n < min_count:
                r = NAN
            elif n == 1:
                r = 0.0
            else:
                r = (g + g + e - 1.0) / 2.0
                r = r / (n - 1.0)
                r = 2.0 * (r - 0.5)
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = r
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


# mover ---------------------------------------------------------------------

ctypedef ndarray (*move_t)(ndarray, int, int, int, flatiter, Py_ssize_t,
                           Py_ssize_t, int, npy_intp*, int)


cdef ndarray mover(arr, int window, min_count, int axis,
                   move_t move_float64,
                   move_t move_float32,
                   move_t move_int64,
                   move_t move_int32,
                   int int_input=0):

    # convert to array if necessary
    cdef ndarray a
    if PyArray_Check(arr):
        a = arr
    else:
        a = PyArray_FROM_O(arr)

    # check for byte swapped input array
    if PyArray_ISBYTESWAPPED(a):
        raise TypeError

    # min_count
    cdef int mc
    if min_count is None:
        mc = window
    else:
        mc = <int>min_count
        if mc > window:
            msg = "min_count (%d) cannot be greater than window (%d)"
            raise ValueError(msg % (mc, window))
        elif mc <= 0:
            raise ValueError("`min_count` must be greater than zero.")

    # input array
    cdef flatiter ita
    cdef Py_ssize_t stride, length
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # defend against 0d beings
    if a_ndim == 0:
        raise ValueError("moving window functions require ndim > 0")

    # defend against the axis of negativity
    if axis < 0:
        axis += a_ndim
        if axis < 0:
            raise ValueError("axis(=%d) out of bounds" % axis)
    elif axis >= a_ndim:
        raise ValueError("axis(=%d) out of bounds" % axis)

    # output array info
    cdef ndarray y
    cdef npy_intp *y_dims = PyArray_DIMS(a)

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis)
    stride = PyArray_STRIDE(a, axis)
    length = y_dims[axis]

    # window
    if (window < 1) or (window > length):
        msg = "Moving window (=%d) must between 1 and %d, inclusive"
        raise ValueError(msg % (window, length))

    if dtype == NPY_float64:
        y = move_float64(a, window, mc, axis, ita, stride, length,
                         a_ndim, y_dims, int_input)
    elif dtype == NPY_float32:
        y = move_float32(a, window, mc, axis, ita, stride, length,
                         a_ndim, y_dims, int_input)
    elif dtype == NPY_int64:
        y = move_int64(a, window, mc, axis, ita, stride, length,
                       a_ndim, y_dims, int_input)
    elif dtype == NPY_int32:
        y = move_int32(a, window, mc, axis, ita, stride, length,
                       a_ndim, y_dims, int_input)
    else:
        raise TypeError
    return y
