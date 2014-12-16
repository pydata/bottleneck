#cython: embedsignature=True

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM

from numpy cimport PyArray_EMPTY
from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.move as slow

cdef double NAN = <double> np.nan
cdef extern from "math.h":
    double sqrt(double x)

cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float32_t MINfloat32 = -np.inf
cdef np.float64_t MINfloat64 = -np.inf


# move_sum -----------------------------------------------------------------

def move_sum(arr, int window, int axis=-1):
    try:
        return mover(arr, window, axis,
                     move_sum_float64,
                     move_sum_float32,
                     move_sum_int64,
                     move_sum_int32)
    except TypeError:
        return slow.move_sum(arr, window, axis)


cdef ndarray move_sum_DTYPE0(int window, int axis, np.flatiter ita,
                             Py_ssize_t stride, Py_ssize_t length,
                             int a_ndim, np.npy_intp* y_dims,
                             int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            count += 1
        if count == window:
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
            if count == window:
                yi = asum
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray move_sum_DTYPE0(int window, int axis, np.flatiter ita,
                              Py_ssize_t stride, Py_ssize_t length,
                              int a_ndim, np.npy_intp* y_dims,
                              int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum, aold, yi
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
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


# move_nansum --------------------------------------------------------------

def move_nansum(arr, int window, int axis=-1):
    try:
        return mover(arr, window, axis,
                     move_nansum_float64,
                     move_nansum_float32,
                     move_sum_int64,
                     move_sum_int32)
    except TypeError:
        return slow.move_nansum(arr, window, axis)


cdef ndarray move_nansum_DTYPE0(int window, int axis, np.flatiter ita,
                                 Py_ssize_t stride, Py_ssize_t length,
                                 int a_ndim, np.npy_intp* y_dims,
                                 int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
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
            if count > 0:
                yi = asum
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


# move_mean -----------------------------------------------------------------

def move_mean(arr, int window, int axis=-1):
    try:
        return mover(arr, window, axis,
                     move_mean_float64,
                     move_mean_float32,
                     move_mean_int64,
                     move_mean_int32)
    except TypeError:
        return slow.move_mean(arr, window, axis)


@cython.cdivision(True)
cdef ndarray move_mean_DTYPE0(int window, int axis, np.flatiter ita,
                              Py_ssize_t stride, Py_ssize_t length,
                              int a_ndim, np.npy_intp* y_dims,
                              int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            count += 1
        if count == window:
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
            if count == window:
                yi = asum / count
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_mean_DTYPE0(int window, int axis, np.flatiter ita,
                              Py_ssize_t stride, Py_ssize_t length,
                              int a_ndim, np.npy_intp* y_dims,
                              int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum, aold, yi
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        asum += ai
        yi = asum / window
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


# move_nanmean --------------------------------------------------------------

def move_nanmean(arr, int window, int axis=-1):
    try:
        return mover(arr, window, axis,
                     move_nanmean_float64,
                     move_nanmean_float32,
                     move_mean_int64,
                     move_mean_int32)
    except TypeError:
        return slow.move_nanmean(arr, window, axis)


@cython.cdivision(True)
cdef ndarray move_nanmean_DTYPE0(int window, int axis, np.flatiter ita,
                                 Py_ssize_t stride, Py_ssize_t length,
                                 int a_ndim, np.npy_intp* y_dims,
                                 int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, ai, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            count += 1
        if count > 0:
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
            if count > 0:
                yi = asum / count
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


# move_std -----------------------------------------------------------------

def move_std(arr, int window, int axis=-1, int ddof=0):
    try:
        return mover(arr, window, axis,
                     move_std_float64,
                     move_std_float32,
                     move_std_int64,
                     move_std_int32,
                     ddof)
    except TypeError:
        return slow.move_std(arr, window, axis, ddof)


@cython.cdivision(True)
cdef ndarray move_std_DTYPE0(int window, int axis, np.flatiter ita,
                             Py_ssize_t stride, Py_ssize_t length,
                             int a_ndim, np.npy_intp* y_dims,
                             int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, a2sum, ai, ssr, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        a2sum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            a2sum += ai * ai
            count += 1
        if count == window:
            ssr = a2sum - asum * asum / count
            if ssr < 0:
                yi = 0
            else:
                yi = sqrt(ssr / (count - ddof))
        else:
            yi = NAN
        (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        for i in range(window, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
            if aold == aold:
                asum -= aold
                a2sum -= aold * aold
                count -= 1
            if count == window:
                ssr = a2sum - asum * asum / count
                if ssr < 0:
                    yi = 0
                else:
                    yi = sqrt(ssr / (count - ddof))
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray move_std_DTYPE0(int window, int axis, np.flatiter ita,
                             Py_ssize_t stride, Py_ssize_t length,
                             int a_ndim, np.npy_intp* y_dims,
                             int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef int winddof
    cdef DTYPE0_t ai, aold
    cdef DTYPE1_t yi, asum, a2sum
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    winddof = window - ddof
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        a2sum = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
            a2sum += ai * ai
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        asum += ai
        a2sum += ai * ai
        yi = sqrt((a2sum - asum * asum / window) / winddof)
        (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        for i in range(window, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
            a2sum += ai * ai
            aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
            asum -= aold
            a2sum -= aold * aold
            yi = sqrt((a2sum - asum * asum / window) / winddof)
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


# move_nanstd --------------------------------------------------------------

def move_nanstd(arr, int window, int axis=-1, int ddof=0):
    try:
        return mover(arr, window, axis,
                     move_nanstd_float64,
                     move_nanstd_float32,
                     move_std_int64,
                     move_std_int32,
                     ddof)
    except TypeError:
        return slow.move_nanstd(arr, window, axis, ddof)


@cython.cdivision(True)
cdef ndarray move_nanstd_DTYPE0(int window, int axis, np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum, a2sum, ai, ssr, aold, yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        a2sum = 0
        count = 0
        for i in range(window - 1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        i = window - 1
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai == ai:
            asum += ai
            a2sum += ai * ai
            count += 1
        if count > 0:
            ssr = a2sum - asum * asum / count
            if ssr < 0:
                yi = 0
            else:
                yi = sqrt(ssr / (count - ddof))
        else:
            yi = NAN
        (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        for i in range(window, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
                a2sum += ai * ai
                count += 1
            aold = (<DTYPE0_t*>((<char*>pid(ita)) + (i-window)*stride))[0]
            if aold == aold:
                asum -= aold
                a2sum -= aold * aold
                count -= 1
            if count > 0:
                ssr = a2sum - asum * asum / count
                if ssr < 0:
                    yi = 0
                else:
                    yi = sqrt(ssr / (count - ddof))
            else:
                yi = NAN
            (<DTYPE0_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


# move_median ---------------------------------------------------------------

cdef extern from "csrc/move_median.c":
    struct _mm_node:
        np.npy_uint32   small
        np.npy_uint64   idx
        np.npy_float64  val
        _mm_node         *next
    ctypedef _mm_node mm_node
    struct _mm_handle:
        int              odd
        np.npy_uint64    n_s
        np.npy_uint64    n_l
        mm_node          **s_heap
        mm_node          **l_heap
        mm_node          **nodes
        mm_node           *node_data
        mm_node           *first
        mm_node           *last
        np.npy_uint64 s_first_leaf
        np.npy_uint64 l_first_leaf
    ctypedef _mm_handle mm_handle
    mm_handle *mm_new(np.npy_uint64 size) nogil
    void mm_insert_init(mm_handle *mm, np.npy_float64 val) nogil
    void mm_update(mm_handle *mm, np.npy_float64 val) nogil
    np.npy_float64 mm_get_median(mm_handle *mm) nogil
    void mm_free(mm_handle *mm) nogil


def move_median(arr, int window, int axis=-1):
    try:
        return mover(arr, window, axis,
                     move_median_float64,
                     move_median_float32,
                     move_median_int64,
                     move_median_int32)
    except TypeError:
        return slow.move_median(arr, window, axis)


@cython.cdivision(True)
cdef ndarray move_median_DTYPE0(int window, int axis, np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int ignore):
    # bn.dtypes = [['float64', 'float64'], ['float32', 'float32'], ['int64', 'float64'], ['int32', 'float64']]
    cdef mm_handle *mm
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef DTYPE1_t yi
    cdef ndarray y = PyArray_EMPTY(a_ndim, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    cdef Py_ssize_t ystride = y.strides[axis]
    mm = mm_new(window)
    while PyArray_ITER_NOTDONE(ita):
        for i in range(window - 1):
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = NAN
        for i in range(window):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            mm_insert_init(mm, ai)
        yi = mm_get_median(mm)
        (<DTYPE1_t*>((<char*>pid(ity)) + (window - 1)*ystride))[0] = yi
        for i in range(window, length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            mm_update(mm, ai)
            yi = mm_get_median(mm)
            (<DTYPE1_t*>((<char*>pid(ity)) + i*ystride))[0] = yi
        mm.n_s = 0
        mm.n_l = 0
        mm_free(mm)
    return y


# mover ---------------------------------------------------------------------

ctypedef ndarray (*move_t)(int, int, np.flatiter, Py_ssize_t, Py_ssize_t, int,
                           np.npy_intp*, int)


cdef ndarray mover(arr, int window, int axis,
                   move_t move_float64,
                   move_t move_float32,
                   move_t move_int64,
                   move_t move_int32,
                   int int_input=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    cdef np.flatiter ita
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
    cdef np.npy_intp *y_dims = np.PyArray_DIMS(a)

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis)
    stride = a.strides[axis]
    length = a.shape[axis]

    if (window < 1) or (window > length):
        msg = "Moving window (=%d) must between 1 and %d, inclusive"
        raise ValueError(msg % (window, length))

    if dtype == NPY_float64:
        y = move_float64(window, axis, ita, stride, length, a_ndim, y_dims,
                         int_input)
    elif dtype == NPY_float32:
        y = move_float32(window, axis, ita, stride, length, a_ndim, y_dims,
                         int_input)
    elif dtype == NPY_int64:
        y = move_int64(window, axis, ita, stride, length, a_ndim, y_dims,
                       int_input)
    elif dtype == NPY_int32:
        y = move_int32(window, axis, ita, stride, length, a_ndim, y_dims,
                       int_input)
    else:
        raise TypeError("Unsupported dtype (%s)." % a.dtype)
    return y
