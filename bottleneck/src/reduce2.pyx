import numpy as np
cimport numpy as np
import cython

from numpy cimport NPY_FLOAT64, NPY_FLOAT32, NPY_INT64, NPY_INT32
from numpy cimport float64_t, float32_t, int64_t, int32_t

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM

from numpy cimport ndarray
from numpy cimport import_array
import_array()

cdef dict dtype_dict = {}
dtype_dict[NPY_FLOAT64] = np.float64
dtype_dict[NPY_FLOAT32] = np.float32
dtype_dict[NPY_INT64] = np.int64
dtype_dict[NPY_INT32] = np.int32


def nansum_all_float64(np.flatiter ita, Py_ssize_t stride,
                                  Py_ssize_t length):
    "reduce along all axes"
    cdef Py_ssize_t i
    cdef float64_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<float64_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
        PyArray_ITER_NEXT(ita)
    return asum


def nansum_all_float32(np.flatiter ita, Py_ssize_t stride,
                                  Py_ssize_t length):
    "reduce along all axes"
    cdef Py_ssize_t i
    cdef float32_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<float32_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
        PyArray_ITER_NEXT(ita)
    return asum


def nansum_all_int64(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length):
    "reduce along all axes"
    cdef Py_ssize_t i
    cdef int64_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<int64_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai
        PyArray_ITER_NEXT(ita)
    return asum


def nansum_all_int32(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length):
    "reduce along all axes"
    cdef Py_ssize_t i
    cdef int32_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<int32_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai
        PyArray_ITER_NEXT(ita)
    return asum


def nansum_one_float64(np.flatiter ita, np.flatiter ity,
                             Py_ssize_t stride, Py_ssize_t length):
    "reduce along a single axis; ndim > 1"
    cdef Py_ssize_t i
    cdef float64_t asum, ai
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<float64_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
        (<float64_t*>((<char*>pid(ity))))[0] = asum
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)


def nansum_one_float32(np.flatiter ita, np.flatiter ity,
                             Py_ssize_t stride, Py_ssize_t length):
    "reduce along a single axis; ndim > 1"
    cdef Py_ssize_t i
    cdef float32_t asum, ai
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<float32_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
        (<float32_t*>((<char*>pid(ity))))[0] = asum
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)


def nansum_one_int64(np.flatiter ita, np.flatiter ity,
                           Py_ssize_t stride, Py_ssize_t length):
    "reduce along a single axis; ndim > 1"
    cdef Py_ssize_t i
    cdef int64_t asum, ai
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<int64_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
        (<int64_t*>((<char*>pid(ity))))[0] = asum
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)


def nansum_one_int32(np.flatiter ita, np.flatiter ity,
                           Py_ssize_t stride, Py_ssize_t length):
    "reduce along a single axis; ndim > 1"
    cdef Py_ssize_t i
    cdef int32_t asum, ai
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<int32_t*>((<char*>pid(ita)) + i*stride))[0]
            asum += ai
        (<int32_t*>((<char*>pid(ity))))[0] = asum
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)


cdef dict nansum_all = {}
nansum_all[NPY_FLOAT64] = nansum_all_float64
nansum_all[NPY_FLOAT32] = nansum_all_float32
nansum_all[NPY_INT64] = nansum_all_int64
nansum_all[NPY_INT32] = nansum_all_int32
cdef dict nansum_one = {}
nansum_one[NPY_FLOAT64] = nansum_one_float64
nansum_one[NPY_FLOAT32] = nansum_one_float32
nansum_one[NPY_INT64] = nansum_one_int64
nansum_one[NPY_INT32] = nansum_one_int32
cdef dict nansum_dict = {}
nansum_dict[0] = nansum_all
nansum_dict[1] = nansum_one


cdef reducer(dict func_dict, arr, axis):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i
    cdef int dtype = PyArray_TYPE(a)
    cdef int ndim = PyArray_NDIM(a)

    # output array, if needed
    cdef list shape = []
    cdef ndarray y
    cdef np.flatiter ity

    # defend against 0d beings
    if ndim == 0:
        if axis is None or axis == 0 or axis == -1:
            out = a[()]
            if out == out:
                return out
            else:
                return 0.0
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
            axis_int += ndim
            if axis_int < 0:
                raise ValueError("axis(=%d) out of bounds" % axis)
        if ndim == 1 and axis_int == 0:
            reduce_all = 1
        axis_reduce = axis_int

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis_reduce)
    stride = a.strides[axis_reduce]
    length = a.shape[axis_reduce]

    if reduce_all == 1:
        # reduce over all axes
        try:
            return func_dict[0][dtype](ita, stride, length)
        except KeyError:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
    else:
        # reduce over a single axis; ndim > 1
        for i in range(ndim):
            if i != axis_int:
                shape.append(a.shape[i])
        try:
            y = np.empty(shape, dtype_dict[dtype])
            ity = PyArray_IterNew(y)
            func_dict[1][dtype](ita, ity, stride, length)
        except KeyError:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
        return y


def nansum(arr, axis=None):
    return reducer(nansum_dict, arr, axis)
