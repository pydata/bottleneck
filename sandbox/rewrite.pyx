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

cdef double NAN = <double> NAN
cdef int axis_negone = -1


ctypedef fused bntype:
    float64_t
    float32_t
    int64_t
    int32_t

cdef float64_t f64 = 1.0
cdef float32_t f32 = 1.0
cdef int64_t i64 = 1
cdef int32_t i32 = 1


cdef inline bntype nansum_all(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, bntype dt):
    "reduce along all axes"
    cdef Py_ssize_t i
    cdef bntype asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<bntype*>((<char*>pid(ita)) + i * stride))[0]
            if bntype is float64_t:
                if ai == ai:
                    asum += ai
            if bntype is float32_t:
                if ai == ai:
                    asum += ai
            if bntype is int64_t:
                asum += ai
            if bntype is int32_t:
                asum += ai
        PyArray_ITER_NEXT(ita)
    return asum


cdef inline void nansum_one(np.flatiter ita, np.flatiter ity,
                            Py_ssize_t stride, Py_ssize_t length,
                            bntype dt):
    "reduce along a single axis; ndim > 1"
    cdef Py_ssize_t i
    cdef bntype asum, ai
    while PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<bntype*>((<char*>pid(ita)) + i*stride))[0]
            if bntype is float64_t:
                if ai == ai:
                    asum += ai
            if bntype is float32_t:
                if ai == ai:
                    asum += ai
            if bntype is int64_t:
                asum += ai
            if bntype is int32_t:
                asum += ai
        (<double*>((<char*>pid(ity))))[0] = asum
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)


def nansum(arr, axis=None):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i
    cdef int dtype = PyArray_TYPE(a)
    cdef int ndim = PyArray_NDIM(a)

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

    # reduce over all axes: axis=None
    if axis is None:
        ita = PyArray_IterAllButAxis(a, &axis_negone)
        stride = a.strides[axis_negone]
        length = a.shape[axis_negone]
        if dtype == NPY_FLOAT64:
            return nansum_all(ita, stride, length, f64)
        if dtype == NPY_FLOAT32:
            return nansum_all(ita, stride, length, f32)
        if dtype == NPY_INT64:
            return nansum_all(ita, stride, length, i64)
        if dtype == NPY_INT32:
            return nansum_all(ita, stride, length, i32)
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    # what if axis is a float or such?
    cdef int axis_int = <int>axis

    # defend against the axis of negativity
    if axis_int < 0:
        axis_int += ndim
        if axis_int < 0:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # reduce over all axes: 1d array, axis=0
    if ndim == 1 and axis_int == 0:
        ita = PyArray_IterAllButAxis(a, &axis_negone)
        stride = a.strides[axis_negone]
        length = a.shape[axis_negone]
        if dtype == NPY_FLOAT64:
            return nansum_all(ita, stride, length, f64)
        if dtype == NPY_FLOAT32:
            return nansum_all(ita, stride, length, f32)
        if dtype == NPY_INT64:
            return nansum_all(ita, stride, length, i64)
        if dtype == NPY_INT32:
            return nansum_all(ita, stride, length, i32)
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    # if we've made it this far then ndim > 1

    # iterators
    ita = PyArray_IterAllButAxis(a, &axis_int)
    stride = a.strides[axis_int]
    length = a.shape[axis_int]

    # shape of output, y
    cdef list shape = []
    for i in range(ndim):
        if i != axis_int:
            shape.append(a.shape[i])

    cdef np.flatiter ity

    if dtype == NPY_FLOAT64:
        y = np.empty(shape, np.float64)
        ity = PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, f64)
        return y
    if dtype == NPY_FLOAT32:
        y = np.empty(shape, np.float64)
        ity = PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, f32)
        return y
    if dtype == NPY_INT64:
        y = np.empty(shape, np.float64)
        ity = PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, i64)
        return y
    if dtype == NPY_INT32:
        y = np.empty(shape, np.float64)
        ity = PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, i32)
        return y
    raise TypeError("Unsupported dtype (%s)." % a.dtype)
