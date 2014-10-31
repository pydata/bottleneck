import numpy as np
cimport numpy as np
import cython
from numpy cimport NPY_FLOAT64
from numpy cimport NPY_FLOAT32
from numpy cimport NPY_INT32
from numpy cimport NPY_INT64
from numpy cimport PyArray_EMPTY, PyArray_DIMS, import_array
from numpy cimport PyArray_ITER_DATA as pid
import_array()
cdef double NAN = <double> NAN
cdef int axis_negone = -1


ctypedef fused bntype:
    np.float64_t
    np.float32_t
    np.int64_t
    np.int32_t

cdef np.float64_t f64 = 1.0
cdef np.float32_t f32 = 1.0
cdef np.int64_t i64 = 1
cdef np.int32_t i32 = 1


def nansum(arr, axis=None):

    # convert to array if necessary
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i
    cdef int dtype = np.PyArray_TYPE(a)
    cdef int ndim = np.PyArray_NDIM(a)

    # defend against 0d beings
    if ndim == 0:
        if axis is None or axis == 0 or axis == -1:
            # TODO what if single element is NaN; should return 0 then
            return a[()]
        else:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # reduce over all axes: axis=None
    if axis is None:
        ita = np.PyArray_IterAllButAxis(a, &axis_negone)
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

    # defend against the evils of negative axes
    if axis_int < 0:
        axis_int += ndim
        if axis_int < 0:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # reduce over all axes: 1d array, axis=0
    if ndim == 1 and axis_int == 0:
        ita = np.PyArray_IterAllButAxis(a, &axis_negone)
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

    # reduce over a single axis
    ita = np.PyArray_IterAllButAxis(a, &axis_int)
    stride = a.strides[axis_int]
    length = a.shape[axis_int]

    # temp python hack
    cdef list shape = []
    for i in range(ndim):
        if i != axis_int:
            shape.append(a.shape[i])

    cdef np.flatiter ity

    if dtype == NPY_FLOAT64:
        y = np.empty(shape, np.float64)
        ity = np.PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, f64)
        return y
    if dtype == NPY_FLOAT32:
        y = np.empty(shape, np.float64)
        ity = np.PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, f32)
        return y
    if dtype == NPY_INT64:
        y = np.empty(shape, np.float64)
        ity = np.PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, i64)
        return y
    if dtype == NPY_INT32:
        y = np.empty(shape, np.float64)
        ity = np.PyArray_IterNew(y)
        nansum_one(ita, ity, stride, length, i32)
        return y
    raise TypeError("Unsupported dtype (%s)." % a.dtype)


cdef inline bntype nansum_all(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, bntype dt):
    cdef Py_ssize_t i
    cdef bntype asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<bntype*>((<char*>pid(ita)) + i * stride))[0]
            if bntype is np.float64_t:
                if ai == ai:
                    asum += ai
            if bntype is np.float32_t:
                if ai == ai:
                    asum += ai
            if bntype is np.int64_t:
                asum += ai
            if bntype is np.int32_t:
                asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.ndarray nansum_one(np.flatiter ita, np.flatiter ity,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  bntype dt):
    cdef Py_ssize_t i
    cdef bntype asum, ai
    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<bntype*>((<char*>pid(ita)) + i*stride))[0]
            if bntype is np.float64_t:
                if ai == ai:
                    asum += ai
            if bntype is np.float32_t:
                if ai == ai:
                    asum += ai
            if bntype is np.int64_t:
                asum += ai
            if bntype is np.int32_t:
                asum += ai
        (<double*>((<char*>pid(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)
