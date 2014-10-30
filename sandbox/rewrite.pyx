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


ctypedef fused bntypes:
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
        if dtype == NPY_FLOAT64:
            return nansum_axisNone(a, f64)
        if dtype == NPY_FLOAT32:
            return nansum_axisNone(a, f32)
        if dtype == NPY_INT64:
            return nansum_axisNone(a, i64)
        if dtype == NPY_INT32:
            return nansum_axisNone(a, i32)
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
        if dtype == NPY_FLOAT64:
            return nansum_axisNone(a, f64)
        if dtype == NPY_FLOAT32:
            return nansum_axisNone(a, f32)
        if dtype == NPY_INT64:
            return nansum_axisNone(a, i64)
        if dtype == NPY_INT32:
            return nansum_axisNone(a, i32)
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    # reduce over a single axis
    if dtype == NPY_FLOAT64:
        return nansum_axisint(a, axis_int, f64)
    if dtype == NPY_FLOAT32:
        return nansum_axisint(a, axis_int, f32)
    if dtype == NPY_INT64:
        return nansum_axisint(a, axis_int, i64)
    if dtype == NPY_INT32:
        return nansum_axisint(a, axis_int, i32)
    raise TypeError("Unsupported dtype (%s)." % a.dtype)


cdef inline bntypes nansum_axisNone(np.ndarray a, bntypes dt):
    cdef int axis=-1
    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis], length = a.shape[axis], i
    cdef bntypes asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<bntypes*>((<char*>pid(ita)) + i * stride))[0]
            if bntypes is np.float64_t:
                if ai == ai:
                    asum += ai
            elif bntypes is np.int64_t:
                asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.ndarray nansum_axisint(np.ndarray a, int axis, bntypes dt):

    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t length = a.shape[axis], i
    cdef int stride = a.strides[axis]

    # temp python hack
    cdef int ndim = np.PyArray_NDIM(a)
    cdef list shape = []
    for i in range(ndim):
        if i != axis:
            shape.append(a.shape[i])

    # TODO: if tree for dtype
    y = np.empty(shape, )
    cdef np.flatiter ity = np.PyArray_IterNew(y)

    cdef bntypes asum, ai

    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<bntypes*>((<char*>pid(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
        (<double*>((<char*>pid(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)

    return y
