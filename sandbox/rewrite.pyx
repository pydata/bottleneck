import numpy as np
cimport numpy as np
import cython
from numpy cimport NPY_FLOAT64
from numpy cimport NPY_FLOAT32
from numpy cimport NPY_INT32
from numpy cimport NPY_INT64
from numpy cimport PyArray_EMPTY, PyArray_DIMS, import_array
import_array()
cdef double NAN = <double> NAN

if np.int_ == np.int32:
    NPY_int_ = NPY_INT32
elif np.int_ == np.int64:
    NPY_int_ = NPY_INT64
else:
    raise RuntimeError('Expecting default NumPy int to be 32 or 64 bit.')


def nansum(arr, axis=None):

    # convert to array if necessary
    cdef np.ndarray a
    if type(arr) is np.ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # dtype
    cdef int dtype = np.PyArray_TYPE(a)
    if dtype < NPY_int_:
        a = a.astype(np.int_)
        dtype = np.PyArray_TYPE(a)

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
            return nansum_float64_axisNone(a)
        if dtype == NPY_FLOAT32:
            return nansum_float32_axisNone(a)
        if dtype == NPY_INT64:
            return nansum_int64_axisNone(a)
        if dtype == NPY_INT32:
            return nansum_int32_axisNone(a)
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    # what if axis is a float or such?
    cdef int axis_int = <int>axis

    # reduce over all axes: 1d array, axis=0
    if ndim == 1 and axis_int == 0:
        if dtype == NPY_FLOAT64:
            return nansum_float64_axisNone(a)
        if dtype == NPY_FLOAT32:
            return nansum_float32_axisNone(a)
        if dtype == NPY_INT64:
            return nansum_int64_axisNone(a)
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    # defend against the evils of negative axes
    if axis < 0:
        axis += ndim
    if (axis >= ndim) or (axis < 0):
        raise ValueError("axis(=%d) out of bounds" % axis)

    # reduce over a single axis
    if dtype == NPY_FLOAT64:
        return nansum_float64_axisint(a, axis_int)
    if dtype == NPY_FLOAT32:
        return nansum_float32_axisint(a, axis_int)
    if dtype == NPY_INT64:
        return nansum_int64_axisint(a, axis_int)
    if dtype == NPY_INT32:
        return nansum_int64_axisint(a, axis_int)
    raise TypeError("Unsupported dtype (%s)." % a.dtype)


cdef inline np.float64_t nansum_float64_axisNone(np.ndarray a):
    cdef int axis=-1
    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis], length = a.shape[axis], i
    cdef np.float64_t asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<np.float64_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.float32_t nansum_float32_axisNone(np.ndarray a):
    cdef int axis=-1
    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis], length = a.shape[axis], i
    cdef np.float32_t asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<np.float32_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.int64_t nansum_int64_axisNone(np.ndarray a):
    cdef int axis=-1
    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis], length = a.shape[axis], i
    cdef np.int64_t asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<np.int64_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i * stride))[0]
            asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.int32_t nansum_int32_axisNone(np.ndarray a):
    cdef int axis=-1
    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis], length = a.shape[axis], i
    cdef np.int32_t asum = 0, ai
    while np.PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<np.int32_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i * stride))[0]
            asum += ai
        np.PyArray_ITER_NEXT(ita)
    return asum


cdef inline np.ndarray nansum_float64_axisint(np.ndarray a, int axis):


    # temp hack
    cdef int ndim = 2

    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t length = a.shape[axis], i
    cdef int stride = a.strides[axis]

    # temp python hack
    cdef list shape = []
    for i in range(ndim):
        if i != axis:
            shape.append(a.shape[i])

    y = np.empty(shape, np.float64)
    cdef np.flatiter ity = np.PyArray_IterNew(y)

    cdef np.float64_t asum, ai

    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<np.float64_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
        (<double*>((<char*>np.PyArray_ITER_DATA(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)

    return y


cdef inline np.ndarray nansum_float32_axisint(np.ndarray a, int axis):

    cdef int ndim = a.ndim
    if ndim <= 1:
        raise ValueError("`a.ndim` must be greater than 1")
    if axis < 0:
        axis += ndim
    if (axis >= ndim) or (axis < 0):
        raise ValueError("axis(=%d) out of bounds" % axis)

    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t length = a.shape[axis], i
    cdef int stride = a.strides[axis]

    # temp python hack
    cdef list shape = []
    for i in range(ndim):
        if i != axis:
            shape.append(a.shape[i])

    y = np.empty(shape, np.float32)
    cdef np.flatiter ity = np.PyArray_IterNew(y)

    cdef np.float32_t asum, ai

    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<np.float32_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i*stride))[0]
            if ai == ai:
                asum += ai
        (<double*>((<char*>np.PyArray_ITER_DATA(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)

    return y


cdef inline np.ndarray nansum_int64_axisint(np.ndarray a, int axis):

    cdef int ndim = a.ndim
    if ndim <= 1:
        raise ValueError("`a.ndim` must be greater than 1")
    if axis < 0:
        axis += ndim
    if (axis >= ndim) or (axis < 0):
        raise ValueError("axis(=%d) out of bounds" % axis)

    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t length = a.shape[axis], i
    cdef int stride = a.strides[axis]

    # temp python hack
    cdef list shape = []
    for i in range(ndim):
        if i != axis:
            shape.append(a.shape[i])

    y = np.empty(shape, np.int64)
    cdef np.flatiter ity = np.PyArray_IterNew(y)

    cdef np.int64_t asum, ai

    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<np.int64_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i*stride))[0]
            asum += ai
        (<double*>((<char*>np.PyArray_ITER_DATA(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)

    return y


cdef inline np.ndarray nansum_int32_axisint(np.ndarray a, int axis):

    cdef int ndim = a.ndim
    if ndim <= 1:
        raise ValueError("`a.ndim` must be greater than 1")
    if axis < 0:
        axis += ndim
    if (axis >= ndim) or (axis < 0):
        raise ValueError("axis(=%d) out of bounds" % axis)

    cdef np.flatiter ita
    ita = np.PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t length = a.shape[axis], i
    cdef int stride = a.strides[axis]

    # temp python hack
    cdef list shape = []
    for i in range(ndim):
        if i != axis:
            shape.append(a.shape[i])

    y = np.empty(shape, np.int32)
    cdef np.flatiter ity = np.PyArray_IterNew(y)

    cdef np.int32_t asum, ai

    while np.PyArray_ITER_NOTDONE(ita):
        asum = 0
        for i in range(length):
            ai = (<np.int32_t*>((<char*>np.PyArray_ITER_DATA(ita)) + i*stride))[0]
            asum += ai
        (<double*>((<char*>np.PyArray_ITER_DATA(ity))))[0] = asum
        np.PyArray_ITER_NEXT(ita)
        np.PyArray_ITER_NEXT(ity)

    return y
