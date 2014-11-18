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

from numpy cimport PyArray_EMPTY
from numpy cimport ndarray
from numpy cimport import_array
import_array()


# nansum --------------------------------------------------------------------

def nansum(arr, axis=None):
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


cdef DTYPE0_t nansum_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
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


cdef void nansum_one_DTYPE0(np.flatiter ita, np.flatiter ity,
                            Py_ssize_t stride, Py_ssize_t length,
                            int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
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


cdef nansum_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return out
    else:
        return 0.0


# reducer -------------------------------------------------------------------

# pointer to functions that reduce along all axes
ctypedef float64_t (*fall_float64_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)
ctypedef float32_t (*fall_float32_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)
ctypedef int64_t (*fall_int64_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)
ctypedef int32_t (*fall_int32_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)

# pointer to functions that reduce along a single axis
ctypedef void (*fone_t)(np.flatiter, np.flatiter, Py_ssize_t, Py_ssize_t, int)

# pointer to functions that handle 0d arrays
ctypedef object (*f0d_t)(ndarray, int)


cdef reducer(arr, axis,
             fall_float64_t fall_float64,
             fall_float32_t fall_float32,
             fall_int64_t fall_int64,
             fall_int32_t fall_int32,
             fone_t fone_float64,
             fone_t fone_float32,
             fone_t fone_int64,
             fone_t fone_int32,
             f0d_t f0d,
             int int_input=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i, j
    cdef int dtype = PyArray_TYPE(a)
    cdef int ndim = PyArray_NDIM(a)

    # output array, if needed
    cdef ndarray y
    cdef np.flatiter ity
    cdef np.npy_intp *adim
    cdef np.npy_intp *ydim = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # TODO max ndim=10

    # defend against 0d beings
    if ndim == 0:
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
        if dtype == NPY_FLOAT64:
            return fall_float64(ita, stride, length, int_input)
        elif dtype == NPY_FLOAT32:
            return fall_float32(ita, stride, length, int_input)
        elif dtype == NPY_INT64:
            return fall_int64(ita, stride, length, int_input)
        elif dtype == NPY_INT32:
            return fall_int32(ita, stride, length, int_input)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
    else:
        # reduce over a single axis; ndim > 1
        adim = np.PyArray_DIMS(a)
        j = 0
        for i in range(ndim):
            if i != axis_reduce:
                ydim[j] = adim[i]
                j += 1
        if dtype == NPY_FLOAT64:
            y = PyArray_EMPTY(ndim - 1, ydim, NPY_FLOAT64, 0)
            ity = PyArray_IterNew(y)
            fone_float64(ita, ity, stride, length, int_input)
        elif dtype == NPY_FLOAT32:
            y = PyArray_EMPTY(ndim - 1, ydim, NPY_FLOAT32, 0)
            ity = PyArray_IterNew(y)
            fone_float32(ita, ity, stride, length, int_input)
        elif dtype == NPY_INT64:
            y = PyArray_EMPTY(ndim - 1, ydim, NPY_INT64, 0)
            ity = PyArray_IterNew(y)
            fone_int64(ita, ity, stride, length, int_input)
        elif dtype == NPY_INT32:
            y = PyArray_EMPTY(ndim - 1, ydim, NPY_INT32, 0)
            ity = PyArray_IterNew(y)
            fone_int32(ita, ity, stride, length, int_input)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
        return y
