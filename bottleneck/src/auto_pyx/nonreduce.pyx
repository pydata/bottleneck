#cython: embedsignature=True

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t, intp_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INTP as NPY_intp

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM
from numpy cimport NPY_CORDER
from numpy cimport PyArray_Copy
from numpy cimport PyArray_EMPTY
from numpy cimport PyArray_Ravel

from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.reduce as slow

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


# replace -------------------------------------------------------------------

def replace(arr, double old, double new):
    try:
        nonreducer(arr,
                   replace_float64,
                   replace_float32,
                   replace_int64,
                   replace_int32,
                   old,
                   new,
                   1)
    except TypeError:
        slow.replace(arr, old, new)


cdef ndarray replace_float64(ndarray a, np.flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, np.npy_intp* y_dims,
                            double old, double new):
    cdef Py_ssize_t i
    cdef float64_t ai
    if old == old:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<float64_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == old:
                    (<float64_t*>((<char*>pid(ita)) + i * stride))[0] = new
            PyArray_ITER_NEXT(ita)
    else:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<float64_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai != ai:
                    (<float64_t*>((<char*>pid(ita)) + i * stride))[0] = new
            PyArray_ITER_NEXT(ita)
    return a

cdef ndarray replace_float32(ndarray a, np.flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, np.npy_intp* y_dims,
                            double old, double new):
    cdef Py_ssize_t i
    cdef float32_t ai
    if old == old:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<float32_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == old:
                    (<float32_t*>((<char*>pid(ita)) + i * stride))[0] = new
            PyArray_ITER_NEXT(ita)
    else:
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<float32_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai != ai:
                    (<float32_t*>((<char*>pid(ita)) + i * stride))[0] = new
            PyArray_ITER_NEXT(ita)
    return a

cdef ndarray replace_int64(ndarray a, np.flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, np.npy_intp* y_dims,
                            double old, double new):
    cdef Py_ssize_t i
    cdef int64_t ai, oldint, newint
    if old == old:
        oldint = <int64_t>old
        newint = <int64_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<int64_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == oldint:
                    (<int64_t*>((<char*>pid(ita)) + i * stride))[0] = newint
            PyArray_ITER_NEXT(ita)
    return a

cdef ndarray replace_int32(ndarray a, np.flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, np.npy_intp* y_dims,
                            double old, double new):
    cdef Py_ssize_t i
    cdef int32_t ai, oldint, newint
    if old == old:
        oldint = <int32_t>old
        newint = <int32_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<int32_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == oldint:
                    (<int32_t*>((<char*>pid(ita)) + i * stride))[0] = newint
            PyArray_ITER_NEXT(ita)
    return a

# nonreduce_axis ------------------------------------------------------------

ctypedef ndarray (*nr_t)(ndarray, np.flatiter,
                         Py_ssize_t, Py_ssize_t,
                         int, np.npy_intp*,
                         double, double)


cdef ndarray nonreducer(arr,
                        nr_t nr_float64,
                        nr_t nr_float32,
                        nr_t nr_int64,
                        nr_t nr_int32,
                        double double_input_1,
                        double double_input_2,
                        int inplace=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        if inplace == 1:
            # works in place so input must be an array, not (e.g.) a list
            raise TypeError("`arr` must be a numpy array.")
        else:
            a = np.array(arr, copy=False)

    # input array
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # input iterator
    cdef int axis = -1
    cdef np.flatiter ita = PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = a.strides[axis]
    cdef Py_ssize_t length = a.shape[axis]

    # output array
    cdef ndarray y
    cdef np.npy_intp *y_dims = np.PyArray_DIMS(a)

    # calc
    if dtype == NPY_float64:
        y = nr_float64(a, ita, stride, length, a_ndim, y_dims, double_input_1, double_input_2)
    elif dtype == NPY_float32:
        y = nr_float32(a, ita, stride, length, a_ndim, y_dims, double_input_1, double_input_2)
    elif dtype == NPY_int64:
        y = nr_int64(a, ita, stride, length, a_ndim, y_dims, double_input_1, double_input_2)
    elif dtype == NPY_int32:
        y = nr_int32(a, ita, stride, length, a_ndim, y_dims, double_input_1, double_input_2)
    else:
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    return y