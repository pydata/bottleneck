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
from numpy cimport NPY_CORDER
from numpy cimport PyArray_Copy
from numpy cimport PyArray_EMPTY
from numpy cimport PyArray_Ravel

from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.nonreduce_axis as slow

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


# partsort ------------------------------------------------------------------

def partsort(arr, int n, axis=-1):
    try:
        return nonreducer_axis(arr, axis,
                               partsort_float64,
                               partsort_float32,
                               partsort_int64,
                               partsort_int32,
                               n)
    except TypeError:
        return slow.partsort(arr, n, axis)


cdef ndarray partsort_DTYPE0(ndarray a, int axis,
                             int a_ndim, np.npy_intp* y_dims, int n):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef np.npy_intp i, j = 0, l, r, k = n-1
    cdef DTYPE0_t x, tmpi, tmpj
    cdef ndarray y = PyArray_Copy(a)
    cdef Py_ssize_t stride = y.strides[axis]
    cdef Py_ssize_t length = y.shape[axis]
    if length == 0:
        return y
    if (n < 1) or (n > length):
        raise ValueError("`n` (=%d) must be between 1 and %d, inclusive." %
                         (n, length))
    cdef np.flatiter ity = PyArray_IterAllButAxis(y, &axis)
    while PyArray_ITER_NOTDONE(ity):
        l = 0
        r = length - 1
        while l < r:
            x = (<DTYPE0_t*>((<char*>pid(ity)) + k*stride))[0]
            i = l
            j = r
            while 1:
                while (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] < x: i += 1
                while x < (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]: j -= 1
                if i <= j:
                    tmpi = (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0]
                    tmpj = (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ity)) + i*stride))[0] = tmpj
                    (<DTYPE0_t*>((<char*>pid(ity)) + j*stride))[0] = tmpi
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        PyArray_ITER_NEXT(ity)
    return y


# nonreduce_axis ------------------------------------------------------------

ctypedef ndarray (*nra_t)(ndarray, int, int, np.npy_intp*, int)


cdef ndarray nonreducer_axis(arr, axis,
                             nra_t nra_float64,
                             nra_t nra_float32,
                             nra_t nra_int64,
                             nra_t nra_int32,
                             int int_input=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # output array
    cdef ndarray y
    cdef np.npy_intp *y_dims = np.PyArray_DIMS(a)

    # axis
    cdef int axis_int
    if axis is None:
        a = PyArray_Ravel(a, NPY_CORDER)
        axis_int = 0
    else:
        axis_int = <int>axis
        if axis_int < 0:
            axis_int += a_ndim
            if axis_int < 0:
                raise ValueError("axis(=%d) out of bounds" % axis)
        elif axis_int >= a_ndim:
            raise ValueError("axis(=%d) out of bounds" % axis)

    # calc
    if dtype == NPY_float64:
        y = nra_float64(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_float32:
        y = nra_float32(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_int64:
        y = nra_int64(a, axis_int, a_ndim, y_dims, int_input)
    elif dtype == NPY_int32:
        y = nra_int32(a, axis_int, a_ndim, y_dims, int_input)
    else:
        raise TypeError("Unsupported dtype (%s)." % a.dtype)

    return y
