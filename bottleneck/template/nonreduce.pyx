#cython: embedsignature=True

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
from numpy cimport PyArray_ISBYTESWAPPED
from numpy cimport PyArray_STRIDE
from numpy cimport PyArray_Check
from numpy cimport PyArray_FROM_O

from numpy cimport ndarray
from numpy cimport import_array
import_array()

import bottleneck.slow.nonreduce as slow


# replace -------------------------------------------------------------------

def replace(arr, double old, double new):
    """
    Replace (inplace) given scalar values of an array with new values.

    The equivalent numpy function:

        arr[arr==old] = new

    Or in the case where old=np.nan:

        arr[np.isnan(old)] = new

    Parameters
    ----------
    arr : numpy.ndarray
        The input array, which is also the output array since this functions
        works inplace.
    old : scalar
        All elements in `arr` with this value will be replaced by `new`.
    new : scalar
        All elements in `arr` with a value of `old` will be replaced by `new`.

    Returns
    -------
    None, the operation is inplace.

    Examples
    --------
    Replace zero with 3 (note that the input array is modified):

    >>> a = np.array([1, 2, 0])
    >>> bn.replace(a, 0, 3)
    >>> a
    array([1, 2, 3])

    Replace np.nan with 0:

    >>> a = np.array([1, 2, np.nan])
    >>> bn.replace(a, np.nan, 0)
    >>> a
    array([ 1.,  2.,  0.])

    """
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


cdef ndarray replace_DTYPE0(ndarray a, flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, npy_intp* y_dims,
                            double old, double new):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    with nogil:
        if old == old:
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == old:
                        (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0] = new
                PyArray_ITER_NEXT(ita)
        else:
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai != ai:
                        (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0] = new
                PyArray_ITER_NEXT(ita)
    return a


cdef ndarray replace_DTYPE0(ndarray a, flatiter ita,
                            Py_ssize_t stride, Py_ssize_t length,
                            int a_ndim, npy_intp* y_dims,
                            double old, double new):
    # bn.dtypes = [['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, oldint, newint
    if old == old:
        oldint = <DTYPE0_t>old
        newint = <DTYPE0_t>new
        if oldint != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if newint != new:
            raise ValueError("Cannot safely cast `new` to int.")
        with nogil:
            while PyArray_ITER_NOTDONE(ita):
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == oldint:
                        (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0] = newint
                PyArray_ITER_NEXT(ita)
    return a


# nonreduce -----------------------------------------------------------------

ctypedef ndarray (*nr_t)(ndarray, flatiter,
                         Py_ssize_t, Py_ssize_t,
                         int, npy_intp*,
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
    if PyArray_Check(arr):
        a = arr
    else:
        if inplace == 1:
            # works in place so input must be an array, not (e.g.) a list
            raise TypeError
        else:
            a = PyArray_FROM_O(arr)

    # check for byte swapped input array
    if PyArray_ISBYTESWAPPED(a):
        raise TypeError

    # input array
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # input iterator
    cdef int axis = -1
    cdef flatiter ita = PyArray_IterAllButAxis(a, &axis)
    cdef Py_ssize_t stride = PyArray_STRIDE(a, axis)

    # output array
    cdef ndarray y
    cdef npy_intp *y_dims = np.PyArray_DIMS(a)
    cdef Py_ssize_t length = y_dims[axis]

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
        raise TypeError

    return y
