#cython: embedsignature=True

# Select smallest k elements code used for inner loop of median method:
# http://projects.scipy.org/numpy/attachment/ticket/1213/quickselect.pyx
# (C) 2009 Sturla Molden
# SciPy license
#
# From the original C function (code in public domain) in:
#   Fast median search: an ANSI C implementation
#   Nicolas Devillard - ndevilla AT free DOT fr
#   July 1998
# which, in turn, took the algorithm from
#   Wirth, Niklaus
#   Algorithms + data structures = programs, p. 366
#   Englewood Cliffs: Prentice-Hall, 1976
#
# Adapted and expanded for Bottleneck:
# (C) 2010, 2015 Keith Goodman

import numpy as np
cimport numpy as np
import cython

from numpy cimport float64_t, float32_t, int64_t, int32_t, intp_t
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INTP as NPY_intp
from numpy cimport NPY_BOOL

from numpy cimport PyArray_ITER_DATA as pid
from numpy cimport PyArray_ITER_NOTDONE
from numpy cimport PyArray_ITER_NEXT
from numpy cimport PyArray_ITER_RESET
from numpy cimport PyArray_IterAllButAxis
from numpy cimport PyArray_IterNew

from numpy cimport PyArray_TYPE
from numpy cimport PyArray_NDIM
from numpy cimport NPY_CORDER

from numpy cimport PyArray_FillWithScalar
from numpy cimport PyArray_Copy
from numpy cimport PyArray_Ravel
from numpy cimport PyArray_EMPTY
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


# nansum --------------------------------------------------------------------

def nansum(arr, axis=None):
    try:
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
    except TypeError:
        return slow.nansum(arr, axis)


cdef object nansum_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
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


cdef ndarray nansum_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
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
    return y


cdef nansum_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return out
    else:
        return 0.0


# nanmean --------------------------------------------------------------------

def nanmean(arr, axis=None):
    try:
        return reducer(arr, axis,
                       nanmean_all_float64,
                       nanmean_all_float32,
                       nanmean_all_int64,
                       nanmean_all_int32,
                       nanmean_one_float64,
                       nanmean_one_float32,
                       nanmean_one_int64,
                       nanmean_one_int32,
                       nanmean_0d)
    except TypeError:
        return slow.nanmean(arr, axis)


@cython.cdivision(True)
cdef object nanmean_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                               Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
                count += 1
        PyArray_ITER_NEXT(ita)
    if count > 0:
        return asum / count
    else:
        return NAN


@cython.cdivision(True)
cdef object nanmean_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                               Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0
    cdef DTYPE0_t ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai
        size += length
        PyArray_ITER_NEXT(ita)
    if size == 0:
        return NAN
    else:
        return asum / size


@cython.cdivision(True)
cdef ndarray nanmean_one_DTYPE0(np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count
    cdef DTYPE0_t asum = 0, ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > 0:
                asum = asum / count
            else:
                asum = NAN
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanmean_one_DTYPE0(np.flatiter ita,
                                Py_ssize_t stride, Py_ssize_t length,
                                int a_ndim, np.npy_intp* y_dims,
                                int int_input):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum / length
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


cdef nanmean_0d(ndarray a, int int_input):
    return <double>a[()]


# nanstd --------------------------------------------------------------------

def nanstd(arr, axis=None, int ddof=0):
    try:
        return reducer(arr, axis,
                       nanstd_all_float64,
                       nanstd_all_float32,
                       nanstd_all_int64,
                       nanstd_all_int32,
                       nanstd_one_float64,
                       nanstd_one_float32,
                       nanstd_one_int64,
                       nanstd_one_int32,
                       nanstd_0d,
                       ddof)
    except TypeError:
        return slow.nanstd(arr, axis, ddof=ddof)


@cython.cdivision(True)
cdef object nanstd_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, amean, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
                count += 1
        PyArray_ITER_NEXT(ita)
    if count > ddof:
        amean = asum / count
        asum = 0
        PyArray_ITER_RESET(ita)
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    ai -= amean
                    asum += ai * ai
            PyArray_ITER_NEXT(ita)
        return sqrt(asum / (count - ddof))
    else:
        return NAN


@cython.cdivision(True)
cdef object nanstd_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai
        size += length
        PyArray_ITER_NEXT(ita)
    if size > ddof:
        amean = asum / size
        asum = 0
        PyArray_ITER_RESET(ita)
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                aj = ai - amean
                asum += aj * aj
            PyArray_ITER_NEXT(ita)
        return sqrt(asum / (size - ddof))
    else:
        return NAN


@cython.cdivision(True)
cdef ndarray nanstd_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai, amean
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > ddof:
                amean = asum / count
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        ai -= amean
                        asum += ai * ai
                asum = sqrt(asum / (count - ddof))
            else:
                asum = NAN
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanstd_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                asum += ai
            if length > ddof:
                amean = asum / length
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    aj = ai - amean
                    asum += aj * aj
                asum = sqrt(asum / (length - ddof))
            else:
                asum = NAN
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


cdef nanstd_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        if out == np.inf or out == -np.inf:
            return NAN
        else:
            return 0.0
    else:
        return NAN


# nanvar --------------------------------------------------------------------

def nanvar(arr, axis=None, int ddof=0):
    try:
        return reducer(arr, axis,
                       nanvar_all_float64,
                       nanvar_all_float32,
                       nanvar_all_int64,
                       nanvar_all_int32,
                       nanvar_one_float64,
                       nanvar_one_float32,
                       nanvar_one_int64,
                       nanvar_one_int32,
                       nanvar_0d,
                       ddof)
    except TypeError:
        return slow.nanvar(arr, axis, ddof=ddof)


@cython.cdivision(True)
cdef object nanvar_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, amean, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai == ai:
                asum += ai
                count += 1
        PyArray_ITER_NEXT(ita)
    if count > ddof:
        amean = asum / count
        asum = 0
        PyArray_ITER_RESET(ita)
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    ai -= amean
                    asum += ai * ai
            PyArray_ITER_NEXT(ita)
        return asum / (count - ddof)
    else:
        return NAN


@cython.cdivision(True)
cdef object nanvar_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i, size = 0
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai
        size += length
        PyArray_ITER_NEXT(ita)
    if size > ddof:
        amean = asum / size
        asum = 0
        PyArray_ITER_RESET(ita)
        while PyArray_ITER_NOTDONE(ita):
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                aj = ai - amean
                asum += aj * aj
            PyArray_ITER_NEXT(ita)
        return asum / (size - ddof)
    else:
        return NAN


@cython.cdivision(True)
cdef ndarray nanvar_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i, count = 0
    cdef DTYPE0_t asum = 0, ai, amean
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            count = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                if ai == ai:
                    asum += ai
                    count += 1
            if count > ddof:
                amean = asum / count
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    if ai == ai:
                        ai -= amean
                        asum += ai * ai
                asum = asum / (count - ddof)
            else:
                asum = NAN
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


@cython.cdivision(True)
cdef ndarray nanvar_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int ddof):
    # bn.dtypes = [['int64', 'float64'], ['int32', 'float64']]
    cdef Py_ssize_t i
    cdef DTYPE1_t asum = 0, amean, aj
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                asum += ai
            if length > ddof:
                amean = asum / length
                asum = 0
                for i in range(length):
                    ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
                    aj = ai - amean
                    asum += aj * aj
                asum = asum / (length - ddof)
            else:
                asum = NAN
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


cdef nanvar_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        if out == np.inf or out == -np.inf:
            return NAN
        else:
            return 0.0
    else:
        return NAN


# nanmin --------------------------------------------------------------------

def nanmin(arr, axis=None):
    try:
        return reducer(arr, axis,
                       nanmin_all_float64,
                       nanmin_all_float32,
                       nanmin_all_int64,
                       nanmin_all_int32,
                       nanmin_one_float64,
                       nanmin_one_float32,
                       nanmin_one_int64,
                       nanmin_one_int32,
                       nanmin_0d)
    except TypeError:
        return slow.nanmin(arr, axis)


cdef object nanmin_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1, is_size_0 = 1
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin = MAXDTYPE0
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai <= amin:
                amin = ai
                allnan = 0
        is_size_0 = 0
        PyArray_ITER_NEXT(ita)
    if is_size_0 == 1:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    if allnan == 0:
        return amin
    else:
        return NAN


cdef object nanmin_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef int is_size_0 = 1
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin = MAXDTYPE0
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai <= amin:
                amin = ai
        is_size_0 = 0
        PyArray_ITER_NEXT(ita)
    if is_size_0 == 1:
        m = "numpy.nanmin raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    return amin


cdef ndarray nanmin_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MAXDTYPE0
        allnan = 1
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai <= amin:
                amin = ai
                allnan = 0
        if allnan != 0:
            amin = NAN
        (<DTYPE0_t*>((<char*>pid(ity))))[0] = amin
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray nanmin_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanmin raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MAXDTYPE0
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai <= amin:
                amin = ai
        (<DTYPE0_t*>((<char*>pid(ity))))[0] = amin
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef nanmin_0d(ndarray a, int int_input):
    raise a[()]


# nanmax --------------------------------------------------------------------

def nanmax(arr, axis=None):
    try:
        return reducer(arr, axis,
                       nanmax_all_float64,
                       nanmax_all_float32,
                       nanmax_all_int64,
                       nanmax_all_int32,
                       nanmax_one_float64,
                       nanmax_one_float32,
                       nanmax_one_int64,
                       nanmax_one_int32,
                       nanmax_0d)
    except TypeError:
        return slow.nanmax(arr, axis)


cdef object nanmax_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1, is_size_0 = 1
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin = MINDTYPE0
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai >= amin:
                amin = ai
                allnan = 0
        is_size_0 = 0
        PyArray_ITER_NEXT(ita)
    if is_size_0 == 1:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    if allnan == 0:
        return amin
    else:
        return NAN


cdef object nanmax_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef int is_size_0 = 1
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin = MINDTYPE0
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai >= amin:
                amin = ai
        is_size_0 = 0
        PyArray_ITER_NEXT(ita)
    if is_size_0 == 1:
        m = "numpy.nanmax raises on a.size==0 and axis=None; Bottleneck too."
        raise ValueError(m)
    return amin


cdef ndarray nanmax_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MINDTYPE0
        allnan = 1
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai >= amin:
                amin = ai
                allnan = 0
        if allnan != 0:
            amin = NAN
        (<DTYPE0_t*>((<char*>pid(ity))))[0] = amin
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray nanmax_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai, amin
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanmax raises on a.shape[axis]==0; so Bottleneck does."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MINDTYPE0
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai >= amin:
                amin = ai
        (<DTYPE0_t*>((<char*>pid(ity))))[0] = amin
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef nanmax_0d(ndarray a, int int_input):
    raise a[()]


# ss ------------------------------------------------------------------------

def ss(arr, axis=None):
    try:
        return reducer(arr, axis,
                       ss_all_float64,
                       ss_all_float32,
                       ss_all_int64,
                       ss_all_int32,
                       ss_one_float64,
                       ss_one_float32,
                       ss_one_int64,
                       ss_one_int32,
                       ss_0d)
    except TypeError:
        return slow.ss(arr, axis)


cdef object ss_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                          Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            asum += ai * ai
        PyArray_ITER_NEXT(ita)
    return asum


cdef ndarray ss_one_DTYPE0(np.flatiter ita,
                           Py_ssize_t stride, Py_ssize_t length,
                           int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t asum = 0, ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ity)
    else:
        while PyArray_ITER_NOTDONE(ita):
            asum = 0
            for i in range(length):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                asum += ai * ai
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = asum
            PyArray_ITER_NEXT(ita)
            PyArray_ITER_NEXT(ity)
    return y


cdef ss_0d(ndarray a, int int_input):
    out = a[()]
    return out * out


# nanmedian -----------------------------------------------------------------

def nanmedian(arr, axis=None):
    cdef int ravel = 0, copy = 1, int_input = 0
    try:
        if axis is None:
            ravel = 1
        return reducer(arr, axis,
                       nanmedian_all_float64,
                       nanmedian_all_float32,
                       median_all_int64,
                       median_all_int32,
                       nanmedian_one_float64,
                       nanmedian_one_float32,
                       median_one_int64,
                       median_one_int32,
                       nanmedian_0d,
                       int_input,
                       ravel,
                       copy)
    except TypeError:
        return slow.nanmedian(arr, axis)


cdef object nanmedian_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                 Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n
    cdef DTYPE0_t x, tmp, amax, ai, bi
    if length == 0:
        return NAN
    j = length - 1
    flag = 1
    for i in range(length):
        bi = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
        if bi != bi:
            while (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] != (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]:
                if j <= 0:
                    break
                j -= 1
            if i >= j:
                flag = 0
                break
            tmp = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
            (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
            (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = bi
    n = i + flag
    k = n >> 1
    l = 0
    r = n - 1
    while l < r:
        x = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
        i = l
        j = r
        while 1:
            while (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] < x: i += 1
            while x < (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]: j -= 1
            if i <= j:
                tmp = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = tmp
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j
    bi = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
    if n % 2 == 0:
        amax = MINDTYPE0
        allnan = 1
        for i in range(k):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai >= amax:
                amax = ai
                allnan = 0
        if allnan == 0:
            return 0.5 * (bi + amax)
        else:
            return bi
    else:
        return bi


cdef ndarray nanmedian_one_DTYPE0(np.flatiter ita,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  int a_ndim, np.npy_intp* y_dims,
                                  int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1, flag = 0
    cdef np.npy_intp i = 0, j = 0, l, r, k, n
    cdef DTYPE0_t x, tmp, amax, ai, bi
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE0, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
        return y
    while PyArray_ITER_NOTDONE(ita):
        j = length - 1
        flag = 1
        for i in range(length):
            bi = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if bi != bi:
                while (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] != (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]:
                    if j <= 0:
                        break
                    j -= 1
                if i >= j:
                    flag = 0
                    break
                tmp = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = bi
        n = i + flag
        k = n >> 1
        l = 0
        r = n - 1
        while l < r:
            x = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
            i = l
            j = r
            while 1:
                while (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] < x: i += 1
                while x < (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]: j -= 1
                if i <= j:
                    tmp = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        bi = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
        if n % 2 == 0:
            amax = MINDTYPE0
            allnan = 1
            for i in range(k):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai >= amax:
                    amax = ai
                    allnan = 0
            if allnan == 0:
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = 0.5 * (bi + amax)
            else:
                (<DTYPE0_t*>((<char*>pid(ity))))[0] = bi
        else:
            (<DTYPE0_t*>((<char*>pid(ity))))[0] = bi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef nanmedian_0d(ndarray a, int int_input):
    return a[()]


# median -----------------------------------------------------------------

def median(arr, axis=None):
    cdef int ravel = 0, copy = 1, int_input = 0
    try:
        if axis is None:
            ravel = 1
        return reducer(arr, axis,
                       median_all_float64,
                       median_all_float32,
                       median_all_int64,
                       median_all_int32,
                       median_one_float64,
                       median_one_float32,
                       median_one_int64,
                       median_one_int32,
                       median_0d,
                       int_input,
                       ravel,
                       copy)
    except TypeError:
        return slow.median(arr, axis)


cdef object median_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32'], ['int64'], ['int32']]
    cdef np.npy_intp i = 0, j = 0, l, r, k
    cdef DTYPE0_t x, tmp, amax, ai, bi
    if length == 0:
        return NAN
    k = length >> 1
    l = 0
    r = length - 1
    while l < r:
        x = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
        i = l
        j = r
        while 1:
            while (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] < x: i += 1
            while x < (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]: j -= 1
            if i <= j:
                tmp = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = tmp
                i += 1
                j -= 1
            if i > j: break
        if j < k: l = i
        if k < i: r = j
    bi = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
    if length % 2 == 0:
        amax = MINDTYPE0
        for i in range(k):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai >= amax:
                amax = ai
        return 0.5 * (bi + amax)
    else:
        return 1.0 * bi


cdef ndarray median_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims,
                               int int_input):
    # bn.dtypes = [['float64', 'float64'], ['float32', 'float32'], ['int64', 'float64'], ['int32', 'float64']]
    cdef np.npy_intp i = 0, j = 0, l, r, k
    cdef DTYPE0_t x, tmp, amax, ai, bi
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_DTYPE1, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        while PyArray_ITER_NOTDONE(ity):
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = NAN
            PyArray_ITER_NEXT(ity)
        return y
    while PyArray_ITER_NOTDONE(ita):
        k = length >> 1
        l = 0
        r = length - 1
        while l < r:
            x = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
            i = l
            j = r
            while 1:
                while (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] < x: i += 1
                while x < (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]: j -= 1
                if i <= j:
                    tmp = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0] = (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0]
                    (<DTYPE0_t*>((<char*>pid(ita)) + j*stride))[0] = tmp
                    i += 1
                    j -= 1
                if i > j: break
            if j < k: l = i
            if k < i: r = j
        bi = (<DTYPE0_t*>((<char*>pid(ita)) + k*stride))[0]
        if length % 2 == 0:
            amax = MINDTYPE0
            for i in range(k):
                ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
                if ai >= amax:
                    amax = ai
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = 0.5 * (bi + amax)
        else:
            (<DTYPE1_t*>((<char*>pid(ity))))[0] = bi
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef median_0d(ndarray a, int int_input):
    return a[()]


# nanargmin -----------------------------------------------------------------

def nanargmin(arr, axis=None):
    cdef int ravel = 0, copy = 0, int_input = 0
    try:
        if axis is None:
            ravel = 1
        return reducer(arr, axis,
                       nanargmin_all_float64,
                       nanargmin_all_float32,
                       nanargmin_all_int64,
                       nanargmin_all_int32,
                       nanargmin_one_float64,
                       nanargmin_one_float32,
                       nanargmin_one_int64,
                       nanargmin_one_int32,
                       nanargmin_0d,
                       int_input,
                       ravel,
                       copy)
    except TypeError:
        return slow.nanargmin(arr, axis)


cdef object nanargmin_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                 Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1
    cdef DTYPE0_t amin, ai
    cdef Py_ssize_t i, idx = 0
    if length == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amin = MAXDTYPE0
    allnan = 1
    for i in range(length - 1, -1, -1):
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai <= amin:
            amin = ai
            allnan = 0
            idx = i
    if allnan == 0:
        return idx
    else:
        raise ValueError("All-NaN slice encountered")


cdef object nanargmin_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                 Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef DTYPE0_t amin, ai
    cdef Py_ssize_t i, idx = 0
    if length == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amin = MAXDTYPE0
    allnan = 1
    for i in range(length - 1, -1, -1):
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai <= amin:
            amin = ai
            idx = i
    return idx


cdef ndarray nanargmin_one_DTYPE0(np.flatiter ita,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  int a_ndim, np.npy_intp* y_dims,
                                  int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1
    cdef DTYPE0_t amin, ai
    cdef Py_ssize_t i, idx = 0
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_intp, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MAXDTYPE0
        allnan = 1
        for i in range(length - 1, -1, -1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai <= amin:
                amin = ai
                allnan = 0
                idx = i
        if allnan == 0:
            (<intp_t*>((<char*>pid(ity))))[0] = idx
        else:
            raise ValueError("All-NaN slice encountered")
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray nanargmin_one_DTYPE0(np.flatiter ita,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  int a_ndim, np.npy_intp* y_dims,
                                  int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef DTYPE0_t amin, ai
    cdef Py_ssize_t i, idx = 0
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_intp, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanargmin raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amin = MAXDTYPE0
        for i in range(length - 1, -1, -1):
            ai = (<intp_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai <= amin:
                amin = ai
                idx = i
        (<intp_t*>((<char*>pid(ity))))[0] = idx
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef nanargmin_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return 0
    else:
        raise ValueError("All-NaN slice encountered")


# nanargmax -----------------------------------------------------------------

def nanargmax(arr, axis=None):
    cdef int ravel = 0, copy = 0, int_input = 0
    try:
        if axis is None:
            ravel = 1
        return reducer(arr, axis,
                       nanargmax_all_float64,
                       nanargmax_all_float32,
                       nanargmax_all_int64,
                       nanargmax_all_int32,
                       nanargmax_one_float64,
                       nanargmax_one_float32,
                       nanargmax_one_int64,
                       nanargmax_one_int32,
                       nanargmax_0d,
                       int_input,
                       ravel,
                       copy)
    except TypeError:
        return slow.nanargmax(arr, axis)


cdef object nanargmax_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                 Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1
    cdef DTYPE0_t amax, ai
    cdef Py_ssize_t i, idx = 0
    if length == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amax = MINDTYPE0
    allnan = 1
    for i in range(length - 1, -1, -1):
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai >= amax:
            amax = ai
            allnan = 0
            idx = i
    if allnan == 0:
        return idx
    else:
        raise ValueError("All-NaN slice encountered")


cdef object nanargmax_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                                 Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef DTYPE0_t amax, ai
    cdef Py_ssize_t i, idx = 0
    if length == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    amax = MINDTYPE0
    allnan = 1
    for i in range(length - 1, -1, -1):
        ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
        if ai >= amax:
            amax = ai
            idx = i
    return idx


cdef ndarray nanargmax_one_DTYPE0(np.flatiter ita,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  int a_ndim, np.npy_intp* y_dims,
                                  int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int allnan = 1
    cdef DTYPE0_t amax, ai
    cdef Py_ssize_t i, idx = 0
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_intp, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amax = MINDTYPE0
        allnan = 1
        for i in range(length - 1, -1, -1):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai >= amax:
                amax = ai
                allnan = 0
                idx = i
        if allnan == 0:
            (<intp_t*>((<char*>pid(ity))))[0] = idx
        else:
            raise ValueError("All-NaN slice encountered")
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray nanargmax_one_DTYPE0(np.flatiter ita,
                                  Py_ssize_t stride, Py_ssize_t length,
                                  int a_ndim, np.npy_intp* y_dims,
                                  int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef DTYPE0_t amax, ai
    cdef Py_ssize_t i, idx = 0
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_intp, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        msg = "numpy.nanargmax raises on a.shape[axis]==0; Bottleneck too."
        raise ValueError(msg)
    while PyArray_ITER_NOTDONE(ita):
        amax = MINDTYPE0
        for i in range(length - 1, -1, -1):
            ai = (<intp_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai >= amax:
                amax = ai
                idx = i
        (<intp_t*>((<char*>pid(ity))))[0] = idx
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef nanargmax_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return 0
    else:
        raise ValueError("All-NaN slice encountered")


# anynan --------------------------------------------------------------------

def anynan(arr, axis=None):
    try:
        return reducer(arr, axis,
                       anynan_all_float64,
                       anynan_all_float32,
                       anynan_all_int64,
                       anynan_all_int32,
                       anynan_one_float64,
                       anynan_one_float32,
                       anynan_one_int64,
                       anynan_one_int32,
                       anynan_0d)
    except TypeError:
        return slow.anynan(arr, axis)


cdef object anynan_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    while PyArray_ITER_NOTDONE(ita):
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i * stride))[0]
            if ai != ai:
                return True
        PyArray_ITER_NEXT(ita)
    return False


cdef object anynan_all_DTYPE0(np.flatiter ita, Py_ssize_t stride,
                              Py_ssize_t length, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    return False


cdef ndarray anynan_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['float64'], ['float32']]
    cdef int f
    cdef Py_ssize_t i
    cdef DTYPE0_t ai
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_BOOL, 0)
    cdef np.flatiter ity = PyArray_IterNew(y)
    if length == 0:
        PyArray_FillWithScalar(y, 0)
        return y
    while PyArray_ITER_NOTDONE(ita):
        f = 1
        for i in range(length):
            ai = (<DTYPE0_t*>((<char*>pid(ita)) + i*stride))[0]
            if ai != ai:
                (<np.uint8_t*>((<char*>pid(ity))))[0] = 1
                f = 0
                break
        if f == 1:
            (<np.uint8_t*>((<char*>pid(ity))))[0] = 0
        PyArray_ITER_NEXT(ita)
        PyArray_ITER_NEXT(ity)
    return y


cdef ndarray anynan_one_DTYPE0(np.flatiter ita,
                               Py_ssize_t stride, Py_ssize_t length,
                               int a_ndim, np.npy_intp* y_dims, int int_input):
    # bn.dtypes = [['int64'], ['int32']]
    cdef ndarray y = PyArray_EMPTY(a_ndim - 1, y_dims, NPY_BOOL, 0)
    PyArray_FillWithScalar(y, 0)
    return y


cdef anynan_0d(ndarray a, int int_input):
    out = a[()]
    if out == out:
        return False
    else:
        return True


# reducer -------------------------------------------------------------------

# pointer to functions that reduce along ALL axes
ctypedef object (*fall_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int)

# pointer to functions that reduce along ONE axis
ctypedef ndarray (*fone_t)(np.flatiter, Py_ssize_t, Py_ssize_t, int,
                           np.npy_intp*, int)

# pointer to functions that handle 0d arrays
ctypedef object (*f0d_t)(ndarray, int)


cdef reducer(arr, axis,
             fall_t fall_float64,
             fall_t fall_float32,
             fall_t fall_int64,
             fall_t fall_int32,
             fone_t fone_float64,
             fone_t fone_float32,
             fone_t fone_int64,
             fone_t fone_int32,
             f0d_t f0d,
             int int_input=0,
             int ravel=0,
             int copy=0):

    # convert to array if necessary
    cdef ndarray a
    if type(arr) is ndarray:
        a = arr
    else:
        a = np.array(arr, copy=False)

    # input array
    if ravel == 1:
        a = PyArray_Ravel(a, NPY_CORDER)
    if copy == 1:
        a = PyArray_Copy(a)

    cdef np.flatiter ita
    cdef Py_ssize_t stride, length, i, j
    cdef int dtype = PyArray_TYPE(a)
    cdef int a_ndim = PyArray_NDIM(a)

    # output array, if needed
    cdef ndarray y
    cdef np.npy_intp *adim
    cdef np.npy_intp *y_dims = [0, 0, 0, 0, 0, 0, 0, 0, 0]  # TODO max ndim=10

    # defend against 0d beings
    if a_ndim == 0:
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
            axis_int += a_ndim
            if axis_int < 0:
                raise ValueError("axis(=%d) out of bounds" % axis)
        elif axis_int >= a_ndim:
            raise ValueError("axis(=%d) out of bounds" % axis)
        if a_ndim == 1 and axis_int == 0:
            reduce_all = 1
        axis_reduce = axis_int

    # input iterator
    ita = PyArray_IterAllButAxis(a, &axis_reduce)
    stride = a.strides[axis_reduce]
    length = a.shape[axis_reduce]

    if reduce_all == 1:
        # reduce over all axes
        if dtype == NPY_float64:
            return fall_float64(ita, stride, length, int_input)
        elif dtype == NPY_float32:
            return fall_float32(ita, stride, length, int_input)
        elif dtype == NPY_int64:
            return fall_int64(ita, stride, length, int_input)
        elif dtype == NPY_int32:
            return fall_int32(ita, stride, length, int_input)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
    else:
        # reduce over a single axis; a_ndim > 1
        adim = np.PyArray_DIMS(a)
        j = 0
        for i in range(a_ndim):
            if i != axis_reduce:
                y_dims[j] = adim[i]
                j += 1
        if dtype == NPY_float64:
            y = fone_float64(ita, stride, length, a_ndim, y_dims, int_input)
        elif dtype == NPY_float32:
            y = fone_float32(ita, stride, length, a_ndim, y_dims, int_input)
        elif dtype == NPY_int64:
            y = fone_int64(ita, stride, length, a_ndim, y_dims, int_input)
        elif dtype == NPY_int32:
            y = fone_int32(ita, stride, length, a_ndim, y_dims, int_input)
        else:
            raise TypeError("Unsupported dtype (%s)." % a.dtype)
        return y
