import numpy as np
cimport numpy as np
import cython
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport PyArray_EMPTY, import_array
import_array()

ctypedef np.float64_t DTYPE_t
cdef double NAN = <double> np.nan

cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float64_t MINfloat64 = np.NINF

cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float64_t MAXfloat64 = np.inf

i32 = np.dtype(np.int32)
i64 = np.dtype(np.int64)
f64 = np.dtype(np.float64)
N = None
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float64 = np.dtype(np.float64)

cdef extern from "math.h":
    double sqrt(double x)
