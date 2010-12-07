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

i32 = np.dtype(np.int32)
i64 = np.dtype(np.int64)
f64 = np.dtype(np.float64)
N = None

cdef extern from "math.h":
    double sqrt(double x)
