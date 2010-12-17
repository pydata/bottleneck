import numpy as np
cimport numpy as np
import cython
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport PyArray_EMPTY, import_array
import_array()
import bottleneck as bn

cdef double NAN = <double> np.nan

int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)

cdef extern from "math.h":
    double sqrt(double x)
