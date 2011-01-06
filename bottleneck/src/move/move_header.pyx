import numpy as np
cimport numpy as np
import cython
cimport stdlib
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport PyArray_EMPTY, import_array
import_array()
import bottleneck as bn

cdef double NAN = <double> np.nan

# Used by move_min and move_max
cdef struct pairs:
    double value
    int death

int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)

cdef extern from "math.h":
    double sqrt(double x)

MOVE_WINDOW_ERR_MSG = "Moving window (=%d) must between 1 and %d, inclusive"    
