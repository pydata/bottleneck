import numpy as np
cimport numpy as np
import cython

ctypedef np.float64_t DTYPE_t
cdef double NAN = <double> np.nan

f64 = np.dtype(np.float64)
N = None

cdef extern from "math.h":
    double sqrt(double x)

MOVE_WINDOW_ERR_MSG = "Moving window (=%d) must between 1 and %d, inclusive"    
