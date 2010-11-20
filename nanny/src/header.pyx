import numpy as np
cimport numpy as np
import cython

ctypedef np.float64_t DTYPE_t
cdef double NAN = <double> np.nan

i32 = np.dtype(np.int32)
i64 = np.dtype(np.int64)
f64 = np.dtype(np.float64)
N = None
