
# nanmean sandbox example -----------------------------------------------
#
# Convert into C and compile using the bottleneck/sandbox/setup.py file.

import numpy as np
cimport numpy as np
import cython

cdef double NAN = <double> np.nan


@cython.boundscheck(False)
@cython.wraparound(False)
def nanmean(np.ndarray[np.float64_t, ndim=1] a):
    "nanmean of 1d numpy array with dtype=np.float64 along axis=0."
    cdef Py_ssize_t i
    cdef int a0 = a.shape[0], count = 0
    cdef np.float64_t asum = 0, ai
    for i in range(a0):
        ai = a[i]
        if ai == ai:
            asum += ai
            count += 1
    if count > 0:
        return np.float64(asum / count)
    else:
        return np.float64(NAN)
