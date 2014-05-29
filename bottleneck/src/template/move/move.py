from __future__ import absolute_import

from bottleneck.src.template.template import template
import bottleneck.src.template.template as tempmod
import os.path

from .move_sum import move_sum
from .move_nansum import move_nansum
from .move_mean import move_mean
from .move_median import move_median
from .move_nanmean import move_nanmean
from .move_std import move_std
from .move_nanstd import move_nanstd
from .move_min import move_min
from .move_max import move_max
from .move_nanmin import move_nanmin
from .move_nanmax import move_nanmax

funcs = {}
funcs['move_sum'] = move_sum
funcs['move_nansum'] = move_nansum
funcs['move_mean'] = move_mean
funcs['move_median'] = move_median
funcs['move_nanmean'] = move_nanmean
funcs['move_std'] = move_std
funcs['move_nanstd'] = move_nanstd
funcs['move_min'] = move_min
funcs['move_max'] = move_max
funcs['move_nanmin'] = move_nanmin
funcs['move_nanmax'] = move_nanmax

header = """#cython: embedsignature=True

import numpy as np
cimport numpy as np
import cython
from libc cimport stdlib
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64
from numpy cimport (PyArray_EMPTY, PyArray_TYPE, PyArray_NDIM, PyArray_DIMS,
                    import_array, PyArray_Copy)
import_array()
import bottleneck as bn

cdef double NAN = <double> np.nan

cdef np.float64_t MINfloat64 = np.NINF
cdef np.float64_t MAXfloat64 = np.inf

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

if np.int_ == np.int32:
    NPY_int_ = NPY_int32
elif np.int_ == np.int64:
    NPY_int_ = NPY_int64

MOVE_WINDOW_ERR_MSG = "Moving window (=%d) must between 1 and %d, inclusive"

include "move_sum.pyx"
include "move_nansum.pyx"
include "move_mean.pyx"
include "move_median.pyx"
include "move_nanmean.pyx"
include "move_std.pyx"
include "move_nanstd.pyx"
include "move_min.pyx"
include "move_max.pyx"
include "move_nanmin.pyx"
include "move_nanmax.pyx"
"""


def movepyx(funcs=funcs):
    for func in funcs:
        template(funcs[func])
    template_path = os.path.dirname(tempmod.__file__)
    fid = open(os.path.join(template_path, '..', "move/move.pyx"), 'w')
    fid.write(header)
    fid.close()
