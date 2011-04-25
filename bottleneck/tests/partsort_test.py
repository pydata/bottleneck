"Test functions."

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
nan = np.nan
import bottleneck as bn


def arrays(dtypes=bn.dtypes):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0,0), (2,0), (2,0,1)]}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3)]}
    ss[3] = {'size':  6, 'shapes': [(1,2,3)]}
    ss[4] = {'size': 24, 'shapes': [(1,2,3,4)]}  # Unaccelerated
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            for shape in shapes:
                a = a.reshape(shape)
                yield a
                yield -a
            if issubclass(a.dtype.type, np.inexact): 
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                    yield -a

def unit_maker(func, func0, decimal=np.inf):
    "Test that bn.partsort gives the same output as bn.slow.partsum."
    msg = '\nfunc %s | input %s (%s) | shape %s | n %d | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays()):
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            if axis is None:
                n = arr.size
            else:
                n = arr.shape[axis]
            n = max(n / 2, 1)
            with np.errstate(invalid='ignore'):
                actual = func(arr.copy(), n, axis=axis)
                actual[:n] = np.sort(actual[:n], axis=axis)
                actual[n:] = np.sort(actual[n:], axis=axis)
                desired = func0(arr.copy(), n, axis=axis)
            tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                   str(arr.shape), n, str(axis), arr)
            err_msg = msg % tup
            if (decimal < np.inf) and (np.isfinite(arr).sum() > 0):
                assert_array_almost_equal(actual, desired, decimal, err_msg)
            else:
                assert_array_equal(actual, desired, err_msg)
            err_msg += '\n dtype mismatch %s %s'
            if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                da = actual.dtype
                dd = desired.dtype
                assert_equal(da, dd, err_msg % (da, dd))

def test_partsort():
    "Test partsort."
    yield unit_maker, bn.partsort, bn.slow.partsort
