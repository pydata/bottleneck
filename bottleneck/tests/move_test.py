"Test moving window functions."

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
try:
    import scipy
    SCIPY = True
except ImportError:
    SCIPY = False
nan = np.nan
import bottleneck as bn


def arrays(dtypes=bn.dtypes, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3)]}
    ss[3] = {'size':  6, 'shapes': [(1,2,3)]}
    if SCIPY:
        # Unaccelerated fallback requires scipy
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
                if nans:
                    for i in range(a.size):
                        a.flat[i] = np.nan
                        yield a
                        yield -a

def unit_maker(func, func0, decimal=np.inf, nans=True):
    "Test that bn.xxx gives the same output as a reference function."
    msg = '\nfunc %s | window %d | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in range(-arr.ndim, arr.ndim):
            windows = range(1, arr.shape[axis])
            if len(windows) == 0:
                windows = [1]
            for window in windows:
                with np.errstate(invalid='ignore'):
                    actual = func(arr, window, axis=axis)
                    desired = func0(arr, window, axis=axis, method='loop')
                tup = (func.__name__, window, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), str(axis), arr)
                err_msg = msg % tup
                if (decimal < np.inf) and (np.isfinite(arr).sum() > 0):
                    assert_array_almost_equal(actual, desired, decimal,
                                              err_msg)
                else:
                    assert_array_equal(actual, desired, err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))

def test_move_sum():
    "Test move_sum."
    yield unit_maker, bn.move_sum, bn.slow.move_sum, 5

def test_move_nansum():
    "Test move_nansum."
    yield unit_maker, bn.move_nansum, bn.slow.move_nansum, 5

def test_move_mean():
    "Test move_mean."
    yield unit_maker, bn.move_mean, bn.slow.move_mean, 5

def test_move_nanmean():
    "Test move_nanmean."
    yield unit_maker, bn.move_nanmean, bn.slow.move_nanmean, 5

def test_move_std():
    "Test move_std."
    yield unit_maker, bn.move_std, bn.slow.move_std, 5

def test_move_nanstd():
    "Test move_nanstd."
    yield unit_maker, bn.move_nanstd, bn.slow.move_nanstd, 5

def test_move_min():
    "Test move_min."
    yield unit_maker, bn.move_min, bn.slow.move_min, 5

def test_move_max():
    "Test move_max."
    yield unit_maker, bn.move_max, bn.slow.move_max, 5

def test_move_nanmin():
    "Test move_nanmin."
    yield unit_maker, bn.move_nanmin, bn.slow.move_nanmin, 5

def test_move_nanmax():
    "Test move_nanmax."
    yield unit_maker, bn.move_nanmax, bn.slow.move_nanmax, 5
