"Test moving window functions."

import warnings

from nose.tools import assert_true
import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal)
nan = np.nan
import bottleneck as bn

DTYPES = [np.float64, np.float32, np.int64, np.int32, np.float16]


def arrays(dtypes=DTYPES, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1, 6), (2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}
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
    msg = ('\nfunc %s | window %d | min_count %s | input %s (%s) | shape %s | '
           'axis %s\n')
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in range(-arr.ndim, arr.ndim):
            windows = range(1, arr.shape[axis])
            if len(windows) == 0:
                windows = [1]
            for window in windows:
                min_counts = [w for w in windows if w <= window]
                min_counts.append(None)
                for min_count in min_counts:
                    with np.errstate(invalid='ignore'):
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if func.__name__ == 'move_median':
                                actual = func(arr, window, axis=axis)
                            else:
                                actual = func(arr, window, min_count,
                                              axis=axis)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            if func.__name__ == 'move_median':
                                desired = func0(arr, window, axis=axis)
                            else:
                                desired = func0(arr, window, min_count,
                                                axis=axis)
                    tup = (func.__name__, window, str(min_count), 'a'+str(i),
                           str(arr.dtype), str(arr.shape), str(axis), arr)
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


def test_move_mean():
    "Test move_mean."
    yield unit_maker, bn.move_mean, bn.slow.move_mean, 5


def test_move_std():
    "Test move_std."
    yield unit_maker, bn.move_std, bn.slow.move_std, 5


def test_move_min():
    "Test move_min."
    yield unit_maker, bn.move_min, bn.slow.move_min, 5


def test_move_max():
    "Test move_max."
    yield unit_maker, bn.move_max, bn.slow.move_max, 5

def test_move_median():
    "Test move_median."
    yield unit_maker, bn.move_median, bn.slow.move_median, 5, False


# ----------------------------------------------------------------------------
# Regression test for square roots of negative numbers

def test_move_std_sqrt():
    "Test move_std for neg sqrt."

    a = [0.0011448196318903589,
         0.00028718669878572767,
         0.00028718669878572767,
         0.00028718669878572767,
         0.00028718669878572767]
    err_msg = "Square root of negative number. ndim = %d"
    b = bn.move_std(a, window=3)
    assert_true(np.isfinite(b[2:]).all(), err_msg % 1)

    a2 = np.array([a, a])
    b = bn.move_std(a2, window=3, axis=1)
    assert_true(np.isfinite(b[:, 2:]).all(), err_msg % 2)

    a3 = np.array([[a, a], [a, a]])
    b = bn.move_std(a3, window=3, axis=2)
    assert_true(np.isfinite(b[:, :, 2:]).all(), err_msg % 3)
