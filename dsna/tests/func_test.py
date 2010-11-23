"Test functions."

import numpy as np
import scipy.stats as sp
from numpy.testing import (assert_equal, assert_array_equal, assert_raises,
                           assert_array_almost_equal)
nan = np.nan
import dsna as ds


def arrays(dtypes=['int32', 'int64', 'float64']):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3), (6,1)]}
    ss[3] = {'size': 24, 'shapes': [(1,1,24), (24,1,1), (1,24,1), (2,3,4)]}
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
                    a.flat[i] = np.nan
                    yield a
                    yield -a
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                    yield -a

def unit_maker(func, func0, decimal=np.inf):
    "Test that ds.xxx gives the same output as np.."
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays()):
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            actual = func(arr, axis=axis)
            desired = func0(arr, axis=axis)
            tup = (func.__name__, 'a'+str(i), str(arr.dtype), str(arr.shape),
                   str(axis), arr)
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

def test_sum():
    "Test sum."
    yield unit_maker, ds.sum, np.nansum

def test_max():
    "Test max."
    yield unit_maker, ds.max, np.nanmax

def test_min():
    "Test min."
    yield unit_maker, ds.min, np.nanmin

def test_mean():
    "Test mean."
    yield unit_maker, ds.mean, sp.nanmean, 13

def test_std():
    "Test min."
    yield unit_maker, ds.std, scipy_nanstd

def test_var():
    "Test min."
    yield unit_maker, ds.var, scipy_nanstd_squared, 13

# ---------------------------------------------------------------------------
# Check that exceptions are raised

def test_max_size_zero():
    "Test max for size zero input arrays."
    dtypes = ['int32', 'int64', 'float64']
    shapes = [(0,), (2,0), (1,2,0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, ds.max, a)
            assert_raises(ValueError, np.nanmax, a)
            
def test_min_size_zero():
    "Test min for size zero input arrays."
    dtypes = ['int32', 'int64', 'float64']
    shapes = [(0,), (2,0), (1,2,0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, ds.min, a)
            assert_raises(ValueError, np.nanmin, a)

# ---------------------------------------------------------------------------
# Unit test utility functions

def scipy_nanstd(a, axis, bias=True):
    "For bias to True for scipy.stats.nanstd"
    return sp.nanstd(a, axis, bias=True)

def scipy_nanstd_squared(a, axis, bias=True):
    "For bias to True for scipy.stats.nanstd"
    x = sp.nanstd(a, axis, bias=True)
    return x * x

