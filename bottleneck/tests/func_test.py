"Test functions."

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal, assert_raises,
                           assert_array_almost_equal)
nan = np.nan
import bottleneck as bn


def arrays(dtypes=bn.dtypes, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3), (6,1)]}
    ss[3] = {'size': 24, 'shapes': [(1,1,24), (24,1,1), (1,24,1), (2,3,4)]}
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
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                    yield -a

def unit_maker(func, func0, decimal=np.inf, nans=True):
    "Test that bn.xxx gives the same output as np.."
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            with np.errstate(invalid='ignore'):
                actualraised = False
                try:
                    actual = func(arr, axis=axis)
                except:
                    actualraised = True
                desiredraised = False
                try:
                    desired = func0(arr, axis=axis)
                except:
                    desiredraised = True
            if actualraised and desiredraised:
                pass
            else:
                tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), str(axis), arr)
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

def test_nanmax():
    "Test nanmax."
    yield unit_maker, bn.nanmax, bn.slow.nanmax

def test_nanargmax():
    "Test nanargmax."
    yield unit_maker, bn.nanargmax, bn.slow.nanargmax

def test_nanmin():
    "Test nanmin."
    yield unit_maker, bn.nanmin, bn.slow.nanmin

def test_nanmean():
    "Test nanmean."
    yield unit_maker, bn.nanmean, bn.slow.nanmean, 5

def test_nanstd():
    "Test nanstd."
    yield unit_maker, bn.nanstd, bn.slow.nanstd, 5

def test_nanvar():
    "Test nanvar."
    yield unit_maker, bn.nanvar, bn.slow.nanvar, 5

def test_median():
    "Test median."
    yield unit_maker, bn.median, bn.slow.median, np.inf, False

def test_nanmedian():
    "Test nanmedian."
    yield unit_maker, bn.nanmedian, bn.slow.nanmedian, np.inf, False

# ---------------------------------------------------------------------------
# Check that exceptions are raised

def test_nanmax_size_zero(dtypes=bn.dtypes):
    "Test nanmax for size zero input arrays."
    shapes = [(0,), (2,0), (1,2,0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmax, a)
            assert_raises(ValueError, bn.slow.nanmax, a)
            
def test_nanmin_size_zero(dtypes=bn.dtypes):
    "Test nanmin for size zero input arrays."
    shapes = [(0,), (2,0), (1,2,0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmin, a)
            assert_raises(ValueError, bn.slow.nanmin, a)
