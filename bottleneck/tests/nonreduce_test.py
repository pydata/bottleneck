"Test replace()."

import warnings

import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_raises
nan = np.nan
import bottleneck as bn

DTYPES = [np.float64, np.float32, np.int64, np.int32, np.float16]


def arrays(dtypes=DTYPES, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0, 0), (2, 0), (2, 0, 1)]}
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
            if issubclass(a.dtype.type, np.inexact):
                if nans:
                    for i in range(a.size):
                        a.flat[i] = np.nan
                        yield a
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
    if nans:
        # nanmedian regression tests
        a = np.array([1, nan, nan, 2])
        yield a
        a = np.vstack((a, a))
        yield a
        yield a.reshape(1, 2, 4)


def unit_maker(func, func0, nans=True):
    "Test that bn.xxx gives the same output as np.xxx."
    msg = '\nfunc %s | input %s (%s) | shape %s | old %f | new %f\n'
    msg += '\nInput array:\n%s\n'
    olds = [0, np.nan, np.inf]
    news = [1, 0, np.nan]
    for i, arr in enumerate(arrays(nans=nans)):
        for old in olds:
            for new in news:
                if not issubclass(arr.dtype.type, np.inexact):
                    if not np.isfinite(old):
                        # Cannot safely cast to int
                        continue
                    if not np.isfinite(new):
                        # Cannot safely cast to int
                        continue
                actual = arr.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    func(actual, old, new)
                desired = arr.copy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    func0(desired, old, new)
                tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), old, new, arr)
                err_msg = msg % tup
                assert_array_equal(actual, desired, err_msg=err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


def test_replace():
    "Test replace."
    yield unit_maker, bn.replace, bn.slow.replace


# ---------------------------------------------------------------------------
# Check that exceptions are raised

def test_replace_unsafe_cast():
    "Test replace for unsafe casts."
    dtypes = ['int32', 'int64']
    shapes = [(0,), (2, 0), (1, 2, 0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.replace, a, 0.1, 0)
            assert_raises(ValueError, bn.replace, a, 0, 0.1)
            assert_raises(ValueError, bn.slow.replace, a, 0.1, 0)
            assert_raises(ValueError, bn.slow.replace, a, 0, 0.1)


def test_non_array():
    "Test that non-array input raises"
    a = [1, 2, 3]
    assert_raises(TypeError, bn.replace, a, 0, 1)
    a = (1, 2, 3)
    assert_raises(TypeError, bn.replace, a, 0, 1)


# ---------------------------------------------------------------------------
# Make sure bn.replace and bn.slow.replace can handle int arrays where
# user wants to replace nans

def test_replace_nan_int():
    "Test replace, int array, old=nan, new=0"
    a = np.arange(2*3*4).reshape(2, 3, 4)
    actual = a.copy()
    bn.replace(actual, np.nan, 0)
    desired = a.copy()
    msg = 'replace failed on int input looking for nans'
    assert_array_equal(actual, desired, err_msg=msg)
    actual = a.copy()
    bn.slow.replace(actual, np.nan, 0)
    msg = 'slow.replace failed on int input looking for nans'
    assert_array_equal(actual, desired, err_msg=msg)
