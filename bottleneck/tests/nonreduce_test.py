"Test replace()."

import warnings

import numpy as np
from numpy.testing import assert_equal, assert_array_equal, assert_raises
import bottleneck as bn
from .reduce_test import array_iter

DTYPES = [np.float64, np.float32, np.int64, np.int32]
nan = np.nan


def test_nonreduce():
    "test nonreduce functions"
    for func in bn.get_functions('nonreduce'):
        yield unit_maker, func


def arrays(dtypes=DTYPES):
    "Iterator that yield arrays to use for unit testing."

    nan = np.nan
    inf = np.inf

    # nan and inf
    yield np.array([inf, nan])
    yield np.array([inf, -inf])
    yield np.array([nan, inf])

    # byte swapped
    yield np.array([1, 2, 3], dtype='>f4')
    yield np.array([1, 2, 3], dtype='<f4')

    # make sure slow is called
    yield np.array([1, 2, 3], dtype=np.float16)

    # ties
    yield np.array([0, 0, 0])
    yield np.array([0, 0, 0], dtype=np.float64)
    yield np.array([1, 1, 1], dtype=np.float64)

    # 0d input
    yield np.array(-9)
    yield np.array(0)
    yield np.array(9)
    yield np.array(-9.0)
    yield np.array(0.0)
    yield np.array(9.0)
    yield np.array(-inf)
    yield np.array(inf)
    yield np.array(nan)

    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0, 0), (2, 0), (2, 0, 1)]}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1, 6), (2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for ndim in ss:
            size = ss[ndim]['size']
            shapes = ss[ndim]['shapes']
            for dtype in dtypes:
                a = np.arange(size, dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = 0
                rs.shuffle(a)
                for shape in shapes:
                    yield a.reshape(shape)


def unit_maker(func):
    "Test that bn.xxx gives the same output as np.xxx."
    msg = '\nfunc %s | input %s (%s) | shape %s | old %f | new %f\n'
    msg += '\nInput array:\n%s\n'
    name = func.__name__
    func0 = eval('bn.slow.%s' % name)
    olds = [0, np.nan, np.inf]
    news = [1, 0, np.nan]
    for i, arr in enumerate(array_iter(arrays)):
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
                tup = (name, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), old, new, arr)
                err_msg = msg % tup
                assert_array_equal(actual, desired, err_msg=err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


# ---------------------------------------------------------------------------
# Check that exceptions are raised

def test_replace_unsafe_cast():
    "Test replace for unsafe casts"
    dtypes = ['int32', 'int64']
    for dtype in dtypes:
        a = np.zeros(3, dtype=dtype)
        assert_raises(ValueError, bn.replace, a.copy(), 0.1, 0)
        assert_raises(ValueError, bn.replace, a.copy(), 0, 0.1)
        assert_raises(ValueError, bn.slow.replace, a.copy(), 0.1, 0)
        assert_raises(ValueError, bn.slow.replace, a.copy(), 0, 0.1)


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
