"Test moving window functions."

from nose.tools import assert_true
import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal
import bottleneck as bn
from .functions import move_functions

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_move():
    "test move functions"
    for func in move_functions():
        yield unit_maker, func


def arrays(dtypes=DTYPES):
    "Iterator that yield arrays to use for unit testing."
    nan = np.nan
    yield np.array([1, 2, 3]) + 1e9  # check that move_std is robust
    yield np.array([1, 2, 3], dtype='>f4')
    yield np.array([1, 2, 3], dtype='<f4')
    yield np.array([1, 1, 1])  # move_argmax should pick index of rightmost tie
    yield np.array([1, 2, 3], dtype=np.float16)  # make sure slow is called
    ss = {}
    ss[1] = {'size':  8, 'shapes': [(8,)]}
    ss[2] = {'size': 12, 'shapes': [(2, 6), (4, 3)]}
    ss[3] = {'size': 60, 'shapes': [(3, 4, 5)]}
    ss[4] = {'size': 48, 'shapes': [(2, 2, 3, 4)]}
    rs = np.random.RandomState([1, 2, 3])
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            if issubclass(a.dtype.type, np.inexact):
                idx = rs.rand(*a.shape) < 0.2
                a[idx] = nan
                idx = rs.rand(*a.shape) < 0.2
                a[idx] *= -1
            rs.shuffle(a)
            for shape in shapes:
                yield a.reshape(shape)


def unit_maker(func):
    "Test that bn.xxx gives the same output as a reference function."
    fmt = ('\nfunc %s | window %d | min_count %s | input %s (%s) | shape %s | '
           'axis %s\n')
    fmt += '\nInput array:\n%s\n'
    aaae = assert_array_almost_equal
    func_name = func.__name__
    func0 = eval('bn.slow.%s' % func_name)
    if func_name == "move_var":
        decimal = 3
    else:
        decimal = 5
    for i, arr in enumerate(arrays()):
        axes = range(-1, arr.ndim)
        for axis in axes:
            windows = range(1, arr.shape[axis])
            for window in windows:
                min_counts = list(range(1, window + 1)) + [None]
                for min_count in min_counts:
                    actual = func(arr, window, min_count, axis=axis)
                    desired = func0(arr, window, min_count, axis=axis)
                    tup = (func_name, window, str(min_count), 'a'+str(i),
                           str(arr.dtype), str(arr.shape), str(axis), arr)
                    err_msg = fmt % tup
                    aaae(actual, desired, decimal, err_msg)
                    err_msg += '\n dtype mismatch %s %s'
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


# ---------------------------------------------------------------------------
# Only some moving window functions can handle input arrays that contains inf.
#
# Those that can't handle inf: move_sum, move_mean, move_std, move_var
#
# Adding code to deal with the rare case of inf in the input array slows down
# the functions and makes the code more complex and harder to maintain.

def test_move_inf():
    "test inf in input array"
    fmt = '\nfunc %s | window %d | min_count %s\n\nInput array:\n%s\n'
    funcs = [bn.move_min, bn.move_max, bn.move_argmin, bn.move_argmax,
             bn.move_rank, bn.move_median]
    arr = np.array([1, 2, np.inf, 3, 4, 5])
    window = 3
    min_count = 2
    for func in funcs:
        actual = func(arr, window=window, min_count=min_count)
        func0 = eval('bn.slow.%s' % func.__name__)
        desired = func0(arr, window=window, min_count=min_count)
        err_msg = fmt % (func.__name__, window, min_count, arr)
        assert_array_almost_equal(actual, desired, decimal=5, err_msg=err_msg)


# ---------------------------------------------------------------------------
# move_median.c is complicated. Let's do some more testing.
#
# If you make changes to move_median.c then do lots of tests by increasing
# range(100) in the two functions below to range(10000). And for extra credit
# increase size to 30. With those two changes the unit tests will take a
# LONG time to run.

def test_move_median_with_nans():
    "test move_median.c with nans"
    fmt = '\nfunc %s | window %d | min_count %s\n\nInput array:\n%s\n'
    aaae = assert_array_almost_equal
    min_count = 1
    size = 10
    func = bn.move_median
    func0 = bn.slow.move_median
    rs = np.random.RandomState([1, 2, 3])
    for i in range(100):
        a = np.arange(size, dtype=np.float64)
        idx = rs.rand(*a.shape) < 0.1
        a[idx] = np.inf
        idx = rs.rand(*a.shape) < 0.2
        a[idx] = np.nan
        rs.shuffle(a)
        for window in range(2, size + 1):
            actual = func(a, window=window, min_count=min_count)
            desired = func0(a, window=window, min_count=min_count)
            err_msg = fmt % (func.__name__, window, min_count, a)
            aaae(actual, desired, decimal=5, err_msg=err_msg)


def test_move_median_without_nans():
    "test move_median.c without nans"
    fmt = '\nfunc %s | window %d | min_count %s\n\nInput array:\n%s\n'
    aaae = assert_array_almost_equal
    min_count = 1
    size = 10
    func = bn.move_median
    func0 = bn.slow.move_median
    rs = np.random.RandomState([1, 2, 3])
    for i in range(100):
        a = np.arange(size, dtype=np.int64)
        rs.shuffle(a)
        for window in range(2, size + 1):
            actual = func(a, window=window, min_count=min_count)
            desired = func0(a, window=window, min_count=min_count)
            err_msg = fmt % (func.__name__, window, min_count, a)
            aaae(actual, desired, decimal=5, err_msg=err_msg)


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
