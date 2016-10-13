"Test moving window functions."

from nose.tools import assert_true
import numpy as np
from numpy.testing import (assert_equal, assert_array_almost_equal,
                           assert_raises)
import bottleneck as bn

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_move():
    "test move functions"
    for func in bn.get_functions('move'):
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
                b = a.reshape(shape)
                yield b
                if b.ndim > 1:
                    yield b.T


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
    for i, a in enumerate(arrays()):
        axes = range(-1, a.ndim)
        for axis in axes:
            windows = range(1, a.shape[axis])
            for window in windows:
                min_counts = list(range(1, window + 1)) + [None]
                for min_count in min_counts:
                    actual = func(a, window, min_count, axis=axis)
                    desired = func0(a, window, min_count, axis=axis)
                    tup = (func_name, window, str(min_count), 'a'+str(i),
                           str(a.dtype), str(a.shape), str(axis), a)
                    err_msg = fmt % tup
                    aaae(actual, desired, decimal, err_msg)
                    err_msg += '\n dtype mismatch %s %s'
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


# ---------------------------------------------------------------------------
# Test with arrays that are not C ordered

def test_strides():
    "test move functions with non-C ordered arrays"
    for func in bn.get_functions('move'):
        yield unit_maker_strides, func


def arrays_strides(dtypes=DTYPES):
    "Iterator that yields non-C orders arrays."

    # 1d
    for dtype in dtypes:
        a = np.arange(12).astype(dtype)
        for start in range(3):
            for step in range(1, 3):
                yield a[start::step]  # don't use astype here; copy created

    # 2d
    for dtype in dtypes:
        a = np.arange(12).reshape(4, 3).astype(dtype)
        yield a[::2]
        yield a[:, ::2]
        yield a[::2][:, ::2]

    # 3d
    for dtype in dtypes:
        a = np.arange(60).reshape(3, 4, 5).astype(dtype)
        for start in range(2):
            for step in range(1, 2):
                yield a[start::step]
                yield a[:, start::step]
                yield a[:, :, start::step]


def unit_maker_strides(func, decimal=5):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    fmt = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    fmt += '\nInput array:\n%s\n'
    fmt += '\nStrides: %s\n'
    fmt += '\nFlags: \n%s\n'
    name = func.__name__
    func0 = eval('bn.slow.%s' % name)
    for i, a in enumerate(arrays_strides()):
        axes = list(range(-1, a.ndim))
        for axis in axes:
            # do not use a.copy() here because it will C order the array
            actual = func(a, window=2, min_count=1, axis=axis)
            desired = func0(a, window=2, min_count=1, axis=axis)
            tup = (name, 'a'+str(i), str(a.dtype), str(a.shape),
                   str(axis), a, a.strides, a.flags)
            err_msg = fmt % tup
            assert_array_almost_equal(actual, desired, decimal, err_msg)
            err_msg += '\n dtype mismatch %s %s'


# ---------------------------------------------------------------------------
# Test argument parsing

def test_arg_parsing():
    "test argument parsing"
    for func in bn.get_functions('move'):
        yield unit_maker_argparse, func


def unit_maker_argparse(func, decimal=5):
    "test argument parsing."

    name = func.__name__
    func0 = eval('bn.slow.%s' % name)

    a = np.array([1., 2, 3])

    fmt = '\n%s' % func
    fmt += '%s\n'
    fmt += '\nInput array:\n%s\n' % a

    actual = func(a, 2)
    desired = func0(a, 2)
    err_msg = fmt % "(a, 2)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, 2, 1)
    desired = func0(a, 2, 1)
    err_msg = fmt % "(a, 2, 1)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, window=2)
    desired = func0(a, window=2)
    err_msg = fmt % "(a, window=2)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, window=2, min_count=1)
    desired = func0(a, window=2, min_count=1)
    err_msg = fmt % "(a, window=2, min_count=1)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, window=2, min_count=1, axis=0)
    desired = func0(a, window=2, min_count=1, axis=0)
    err_msg = fmt % "(a, window=2, min_count=1, axis=0)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, min_count=1, window=2, axis=0)
    desired = func0(a, min_count=1, window=2, axis=0)
    err_msg = fmt % "(a, min_count=1, window=2, axis=0)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, axis=-1, min_count=None, window=2)
    desired = func0(a, axis=-1, min_count=None, window=2)
    err_msg = fmt % "(a, axis=-1, min_count=None, window=2)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a=a, axis=-1, min_count=None, window=2)
    desired = func0(a=a, axis=-1, min_count=None, window=2)
    err_msg = fmt % "(a=a, axis=-1, min_count=None, window=2)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    if name in ('move_std', 'move_var'):
        actual = func(a, 2, 1, -1, ddof=1)
        desired = func0(a, 2, 1, -1, ddof=1)
        err_msg = fmt % "(a, 2, 1, -1, ddof=1)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

    # regression test: make sure len(kwargs) == 0 doesn't raise
    args = (a, 1, 1, -1)
    kwargs = {}
    func(*args, **kwargs)


def test_arg_parse_raises():
    "test argument parsing raises in move"
    for func in bn.get_functions('move'):
        yield unit_maker_argparse_raises, func


def unit_maker_argparse_raises(func):
    "test argument parsing raises in move"
    a = np.array([1., 2, 3])
    assert_raises(TypeError, func)
    assert_raises(TypeError, func, axis=a)
    assert_raises(TypeError, func, a, 2, axis=0, extra=0)
    assert_raises(TypeError, func, a, 2, axis=0, a=a)
    assert_raises(TypeError, func, a, 2, 2, 0, 0, 0)
    assert_raises(TypeError, func, a, 2, axis='0')
    assert_raises(TypeError, func, a, 1, min_count='1')
    if func.__name__ not in ('move_std', 'move_var'):
        assert_raises(TypeError, func, a, 2, ddof=0)


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
    a = np.array([1, 2, np.inf, 3, 4, 5])
    window = 3
    min_count = 2
    for func in funcs:
        actual = func(a, window=window, min_count=min_count)
        func0 = eval('bn.slow.%s' % func.__name__)
        desired = func0(a, window=window, min_count=min_count)
        err_msg = fmt % (func.__name__, window, min_count, a)
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
