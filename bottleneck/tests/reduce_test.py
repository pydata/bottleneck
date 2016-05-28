"Test reduce functions."

import warnings
import traceback

from nose.tools import ok_
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
                           assert_array_almost_equal)

import bottleneck as bn
from .functions import reduce_functions

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_reduce():
    "test reduce functions"
    for func in reduce_functions():
        yield unit_maker, func


def arrays(dtypes=DTYPES):
    "Iterator that yields arrays to use for unit testing."
    nan = np.nan
    inf = np.inf
    yield np.array([1, 2, 3], dtype='>f4')
    yield np.array([1, 2, 3], dtype='<f4')
    yield np.array([1, 2, 3], dtype=np.float16)  # make sure slow is called
    yield np.array([0, 0, 0])  # nanargmax/nanargmin regression tests
    yield np.array([0, 0, 0], dtype=np.float64)
    yield np.array([inf, nan])
    # yield np.array([nan, inf])  np.nanargmin can't handle this one
    yield np.array([inf, -inf])
    yield np.array([1, nan, nan, 2])  # nanmedian regression tests
    # check 0d input
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
    ss[1] = {'size':  8, 'shapes': [(8,)]}
    ss[2] = {'size': 12, 'shapes': [(2, 6), (3, 4)]}
    ss[3] = {'size': 16, 'shapes': [(2, 2, 4)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}
    rs = np.random.RandomState([1, 2, 4])
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
            rs.shuffle(a)
            for shape in shapes:
                yield a.reshape(shape)


def unit_maker(func, decimal=5):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    fmt = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    fmt += '\nInput array:\n%s\n'
    func0 = eval('bn.slow.%s' % func.__name__)
    for i, arr in enumerate(arrays()):
        if arr.ndim == 0:
            axes = [None]  # numpy can't handle e.g. np.nanmean(9, axis=-1)
        else:
            axes = list(range(-1, arr.ndim)) + [None]
        for axis in axes:
            actual = 'Crashed'
            desired = 'Crashed'
            actualraised = False
            try:
                actual = func(arr.copy(), axis=axis)
            except:
                actualraised = True
            desiredraised = False
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    desired = func0(arr.copy(), axis=axis)
            except:
                desiredraised = True
            if actualraised and desiredraised:
                pass
            else:
                name = func.__name__
                tup = (name, 'a'+str(i), str(arr.dtype), str(arr.shape),
                       str(axis), arr)
                err_msg = fmt % tup
                if actualraised != desiredraised:
                    if actualraised:
                        fmt2 = '\nbn.%s raised\nbn.slow.%s ran\n\n%s'
                    else:
                        fmt2 = '\nbn.%s ran\nbn.slow.%s raised\n\n%s'
                    msg = fmt2 % (name, name, traceback.format_exc())
                    err_msg += msg
                    ok_(False, err_msg)
                assert_array_almost_equal(actual, desired, decimal, err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if hasattr(actual, 'dtype') and hasattr(desired, 'dtype'):
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


# ---------------------------------------------------------------------------
# Manual loop unrolling in bottleneck increase the number of possible code
# paths. Let's test the paths.

def test_loop_unrolling(dtypes=DTYPES):
    "test loop unrolling"

    fmt = '\nfunc %s | dtype %s | shape %s | axis %s\n\nInput array:\n%s\n'
    aaae = assert_array_almost_equal
    funcs = [bn.nansum2]

    # 1d input
    rs = np.random.RandomState([1, 2, 3])
    for func in funcs:
        func0 = eval('bn.slow.%s' % func.__name__)
        for size in range(32):
            arr = np.arange(size)
            rs.shuffle(arr)
            for dtype in dtypes:
                a = arr.astype(dtype)
                actual = func(a)
                desired = func0(a)
                err_msg = fmt % (func.__name__, str(a.dtype),
                                 str(a.shape), str(None), a)
                aaae(actual, desired, decimal=5, err_msg=err_msg)

    # 2d input
    axes = (0, 1, None)
    rs = np.random.RandomState([1, 2, 3])
    for func in funcs:
        func0 = eval('bn.slow.%s' % func.__name__)
        for width in range(2, 32):
            arr = np.arange(width * width)
            rs.shuffle(arr)
            arr = arr.reshape(width, width)
            for dtype in dtypes:
                a = arr.astype(dtype)
                for axis in axes:
                    actual = func(a, axis)
                    desired = func0(a, axis)
                    err_msg = fmt % (func.__name__, str(a.dtype),
                                     str(a.shape), str(axis), a)
                    aaae(actual, desired, decimal=5, err_msg=err_msg)


# ---------------------------------------------------------------------------
# Check that exceptions are raised

def test_nanmax_size_zero(dtypes=DTYPES):
    "Test nanmax for size zero input arrays."
    shapes = [(0,), (2, 0), (1, 2, 0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmax, a)
            assert_raises(ValueError, bn.slow.nanmax, a)


def test_nanmin_size_zero(dtypes=DTYPES):
    "Test nanmin for size zero input arrays."
    shapes = [(0,), (2, 0), (1, 2, 0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmin, a)
            assert_raises(ValueError, bn.slow.nanmin, a)


# ---------------------------------------------------------------------------
# nanstd and nanvar regression test (issue #60)

def test_nanstd_issue60():
    "nanstd regression test (issue #60)"

    f = bn.nanstd([1.0], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanstd([1.0], ddof=1)
    assert_equal(f, s, err_msg="bn.nanstd([1.0], ddof=1) wrong")

    f = bn.nanstd([1], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanstd([1], ddof=1)
    assert_equal(f, s, err_msg="bn.nanstd([1], ddof=1) wrong")

    f = bn.nanstd([1, np.nan], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanstd([1, np.nan], ddof=1)
    assert_equal(f, s, err_msg="bn.nanstd([1, nan], ddof=1) wrong")

    f = bn.nanstd([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanstd([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
    assert_equal(f, s, err_msg="issue #60 regression")


def test_nanvar_issue60():
    "nanvar regression test (issue #60)"

    f = bn.nanvar([1.0], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanvar([1.0], ddof=1)
    assert_equal(f, s, err_msg="bn.nanvar([1.0], ddof=1) wrong")

    f = bn.nanvar([1], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanvar([1], ddof=1)
    assert_equal(f, s, err_msg="bn.nanvar([1], ddof=1) wrong")

    f = bn.nanvar([1, np.nan], ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanvar([1, np.nan], ddof=1)
    assert_equal(f, s, err_msg="bn.nanvar([1, nan], ddof=1) wrong")

    f = bn.nanvar([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
    with np.errstate(invalid='ignore'):
        s = bn.slow.nanvar([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
    assert_equal(f, s, err_msg="issue #60 regression")
