"Test reduce functions."

import warnings

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal, assert_raises,
                           assert_array_almost_equal)
nan = np.nan
import bottleneck as bn
from .functions import reduce_functions

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def arrays(dtypes=DTYPES, nans=True):
    "Iterator that yields arrays to use for unit testing."
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
                yield -a
                # nanargmax/nanargmin regression tests
                yield np.zeros_like(a)
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
    yield np.array([1, 2, 3], dtype='>f4')
    yield np.array([1, 2, 3], dtype='<f4')
    yield np.array([1, 2, 3], dtype=np.float16)  # make sure slow is called
    if nans:
        # nanmedian regression tests
        a = np.array([1, nan, nan, 2])
        yield a
        a = np.vstack((a, a))
        yield a
        yield a.reshape(1, 2, 4)
    # check 0d input
    yield np.array(-9)
    yield np.array(0)
    yield np.array(9)
    yield np.array(-9.0)
    yield np.array(0.0)
    yield np.array(9.0)
    yield np.array(-np.inf)
    yield np.array(np.inf)
    yield np.array(np.nan)


def unit_maker(func, func0, decimal=6, nans=True, check_dtype=True):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid='ignore'):
                actual = 'Crashed'
                desired = 'Crashed'
                actualraised = False
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
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
                tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), str(axis), arr)
                err_msg = msg % tup
                if (decimal < np.inf) and (np.isfinite(arr).sum() > 0):
                    assert_array_almost_equal(actual, desired, decimal,
                                              err_msg)
                else:
                    assert_array_equal(actual, desired, err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if check_dtype:
                    if hasattr(actual, 'dtype') and hasattr(desired, 'dtype'):
                        da = actual.dtype
                        dd = desired.dtype
                        assert_equal(da, dd, err_msg % (da, dd))


def test_reduce():
    "test reduce functions"
    for func in reduce_functions():
        yield unit_maker, func, eval('bn.slow.%s' % func.__name__)


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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

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
