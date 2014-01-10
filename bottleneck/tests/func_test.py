"Test functions."

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal, assert_raises,
                           assert_array_almost_equal, assert_almost_equal)
nan = np.nan
import bottleneck as bn


def arrays(dtypes=bn.dtypes, nans=True):
    "Iterator that yields arrays to use for unit testing."
    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0, 0), (2, 0), (2, 0, 1)]}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1, 6), (2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}  # Unaccelerated
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
    if nans:
        # nanmedian regression tests
        a = np.array([1, nan, nan, 2])
        yield a
        a = np.vstack((a, a))
        yield a
        yield a.reshape(1, 2, 4)


def unit_maker(func, func0, decimal=np.inf, nans=True):
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
                    actual = func(arr.copy(), axis=axis)
                except:
                    actualraised = True
                desiredraised = False
                try:
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
                if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                    da = actual.dtype
                    dd = desired.dtype
                    assert_equal(da, dd, err_msg % (da, dd))


def test_nansum():
    "Test nansum."
    yield unit_maker, bn.nansum, bn.slow.nansum


def test_nanmax():
    "Test nanmax."
    yield unit_maker, bn.nanmax, bn.slow.nanmax


def test_nanargmin():
    "Test nanargmin."
    yield unit_maker, bn.nanargmin, bn.slow.nanargmin


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
    yield unit_maker, bn.nanmedian, bn.slow.nanmedian


def test_rankdata():
    "Test rankdata."
    yield unit_maker, bn.rankdata, bn.slow.rankdata


def test_nanrankdata():
    "Test nanrankdata."
    yield unit_maker, bn.nanrankdata, bn.slow.nanrankdata


def test_ss():
    "Test ss."
    yield unit_maker, bn.ss, bn.slow.ss


def test_anynan():
    "Test anynan."
    yield unit_maker, bn.anynan, bn.slow.anynan


def test_allnan():
    "Test allnan."
    yield unit_maker, bn.allnan, bn.slow.allnan

# ---------------------------------------------------------------------------
# Check that exceptions are raised


def test_nanmax_size_zero(dtypes=bn.dtypes):
    "Test nanmax for size zero input arrays."
    shapes = [(0,), (2, 0), (1, 2, 0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmax, a)
            assert_raises(ValueError, bn.slow.nanmax, a)


def test_nanmin_size_zero(dtypes=bn.dtypes):
    "Test nanmin for size zero input arrays."
    shapes = [(0,), (2, 0), (1, 2, 0)]
    for shape in shapes:
        for dtype in dtypes:
            a = np.zeros(shape, dtype=dtype)
            assert_raises(ValueError, bn.nanmin, a)
            assert_raises(ValueError, bn.slow.nanmin, a)

# ---------------------------------------------------------------------------
# Check nn


def arrays2(dtypes=bn.dtypes):
    "Iterator that yield arrays to use for unit testing."
    for dtype in bn.dtypes:
        arr = np.random.randint(0, 10, (2, 3)).astype(dtype)
        arr0 = np.random.randint(0, 10, 3).astype(dtype)
        axis = 0
        yield arr.copy(), arr0[:2].copy(), axis
        axis = 1
        yield arr.copy(), arr0.copy(), axis
        # Make sure ties are handled in the same way
        arr.fill(0)
        arr0.fill(0)
        axis = 0
        yield arr.copy(), arr0[:2].copy(), axis
        axis = 1
        yield arr.copy(), arr0.copy(), axis
        if issubclass(arr.dtype.type, np.inexact):
            # Make sure NaNs are handled in the same way
            arr.fill(np.nan)
            arr0.fill(np.nan)
            axis = 0
            yield arr.copy(), arr0[:2].copy(), axis
            axis = 1
            yield arr.copy(), arr0.copy(), axis


def test_nn():
    "Test that bn.nn gives the same output as bn.slow.nn."
    msg = '\nfunc %s | input %s (%s) | axis %s\n'
    msg += '\nInput array `arr`:\n%s\n'
    msg += '\nInput array `arr0`:\n%s\n'
    count = -1
    for arr, arr0, axis in arrays2():
        actual = bn.nn(arr, arr0, axis)
        desired = bn.slow.nn(arr, arr0, axis)
        count += 1
        tup = ('bn.nn', 'arr'+str(count), str(arr.dtype),
               str(axis), arr, arr0)
        err_msg = msg % tup
        assert_almost_equal(actual, desired, decimal=5, err_msg=err_msg)

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

    b = bn.nanstd([1, np.nan], ddof=1)
    with np.errstate(invalid='ignore'):
        b = bn.slow.nanstd([1, np.nan], ddof=1)
    assert_equal(f, s, err_msg="bn.nanstd([1, nan], ddof=1) wrong")

    b = bn.nanstd([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
    with np.errstate(invalid='ignore'):
        b = bn.slow.nanstd([[1, np.nan], [np.nan, 1]], axis=0, ddof=1)
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
