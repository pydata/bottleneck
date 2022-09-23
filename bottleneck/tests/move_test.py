"""Test moving window functions."""

import numpy as np
from numpy.testing import assert_equal, assert_array_almost_equal, assert_raises
import bottleneck as bn
from .util import arrays, array_order
import pytest
import itertools
import warnings


@pytest.mark.parametrize("func", bn.get_functions("move"), ids=lambda x: x.__name__)
def test_move(func):
    """Test that bn.xxx gives the same output as a reference function."""
    fmt = (
        "\nfunc %s | window %d | min_count %s | input %s (%s) | shape %s | "
        "axis %s | order %s\n"
    )
    fmt += "\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    func_name = func.__name__
    func0 = eval("bn.slow.%s" % func_name)
    if func_name == "move_var":
        decimal = 3
    else:
        decimal = 5
    quantiles = [1.]
    if func_name == "move_quantile":
        quantiles = [0.33, 0.67]
    for i, a in enumerate(arrays(func_name)):
        axes = range(-1, a.ndim)
        for axis in axes:
            windows = range(1, a.shape[axis])
            for window in windows:
                min_counts = list(range(1, window + 1)) + [None]
                for min_count in min_counts:
                    for q in quantiles:
                        kwargs = {}
                        if func_name == "move_quantile":
                            kwargs = {"q" : q}
                        actual = func(a, window, min_count, axis=axis, **kwargs)
                        desired = func0(a, window, min_count, axis=axis, **kwargs)
                        tup = (
                            func_name,
                            window,
                            str(min_count),
                            "a" + str(i),
                            str(a.dtype),
                            str(a.shape),
                            str(axis),
                            array_order(a),
                            a,
                        )
                        err_msg = fmt % tup
                        aaae(actual, desired, decimal, err_msg)
                        err_msg += "\n dtype mismatch %s %s"
                        da = actual.dtype
                        dd = desired.dtype
                        assert_equal(da, dd, err_msg % (da, dd))


# ---------------------------------------------------------------------------
# Test argument parsing


@pytest.mark.parametrize("func", bn.get_functions("move"), ids=lambda x: x.__name__)
def test_arg_parsing(func, decimal=5):
    """test argument parsing."""

    name = func.__name__
    func0 = eval("bn.slow.%s" % name)

    a = np.array([1.0, 2, 3])

    fmt = "\n%s" % func
    fmt += "%s\n"
    fmt += "\nInput array:\n%s\n" % a

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

    actual = func(a=a, axis=-1, min_count=None, window=2)
    desired = func0(a=a, axis=-1, min_count=None, window=2)
    err_msg = fmt % "(a=a, axis=-1, min_count=None, window=2)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    if name in ("move_std", "move_var"):
        actual = func(a, 2, 1, -1, ddof=1)
        desired = func0(a, 2, 1, -1, ddof=1)
        err_msg = fmt % "(a, 2, 1, -1, ddof=1)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

    if name == "move_quantile":
        q = 0.3
        actual = func(q=q, axis=-1, a=a, min_count=None, window=2)
        desired = func0(q=q, axis=-1, a=a, min_count=None, window=2)
        err_msg = fmt % "(q=q, axis=-1, a=a, min_count=None, window=2)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

        actual = func(a, axis=-1, q=q, window=2, min_count=None)
        desired = func0(a, axis=-1, q=q, window=2, min_count=None)
        err_msg = fmt % "(a, axis=-1, q=q, window=2, min_count=None)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

        actual = func(axis=-1, a=a, q=q, window=2)
        desired = func0(axis=-1, a=a, q=q, window=2)
        err_msg = fmt % "(axis=-1, a=a, q=q, window=2)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

    # regression test: make sure len(kwargs) == 0 doesn't raise
    args = (a, 1, 1, -1)
    kwargs = {}
    func(*args, **kwargs)


@pytest.mark.parametrize("func", bn.get_functions("move"), ids=lambda x: x.__name__)
def test_arg_parse_raises(func):
    """test argument parsing raises in move"""
    a = np.array([1.0, 2, 3])
    assert_raises(TypeError, func)
    assert_raises(TypeError, func, axis=a)
    assert_raises(TypeError, func, a, 2, axis=0, extra=0)
    assert_raises(TypeError, func, a, 2, axis=0, a=a)
    assert_raises(TypeError, func, a, 2, 2, 0, 0, 0)
    assert_raises(TypeError, func, a, 2, axis="0")
    assert_raises(TypeError, func, a, 1, min_count="1")
    if func.__name__ not in ("move_std", "move_var"):
        assert_raises(TypeError, func, a, 2, ddof=0)


# ---------------------------------------------------------------------------
# move_median.c is complicated. Let's do some more testing.
#
# If you make changes to move_median.c then do lots of tests by increasing
# range(100) in the two functions below to range(10000). And for extra credit
# increase size to 30. With those two changes the unit tests will take a
# LONG time to run.

REPEAT_MEDIAN = 10

def test_move_median_with_nans():
    """test move_median.c with nans"""
    fmt = "\nfunc %s | window %d | min_count %s\n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    min_count = 1
    func = bn.move_median
    func0 = bn.slow.move_median
    rs = np.random.RandomState([1, 2, 3])
    for size in [1, 2, 3, 4, 5, 9, 10, 19, 20, 21]:
        for _ in range(REPEAT_MEDIAN):
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
    """test move_median.c without nans"""
    fmt = "\nfunc %s | window %d | min_count %s\n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    min_count = 1
    func = bn.move_median
    func0 = bn.slow.move_median
    rs = np.random.RandomState([1, 2, 3])
    for size in [1, 2, 3, 4, 5, 9, 10, 19, 20, 21]:
        for _ in range(REPEAT_MEDIAN):
            a = np.arange(size, dtype=np.int64)
            rs.shuffle(a)
            for window in range(2, size + 1):
                actual = func(a, window=window, min_count=min_count)
                desired = func0(a, window=window, min_count=min_count)
                err_msg = fmt % (func.__name__, window, min_count, a)
                aaae(actual, desired, decimal=5, err_msg=err_msg)


# ---------------------------------------------------------------------------
# move_quantile is newly added. So let's do (very) extensive testing
#
# Unfortunately, np.nanmedian(a) and np.nanquantile(a, q=0.5) don't always agree
# when a contains inf or -inf values. See for instance:
# https://github.com/numpy/numpy/issues/21932
# https://github.com/numpy/numpy/issues/21091
#
# Let's first test without inf and -inf. 
# When there are no infs in data, bn.slow.move_quantile calls 
# move_func for np_nanquantile_infs, which just runs 
# np.nanquantile with interpolation="midpoint"

REPEAT_QUANTILE = 10

def test_move_quantile_with_nans():
    """test move_quantile.c with nans"""
    fmt = "\nfunc %s | window %d | min_count %s | q %f\n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    min_count = 1
    size = 10
    func = bn.move_quantile
    func0 = bn.slow.move_quantile
    rs = np.random.RandomState([1, 2, 3])
    for size in [1, 2, 3, 5, 9, 10, 13, 16]:
        for _ in range(REPEAT_QUANTILE):
            for nan_frac in [0.2, 0.5, 0.7, 1.]:
                # test more variants of arrays (ints, floats, diff values)
                arrays = [np.arange(size, dtype=np.float64),
                        (rs.random(size) - 0.5) * rs.randint(0, 100_000),
                        (rs.random(size) - 0.5) / rs.randint(0, 100_000)]
                for a in arrays:
                    # q = 0. and 1. are important edge cases. We call existing 
                    # move_min and move_max for these, but still must check 
                    # that it works properly
                    for q in [0., 1., rs.rand()]:
                        idx = rs.rand(*a.shape) < nan_frac
                        a[idx] = np.nan
                        rs.shuffle(a) 
                        for window in range(2, size + 1):
                            actual = func(a, window=window, min_count=min_count, q=q)
                            desired = func0(a, window=window, min_count=min_count, q=q)
                            err_msg = fmt % (func.__name__, window, min_count, q, a)
                            aaae(actual, desired, decimal=5, err_msg=err_msg)

def test_move_quantile_without_nans():
    """test move_quantile.c without nans"""
    fmt = "\nfunc %s | window %d | min_count %s | q %f\n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    min_count = 1
    size = 10
    func = bn.move_quantile
    func0 = bn.slow.move_quantile
    rs = np.random.RandomState([1, 2, 3])
    for size in [1, 2, 3, 5, 9, 10, 13, 16]:        
        for _ in range(REPEAT_QUANTILE):
            # test more variants of arrays (ints, floats, diff values)
            arrays = [np.arange(size, dtype=np.float64),
                    (rs.random(size) - 0.5) * rs.randint(0, 100_000),
                    (rs.random(size) - 0.5) / rs.randint(0, 100_000)]
            for a in arrays:
                # q = 0. and 1. are important edge cases. We call existing 
                # move_min and move_max for these, but still must check 
                # that it works properly
                for q in [0., 1., rs.rand()]:
                    rs.shuffle(a) 
                    for window in range(2, size + 1):
                        actual = func(a, window=window, min_count=min_count, q=q)
                        desired = func0(a, window=window, min_count=min_count, q=q)
                        err_msg = fmt % (func.__name__, window, min_count, q, a)
                        aaae(actual, desired, decimal=5, err_msg=err_msg)


# Now let's deal with inf ans -infs
# np.nanquantile doesn't give desired results when infs are present,
# due to abmiguities with arithmetic operations with infs.
# For instance, 
# np.nanquantile([np.inf, np.inf], q=0.5, method="midpoint") returns np.nan, 
# while
# np.nanmedian([np.inf, np.inf]) returns np.inf,
# although arguably these should give the same result. The behaviour of 
# np.nanmedian is also agruably more expected.
# We check that the following will always give the same result as np.nanmedian(a):
# (np.nanquantile(a, q=0.5, method="lower") + np.nanquantile(a, q=0.5, method="higher")) / 2
# It is also clear that this essentially is the same as method="midpoint".
# The next test verifies this. 

from ..slow.move import np_nanquantile_infs

REPEAT_NUMPY_QUANTILE = 10

def test_numpy_nanquantile_infs():
    """test move_quantile.c with nans"""
    fmt = "\nfunc %s \n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    min_count = 1
    func = np.nanmedian
    func0 = np_nanquantile_infs
    rs = np.random.RandomState([1, 2, 3])
    sizes = [1, 2, 3, 4, 5, 9, 10, 20, 31]
    fracs = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    inf_minf_nan_fracs = [triple for triple in itertools.product(fracs, fracs, fracs) if np.sum(triple) <= 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for size in sizes:
            for _ in range(REPEAT_NUMPY_QUANTILE):            
                for (inf_frac, minus_inf_frac, nan_frac) in inf_minf_nan_fracs:
                    arrays = [np.arange(size, dtype=np.float64),
                            (rs.random(size) - 0.5) * rs.randint(0, 100_000),
                            (rs.random(size) - 0.5) / rs.randint(0, 100_000)]
                    for a in arrays:
                        randoms = rs.rand(*a.shape)
                        idx_nans = randoms < inf_frac + minus_inf_frac + nan_frac
                        a[idx_nans] = np.nan
                        idx_minfs = randoms < inf_frac + minus_inf_frac
                        a[idx_minfs] = -np.inf
                        idx_infs = randoms < inf_frac
                        a[idx_infs] = np.inf
                        rs.shuffle(a)
                        actual = func(a)
                        desired = func0(a, q=0.5)
                        err_msg = fmt % (func.__name__, a)
                        aaae(actual, desired, decimal=5, err_msg=err_msg)

# This shows that np_nanquantile_infs indeed replicates the
# behaviour of np.nanquantile, while correclty handling infs in data.
# So we use np_nanquantile_infs in our bn.slow.move_quantile for testing

REPEAT_FULL_QUANTILE = 1

def test_move_quantile_with_infs_and_nans():
    """test move_quantile.c with infs and nans"""
    fmt = "\nfunc %s | window %d | min_count %s | q %f\n\nInput array:\n%s\n"
    aaae = assert_array_almost_equal
    func = bn.move_quantile
    func0 = bn.slow.move_quantile
    rs = np.random.RandomState([1, 2, 3])
    fracs = [0., 0.2, 0.4, 0.6, 0.8, 1.]
    inf_minf_nan_fracs = [triple for triple in itertools.product(fracs, fracs, fracs) if np.sum(triple) <= 1]
    for size in [1, 2, 3, 5, 9, 10, 17, 20, 31]:
        for min_count in [1, 2, 3, size//2, size - 1, size]:
            if min_count < 1 or min_count > size:
                continue
            for (inf_frac, minus_inf_frac, nan_frac) in inf_minf_nan_fracs:
                for window in range(min_count, size + 1):
                    for _ in range(REPEAT_FULL_QUANTILE):
                        for q in [0., 1., 0.25, 0.75, rs.rand()]:
                            arrays = [np.arange(size, dtype=np.float64),
                                    (rs.random(size) - 0.5) * rs.randint(0, 100_000),
                                    (rs.random(size) - 0.5) / rs.randint(0, 100_000)]
                            for a in arrays:
                                a = np.arange(size, dtype=np.float64)
                                randoms = rs.rand(*a.shape)
                                idx_nans = randoms < inf_frac + minus_inf_frac + nan_frac
                                a[idx_nans] = np.nan
                                idx_minfs = randoms < inf_frac + minus_inf_frac
                                a[idx_minfs] = -np.inf
                                idx_infs = randoms < inf_frac
                                a[idx_infs] = np.inf
                                rs.shuffle(a)
                                actual = func(a, window=window, min_count=min_count, q=q)
                                desired = func0(a, window=window, min_count=min_count, q=q)
                                err_msg = fmt % (func.__name__, window, min_count, q, a)
                                aaae(actual, desired, decimal=5, err_msg=err_msg)



# ----------------------------------------------------------------------------
# Regression test for square roots of negative numbers


def test_move_std_sqrt():
    """Test move_std for neg sqrt."""

    a = [
        0.0011448196318903589,
        0.00028718669878572767,
        0.00028718669878572767,
        0.00028718669878572767,
        0.00028718669878572767,
    ]
    err_msg = "Square root of negative number. ndim = %d"
    b = bn.move_std(a, window=3)
    assert np.isfinite(b[2:]).all(), err_msg % 1

    a2 = np.array([a, a])
    b = bn.move_std(a2, window=3, axis=1)
    assert np.isfinite(b[:, 2:]).all(), err_msg % 2

    a3 = np.array([[a, a], [a, a]])
    b = bn.move_std(a3, window=3, axis=2)
    assert np.isfinite(b[:, :, 2:]).all(), err_msg % 3
