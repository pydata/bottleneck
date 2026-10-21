"""Precision regressions for the float32 variance reductions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

import bottleneck as bn


@pytest.mark.parametrize("func", [bn.nanstd, bn.nanvar])
@pytest.mark.parametrize("size", [100, 150_000])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("layout", ["flat", "C", "F", "strided", "reversed"])
@pytest.mark.parametrize("with_nan", [False, True])
def test_float32_variance_constant(func, size, ddof, layout, with_nan):
    # gh-443: accumulating the mean in float32 made constants appear variable.
    a = np.full((size, 2), 271.46, dtype=np.float32)
    if with_nan:
        a[::7] = np.nan
    if layout == "flat":
        a = a[:, 0].copy()
    elif layout == "F":
        a = np.asfortranarray(a)
    elif layout == "strided":
        a = a[::2]
    elif layout == "reversed":
        a = a[::-1]
    before = a.copy()
    for axis in [None, 0, -1]:
        actual = func(a, axis=axis, ddof=ddof)
        # Reducing an all-NaN row produces NaN, not zero.
        expected = np.where(np.all(np.isnan(a), axis=axis), np.nan, 0.0)
        assert_equal(actual, expected)
        if isinstance(actual, np.ndarray):
            assert_equal(actual.dtype, np.dtype(np.float32))
        else:
            assert isinstance(actual, float)
    assert_equal(a, before)


@pytest.mark.parametrize("func", [bn.nanstd, bn.nanvar])
@pytest.mark.parametrize("ddof", [0, 1])
@pytest.mark.parametrize("scale", [2.0**-100, 1.0, 2.0**100])
@pytest.mark.parametrize("offset", [0, 2**20])
def test_float32_variance_nonconstant(func, ddof, scale, offset):
    # Values and their small spread are exact in float32. Squaring deviations
    # in float32 would overflow or underflow at the extreme scales.
    values = (np.array([1, 3, 5, 7], dtype=np.float32) + offset) * scale
    a = np.tile(values, (3, 1))
    expected = 20.0 * scale**2 / (4 - ddof)
    if func is bn.nanstd:
        expected = np.sqrt(expected)
    with np.errstate(over="ignore"):
        expected = np.float32(expected)
    actual = func(a, axis=1, ddof=ddof)
    assert_equal(actual.dtype, np.dtype(np.float32))
    assert_allclose(actual, expected, rtol=2 * np.finfo(np.float32).eps, atol=0)
    assert_allclose(func(values, ddof=ddof), expected, rtol=2e-7, atol=0)


@pytest.mark.parametrize("axis", [None, -1])
def test_float32_variance_large_constant(axis):
    # A double accumulator alone eventually loses bits even for constants.
    # A zero-stride view exercises a large count without a large allocation.
    value = np.nextafter(np.float32(2), np.float32(0))
    a = np.broadcast_to(value, (1, 2**29 + 33))
    assert_equal(bn.nanstd(a, axis=axis), 0)
