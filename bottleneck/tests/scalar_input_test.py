"Test scalar input."

from numpy.testing import assert_array_almost_equal
import bottleneck as bn


# ---------------------------------------------------------------------------
# Check that functions can handle scalar input


def unit_maker(func, func0, args=tuple()):
    "Test that bn.xxx gives the same output as bn.slow.xxx for scalar input."
    msg = '\nfunc %s | input %s\n'
    a = -9
    argsi = [a] + list(args)
    actual = func(*argsi)
    desired = func0(*argsi)
    err_msg = msg % (func.__name__, a)
    assert_array_almost_equal(actual, desired, err_msg=err_msg)


def test_nansum():
    "Test nansum."
    yield unit_maker, bn.nansum, bn.slow.nansum


def test_nanmean():
    "Test nanmean."
    yield unit_maker, bn.nanmean, bn.slow.nanmean


def test_nanstd():
    "Test nanstd."
    yield unit_maker, bn.nanstd, bn.slow.nanstd


def test_nanvar():
    "Test nanvar."
    yield unit_maker, bn.nanvar, bn.slow.nanvar


def test_nanmin():
    "Test nanmin."
    yield unit_maker, bn.nanmin, bn.slow.nanmin


def test_nanmax():
    "Test nanmax."
    yield unit_maker, bn.nanmax, bn.slow.nanmax


def test_median():
    "Test median."
    yield unit_maker, bn.median, bn.slow.median


def test_nanmedian():
    "Test nanmedian."
    yield unit_maker, bn.nanmedian, bn.slow.nanmedian


def test_ss():
    "Test ss."
    yield unit_maker, bn.ss, bn.slow.ss


def test_nanargmin():
    "Test nanargmin."
    yield unit_maker, bn.nanargmin, bn.slow.nanargmin


def test_nanargmax():
    "Test nanargmax."
    yield unit_maker, bn.nanargmax, bn.slow.nanargmax


def test_anynan():
    "Test anynan."
    yield unit_maker, bn.anynan, bn.slow.anynan


def test_allnan():
    "Test allnan."
    yield unit_maker, bn.allnan, bn.slow.allnan


def test_rankdata():
    "Test rankdata."
    yield unit_maker, bn.rankdata, bn.slow.rankdata


def test_nanrankdata():
    "Test nanrankdata."
    yield unit_maker, bn.nanrankdata, bn.slow.nanrankdata
