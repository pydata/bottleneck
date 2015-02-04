"Test partsort and argpartsort."

import warnings

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
nan = np.nan
import bottleneck as bn

DTYPES = [np.float64, np.float32, np.int64, np.int32, np.float16]


# ---------------------------------------------------------------------------
# partsort, argpartsort

def arrays(dtypes=DTYPES):
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
                yield -a
            for i in range(0, a.size, 2):
                a.flat[i] *= -1
                yield a


def unit_maker(func, func0):
    "Test bn.(arg)partsort gives same output as bn.slow.(arg)partsort."
    msg = '\nfunc %s | input %s (%s) | shape %s | n %d | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays()):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            if axis is None:
                n = arr.size
            else:
                n = arr.shape[axis]
            n = max(n // 2, 1)
            with np.errstate(invalid='ignore'):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    actual = func(arr.copy(), n, axis=axis)
                actual[:n] = np.sort(actual[:n], axis=axis)
                actual[n:] = np.sort(actual[n:], axis=axis)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    desired = func0(arr.copy(), n, axis=axis)
                    desired[:n] = np.sort(desired[:n], axis=axis)
                    desired[n:] = np.sort(desired[n:], axis=axis)
            tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                   str(arr.shape), n, str(axis), arr)
            err_msg = msg % tup
            assert_array_equal(actual, desired, err_msg)
            err_msg += '\n dtype mismatch %s %s'
            if hasattr(actual, 'dtype') and hasattr(desired, 'dtype'):
                da = actual.dtype
                dd = desired.dtype
                assert_equal(da, dd, err_msg % (da, dd))


def test_partsort():
    "Test partsort."
    yield unit_maker, bn.partsort, bn.slow.partsort


def test_argpartsort():
    "Test argpartsort."
    yield unit_maker, bn.argpartsort, bn.slow.argpartsort


def test_transpose():
    "partsort transpose test"
    a = np.arange(12).reshape(4, 3)
    actual = bn.partsort(a.T, 2, -1).T
    desired = bn.slow.partsort(a.T, 2, -1).T
    assert_equal(actual, desired, 'partsort transpose test')


# ---------------------------------------------------------------------------
# rankdata, nanrankdata

from .reduce_test import unit_maker as reduce_unit_maker

def test_rankdata():
    "Test rankdata."
    yield reduce_unit_maker, bn.rankdata, bn.slow.rankdata


def test_nanrankdata():
    "Test nanrankdata."
    yield reduce_unit_maker, bn.nanrankdata, bn.slow.nanrankdata
