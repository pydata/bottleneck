"Test reduce functions."

import warnings
import traceback

from nose.tools import ok_
import numpy as np
from numpy.testing import (assert_equal, assert_raises,
                           assert_array_almost_equal)

import bottleneck as bn

DTYPES = [np.float64, np.float32, np.int64, np.int32]


def test_reduce():
    "test reduce functions"
    for func in bn.get_functions('reduce'):
        yield unit_maker, func


def arrays(dtypes, name):
    "Iterator that yields arrays to use for unit testing."

    yield np.array([7, 5, 1, 6, 0, 2, 4, 3])
    """
    # nan and inf
    nan = np.nan
    inf = np.inf
    yield np.array([inf, nan])
    yield np.array([inf, -inf])
    # yield np.array([nan, inf])  np.nanargmin can't handle this one

    # byte swapped
    yield np.array([1, 2, 3], dtype='>f4')
    yield np.array([1, 2, 3], dtype='<f4')

    # make sure slow is called
    yield np.array([1, 2, 3], dtype=np.float16)

    # regression tests
    yield np.array([0, 0, 0])  # nanargmax/nanargmin
    yield np.array([1, nan, nan, 2])  # nanmedian

    # ties
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

    # Automate a bunch of arrays to test
    ss = {}
    ss[0] = {'size':  0, 'shapes': [(0,), (0, 0), (2, 0), (2, 0, 1)]}
    ss[1] = {'size':  8, 'shapes': [(8,)]}
    ss[2] = {'size': 12, 'shapes': [(2, 6), (3, 4)]}
    ss[3] = {'size': 16, 'shapes': [(2, 2, 4)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for ndim in ss:
            size = ss[ndim]['size']
            shapes = ss[ndim]['shapes']
            for dtype in dtypes:
                a = np.arange(size, dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    if name not in ('nanargmin', 'nanargmax'):
                        # numpy can't handle eg np.nanargmin([np.nan, np.inf])
                        idx = rs.rand(*a.shape) < 0.2
                        a[idx] = inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                rs.shuffle(a)
                for shape in shapes:
                    yield a.reshape(shape)
    """


def unit_maker(func, decimal=5):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    fmt = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    fmt += '\nInput array:\n%s\n'
    name = func.__name__
    func0 = eval('bn.slow.%s' % name)
    for i, a in enumerate(arrays(DTYPES, name)):
        axes = (0,)
        for axis in axes:
            actual = 'Crashed'
            desired = 'Crashed'
            actualraised = False
            try:
                # do not use a.copy() here because it will C order the array
                actual = func(a, axis=axis)
            except:
                actualraised = True
            desiredraised = False
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    desired = func0(a, axis=axis)
            except:
                desiredraised = True
            if actualraised and desiredraised:
                pass
            else:
                tup = (name, 'a'+str(i), str(a.dtype), str(a.shape),
                       str(axis), a)
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
