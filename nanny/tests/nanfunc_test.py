"Test nan functions."

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
nan = np.nan
from nanny import nansum, nanmax


def unit_maker(func, func0):
    "Test that ny.nanxxx gives the same output as np.."
    a0 = np.array([1, 2, 3, 4, 5, 6, nan, nan, 7, 8, 9])
    a1 = np.arange(10)
    a2 = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                   [1.0, 1.0, 1.0, nan, nan, nan],
                   [2.0, 2.0, 0.1, nan, 1.0, nan],
                   [3.0, 9.0, 2.0, nan, nan, nan],
                   [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                   [5.0, 5.0, 4.0, 4.0, nan, nan]])
    a3 = np.arange(12).reshape(3,4)
    a4 = np.random.rand(3,4)
    a5 = np.arange(60).reshape(3, 4, 5)
    a6 = np.random.rand(3, 4, 5)
    a7 = np.array([nan, nan, nan])
    a8 = np.array([1, np.nan, np.inf, np.NINF])
    a9 = np.arange(5, dtype=np.int32)
    a10 = np.arange(10, dtype=np.int32).reshape(2, 5)
    a11 = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    arrs = [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11]
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    for i, arr in enumerate(arrs):
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            actual = func(arr, axis=axis)
            desired = func0(arr, axis=axis)
            tup = (func.__name__, 'a'+str(i), str(arr.dtype), str(arr.shape),
                   str(axis))
            err_msg = msg % tup
            assert_array_equal(actual, desired, err_msg)
            err_msg += '\n dtype mismatch %s %s'
            if hasattr(actual, 'dtype') or hasattr(desired, 'dtype'):
                da = actual.dtype
                dd = desired.dtype
                assert_equal(da, dd, err_msg % (da, dd))

def test_nansum():
    "Test nansum."
    yield unit_maker, nansum, np.nansum

def test_nanmax():
    "Test nanmax."
    yield unit_maker, nanmax, np.nanmax
    
