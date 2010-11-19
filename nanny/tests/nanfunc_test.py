"Test nan functions."

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
nan = np.nan
from nanny import nansum


def mov_unit_maker(func, func0):
    "Test that different mov methods give the same results on 2d input."
    arr0 = np.array([1, 2, 3, 4, 5, 6, nan, nan, 7, 8, 9])
    arr1 = np.arange(10)
    arr2 = np.array([[9.0, 3.0, nan, nan, 9.0, nan],
                      [1.0, 1.0, 1.0, nan, nan, nan],
                      [2.0, 2.0, 0.1, nan, 1.0, nan],
                      [3.0, 9.0, 2.0, nan, nan, nan],
                      [4.0, 4.0, 3.0, 9.0, 2.0, nan],
                      [5.0, 5.0, 4.0, 4.0, nan, nan]])
    arr3 = np.arange(12).reshape(3,4)
    arr4 = np.random.rand(3,4)
    arr5 = np.arange(60).reshape(3, 4, 5)
    arr6 = np.random.rand(3, 4, 5)
    arr7 = np.array([nan, nan, nan])
    arr8 = np.array([1, np.nan, np.inf, np.NINF])
    arrs = [arr0, arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8]
    msg = '\nfunc %s | ndim %d | axis %s\n'
    for arr in arrs:
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            actual = func(arr, axis=axis)
            desired = func0(arr, axis=axis)
            err_msg = msg % (func.__name__, arr.ndim, str(axis))
            assert_array_equal(actual, desired, err_msg)
            err_msg += '\n dtype mismatch %s %s'
            if hasattr(actual, 'dtype') and hasattr(desired, 'dtype'):
                da = actual.dtype
                dd = desired.dtype
                assert_equal(da, dd, err_msg % (da, dd))

def test_nansum():
    "Test nansum."
    yield mov_unit_maker, nansum, np.nansum 
