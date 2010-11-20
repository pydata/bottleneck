"Test nan functions."

import numpy as np
from numpy.testing import assert_equal, assert_array_equal
nan = np.nan
from nanny import nansum, nanmax


def arrays(dtypes=['int32', 'int64', 'float64']):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1,6), (2,3), (6,1)]}
    ss[3] = {'size': 24, 'shapes': [(1,1,24), (24,1,1), (1,24,1), (2,3,4)]}
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            for shape in shapes:
                a = a.reshape(shape)
                yield a
            if issubclass(a.dtype.type, np.inexact):        
                for i in range(a.size):
                    a.flat[i] = np.nan
                    yield a
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a
                a[:] = np.nan    
                for i in range(a.size):
                    a.flat[i] = -np.inf
                    yield a

def unit_maker(func, func0):
    "Test that ny.nanxxx gives the same output as np.."
    msg = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(arrays()):
        for axis in range(-arr.ndim, arr.ndim) + [None]:
            actual = func(arr, axis=axis)
            desired = func0(arr, axis=axis)
            tup = (func.__name__, 'a'+str(i), str(arr.dtype), str(arr.shape),
                   str(axis), arr)
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
    
