"Test functions."

import warnings

import numpy as np
from numpy.testing import assert_equal
import bottleneck as bn  # noqa
from .functions import all_functions

DTYPES = [np.float64, np.float32, np.int64, np.int32]
nan = np.nan


def arrays(dtypes):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    for seed in (1, 2):
        rs = np.random.RandomState(seed)
        for ndim in ss:
            size = ss[ndim]['size']
            shapes = ss[ndim]['shapes']
            for dtype in dtypes:
                a = np.arange(size, dtype=dtype)
                if issubclass(a.dtype.type, np.inexact):
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = np.inf
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] = np.nan
                    idx = rs.rand(*a.shape) < 0.2
                    a[idx] *= -1
                for shape in shapes:
                    a = a.reshape(shape)
                    yield a


def unit_maker(func, nans=True):
    "Test that bn.xxx gives the same output as np.xxx."
    msg = "\nInput array modifed by %s.\n\n"
    msg += "input array before:\n%s\nafter:\n%s\n"
    for i, arr in enumerate(arrays(DTYPES)):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid='ignore'):
                arr1 = arr.copy()
                arr2 = arr.copy()
                if ('move_' in func.__name__) or ('sort' in func.__name__):
                    if axis is None:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        func(arr1, 1, axis=axis)
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            func(arr1, axis=axis)
                    except:
                        continue
                assert_equal(arr1, arr2, msg % (func.__name__, arr1, arr2))


def test_modification():
    "Test for illegal inplace modification of input array"
    for func in all_functions():
        yield unit_maker, func
