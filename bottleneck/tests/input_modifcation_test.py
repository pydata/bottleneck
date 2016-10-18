"Test functions."

import warnings

import numpy as np
from numpy.testing import assert_equal
import bottleneck as bn
from .util import DTYPES


def arrays(dtypes):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    rs = np.random.RandomState([1, 2, 3])
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
    for i, a in enumerate(arrays(DTYPES)):
        for axis in list(range(-a.ndim, a.ndim)) + [None]:
            with np.errstate(invalid='ignore'):
                a1 = a.copy()
                a2 = a.copy()
                if ('move_' in func.__name__) or ('sort' in func.__name__):
                    if axis is None:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        func(a1, 1, axis=axis)
                else:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            func(a1, axis=axis)
                    except:
                        continue
                assert_equal(a1, a2, msg % (func.__name__, a1, a2))


def test_modification():
    "Test for illegal inplace modification of input array"
    for func in bn.get_functions('all'):
        yield unit_maker, func
