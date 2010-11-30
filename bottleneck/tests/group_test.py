"Test group functions."

import numpy as np
from numpy.testing import (assert_equal, assert_array_equal, assert_raises,
                           assert_array_almost_equal)
nan = np.nan
from scipy.stats import nanmean
import bottleneck as bn
from bottleneck.testing.group_validator import group_func


def array_iter(dtypes=['float64']):
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
                yield -a
            if issubclass(a.dtype.type, np.inexact):        
                for i in range(a.size):
                    a.flat[i] = np.nan
                    yield a
                    yield -a

def label_iter(n):
    "Iterator that yielbn a variety of labels of given length"
    dtypes = ['int32', 'int64', 'float64', 'str']
    for dtype in dtypes:
        label0 = np.ones(n, dtype=dtype)
        label1 = np.zeros(n, dtype=dtype)
        label = label0.copy()
        yield label
        if dtype != 'int64':
            yield label.tolist()
        if all(label != label[::-1]):
            yield label[::-1]
        if n > 2:
            label = label0.copy()
            label[1] = label1[1]
        yield label
        if all(label != label[::-1]):
            yield label[::-1]
        if dtype != 'int64':
            yield label.tolist()
            if all(label != label[::-1]):
                yield label[::-1].tolist()

def unit_maker(func, func0, decimal=np.inf):
    "Test that bn.xxx gives the same output as a reference function."
    msg = "\nfunc %s | input %s (%s) | shape %s | axis %s\n"
    msg += "\nInput array:\n%s\n"
    msg += "\nLabel (%s):\n%s\n"
    for i, arr in enumerate(array_iter()):
        for axis in range(-arr.ndim, arr.ndim):
            for label in label_iter(arr.shape[axis]):
                with np.errstate(invalid='ignore'):
                    a1, lab1 = func(arr, label, axis=axis)
                    a0, lab0 = group_func(func0, arr, label, axis=axis)
                if type(label) == np.ndarray:
                    labeltype = 'array'
                elif type(label) == list:
                    labeltype = 'list'
                else:
                    raise ValueError("label must be an array or list")
                tup = (func.__name__, 'a'+str(i), str(arr.dtype),
                       str(arr.shape), str(axis), arr, labeltype, label)
                err_msg = msg % tup
                if (decimal < np.inf) and (np.isfinite(arr).sum() > 0):
                    assert_array_almost_equal(a1, a0, decimal, err_msg)
                else:
                    assert_array_equal(a1, a0, err_msg)
                err_msg += '\n dtype mismatch %s %s'
                if hasattr(a1, 'dtype') or hasattr(a0, 'dtype'):
                    da = a1.dtype
                    dd = a0.dtype
                    assert_equal(da, dd, err_msg % (da, dd))
                assert_equal(lab1, lab0, err_msg="labels not equal")

def test_group_nanmean():
    "Test group_nanmean."
    yield unit_maker, bn.group_nanmean, nanmean, 13
