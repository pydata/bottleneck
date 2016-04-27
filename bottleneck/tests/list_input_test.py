"Check that functions can handle list input"

import warnings

import numpy as np
from numpy.testing import assert_array_almost_equal
import bottleneck as bn  # noqa
from .functions import all_functions


def lists():
    "Iterator that yields lists to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(1, 6), (2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    ss[4] = {'size': 24, 'shapes': [(1, 2, 3, 4)]}  # Unaccelerated
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        a = np.arange(size)
        for shape in shapes:
            a = a.reshape(shape)
            yield a.tolist()


def unit_maker(func, func0):
    "Test that bn.xxx gives the same output as bn.slow.xxx for list input."
    msg = '\nfunc %s | input %s | shape %s\n'
    msg += '\nInput array:\n%s\n'
    for i, arr in enumerate(lists()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                actual = func(arr)
                desired = func0(arr)
            except TypeError:
                actual = func(arr, 2)
                desired = func0(arr, 2)
        tup = (func.__name__, 'a'+str(i), str(np.array(arr).shape), arr)
        err_msg = msg % tup
        assert_array_almost_equal(actual, desired, err_msg=err_msg)


def test_list_input():
    "Check that functions can handle list input"
    for func in all_functions():
        if func.__name__ != 'replace':
            yield unit_maker, func, eval('bn.slow.%s' % func.__name__)
