"Test functions."

# For support of python 2.5
from __future__ import with_statement

import numpy as np
from numpy.testing import assert_equal
nan = np.nan
import bottleneck as bn


def arrays(dtypes=bn.dtypes, nans=True):
    "Iterator that yield arrays to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(2, 3)]}
    ss[3] = {'size':  6, 'shapes': [(1, 2, 3)]}
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            for shape in shapes:
                a = a.reshape(shape)
                yield a
            if issubclass(a.dtype.type, np.inexact):
                if nans:
                    for i in range(a.size):
                        a.flat[i] = np.nan
                        yield a
                for i in range(a.size):
                    a.flat[i] = np.inf
                    yield a


def unit_maker(func, nans=True):
    "Test that bn.xxx gives the same output as np.xxx."
    msg = "\nInput array modifed by %s.\n\n"
    msg += "input array before:\n%s\nafter:\n%s\n"
    for i, arr in enumerate(arrays(nans=nans)):
        for axis in list(range(-arr.ndim, arr.ndim)) + [None]:
            with np.errstate(invalid='ignore'):
                arr1 = arr.copy()
                arr2 = arr.copy()
                if ('move_' in func.__name__) or ('sort' in func.__name__):
                    if axis is None:
                        continue
                    actual = func(arr1, 1, axis=axis)
                else:
                    try:
                        actual = func(arr1, axis=axis)
                    except:
                        continue
                assert_equal(arr1, arr2, msg % (func.__name__, arr1, arr2))


def test_modification():
    "Test for illegal inplace modification of input array"
    funcs = [bn.nansum,
             bn.nanmax,
             bn.nanargmin,
             bn.nanargmax,
             bn.nanmin,
             bn.nanmean,
             bn.nanstd,
             bn.nanvar,
             bn.median,
             bn.nanmedian,
             bn.rankdata,
             bn.nanrankdata,
             bn.ss,
             bn.partsort,
             bn.argpartsort,
             bn.anynan,
             bn.allnan,
             bn.move_sum,
             bn.move_nansum,
             bn.move_mean,
             bn.move_median,
             bn.move_nanmean,
             bn.move_std,
             bn.move_nanstd,
             bn.move_min,
             bn.move_max,
             bn.move_nanmin,
             bn.move_nanmax]
    for func in funcs:
        yield unit_maker, func
