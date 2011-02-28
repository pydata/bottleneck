"Test that no slow functions creep in where they don't belong."

import numpy as np
import bottleneck as bn


def arrayaxis(dtypes=bn.dtypes):
    "Iterator that yield arrays and axis to use for unit testing."
    ss = {}
    ss[1] = {'size':  4, 'shapes': [(4,)]}
    ss[2] = {'size':  6, 'shapes': [(2,3)]}
    ss[3] = {'size': 24, 'shapes': [(2,3,4)]}
    for ndim in ss:
        size = ss[ndim]['size']
        shapes = ss[ndim]['shapes']
        for dtype in dtypes:
            a = np.arange(size, dtype=dtype)
            if not issubclass(a.dtype.type, np.inexact):
                for shape in shapes:
                    a = a.reshape(shape)
                    for axis in range(-a.ndim, a.ndim) + [None]:
                        yield a.copy(), axis

def fast_checker(selector, mode='func'):
    for arr, axis in arrayaxis():
        if mode == 'func':
            func, a = selector(arr, axis)
        elif mode == 'move':
            if axis is not None:
                func, a = selector(arr, axis)
            else:
                func = np.sum
        else:
            raise ValueError("`mode` value not recognized.")
        if 'slow' in func.__name__:
            raise AssertionError("slow version of func unexpectedly called.")

# Functions -----------------------------------------------------------------

def test_median_selector():
    "Test median_selector."
    fast_checker(bn.func.median_selector)

def test_nanmedian_selector():
    "Test nanmedian_selector."
    fast_checker(bn.func.nanmedian_selector)

def test_nansum_selector():
    "Test nansum_selector."
    fast_checker(bn.func.nansum_selector)

def test_nanmin_selector():
    "Test nanmin_selector."
    fast_checker(bn.func.nanmin_selector)

def test_nanmax_selector():
    "Test nanmax_selector."
    fast_checker(bn.func.nanmax_selector)

def test_nanmean_selector():
    "Test nanmean_selector."
    fast_checker(bn.func.nanmean_selector)

def test_nanstd_selector():
    "Test nanstd_selector."
    fast_checker(bn.func.nanstd_selector)

def test_nanargmin_selector():
    "Test nanargmin_selector."
    fast_checker(bn.func.nanargmin_selector)

def test_nanargmax_selector():
    "Test nanargmax_selector."
    fast_checker(bn.func.nanargmax_selector)

def test_nanvar_selector():
    "Test nanvar_selector."
    fast_checker(bn.func.nanvar_selector)

def test_rankdata_selector():
    "Test rankdata_selector."
    fast_checker(bn.func.rankdata_selector)

def test_nanrankdata_selector():
    "Test nanrankdata_selector."
    fast_checker(bn.func.nanrankdata_selector)

# Moving functions ----------------------------------------------------------

def test_move_sum_selector():
    "Test move_sum_selector."
    fast_checker(bn.move.move_sum_selector, mode='move')

def test_move_nansum_selector():
    "Test move_nansum_selector."
    fast_checker(bn.move.move_nansum_selector, mode='move')

def test_move_mean_selector():
    "Test move_mean_selector."
    fast_checker(bn.move.move_mean_selector, mode='move')

def test_move_nanmean_selector():
    "Test move_nanmean_selector."
    fast_checker(bn.move.move_nanmean_selector, mode='move')

def test_move_std_selector():
    "Test move_std_selector."
    fast_checker(bn.move.move_std_selector, mode='move')

def test_move_nanstd_selector():
    "Test move_nanstd_selector."
    fast_checker(bn.move.move_nanstd_selector, mode='move')

def test_move_min_selector():
    "Test move_min_selector."
    fast_checker(bn.move.move_min_selector, mode='move')

def test_move_max_selector():
    "Test move_max_selector."
    fast_checker(bn.move.move_max_selector, mode='move')

def test_move_nanmin_selector():
    "Test move_nanmin_selector."
    fast_checker(bn.move.move_nanmin_selector, mode='move')

def test_move_nanmixn_selector():
    "Test move_nanmax_selector."
    fast_checker(bn.move.move_nanmax_selector, mode='move')
