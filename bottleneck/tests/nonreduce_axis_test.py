import numpy as np
from numpy.testing import (assert_equal, assert_array_equal,
                           assert_array_almost_equal, assert_raises)

import bottleneck as bn
from .reduce_test import (arrays_strides, unit_maker as reduce_unit_maker,
                          unit_maker_argparse as unit_maker_parse_rankdata)

DTYPES = [np.float64, np.float32, np.int64, np.int32]
nan = np.nan


# ---------------------------------------------------------------------------
# partsort, argpartsort

def test_partsort():
    "test partsort"
    for func in (bn.partsort,):
        yield unit_maker, func


def test_argpartsort():
    "test argpartsort"
    for func in (bn.argpartsort,):
        yield unit_maker, func


def unit_maker(func):
    "test partsort or argpartsort"

    length = 9
    nrepeat = 10

    msg = '\nfunc %s | input %s (%s) | shape %s | n %d | axis %s\n'
    msg += '\nInput array:\n%s\n'

    name = func.__name__
    func0 = eval('bn.slow.%s' % name)

    rs = np.random.RandomState([1, 2, 3])
    for ndim in (1, 2):
        if ndim == 1:
            shape = (length,)
        elif ndim == 2:
            shape = (2, length)
        for i in range(nrepeat):
            a = rs.randint(0, 10, shape)
            for dtype in DTYPES:
                a = a.astype(dtype)
                for axis in list(range(-1, ndim)) + [None]:
                    if axis is None:
                        nmax = a.size
                    else:
                        nmax = a.shape[axis]
                    n = rs.randint(1, nmax)
                    s0 = func0(a, n, axis)
                    s1 = func(a, n, axis)
                    if name == 'argpartsort':
                        s0 = complete_the_argpartsort(s0, a, n, axis)
                        s1 = complete_the_argpartsort(s1, a, n, axis)
                    else:
                        s0 = complete_the_partsort(s0, n, axis)
                        s1 = complete_the_partsort(s1, n, axis)
                    tup = (name, 'a'+str(i), str(a.dtype), str(a.shape), n,
                           str(axis), a)
                    err_msg = msg % tup
                    assert_array_equal(s1, s0, err_msg)


def complete_the_partsort(a, n, axis):
    ndim = a.ndim
    if axis is None:
        if ndim != 1:
            raise ValueError("`a` must be 1d when axis is None")
        axis = 0
    elif axis < 0:
        axis += ndim
        if axis < 0:
            raise ValueError("`axis` out of range")
    if ndim == 1:
        a[:n-1] = np.sort(a[:n-1])
        a[n:] = np.sort(a[n:])
    elif ndim == 2:
        if axis == 0:
            for i in range(a.shape[1]):
                a[:n-1, i] = np.sort(a[:n-1, i])
                a[n:, i] = np.sort(a[n:, i])
        elif axis == 1:
            for i in range(a.shape[0]):
                a[i, :n-1] = np.sort(a[i, :n-1])
                a[i, n:] = np.sort(a[i, n:])
        else:
            raise ValueError("`axis` out of range")
    else:
        raise ValueError("`a.ndim` must be 1 or 2")
    return a


def complete_the_argpartsort(index, a, n, axis):
    ndim = a.ndim
    if axis is None:
        if index.ndim != 1:
            raise ValueError("`index` must be 1d when axis is None")
        axis = 0
        ndim = 1
        a = a.copy().reshape(-1)
    elif axis < 0:
        axis += ndim
        if axis < 0:
            raise ValueError("`axis` out of range")
    if ndim == 1:
        a = a[index]
    elif ndim == 2:
        if axis == 0:
            for i in range(a.shape[1]):
                a[:, i] = a[index[:, i], i]
        elif axis == 1:
            for i in range(a.shape[0]):
                a[i] = a[i, index[i]]
        else:
            raise ValueError("`axis` out of range")
    else:
        raise ValueError("`a.ndim` must be 1 or 2")
    a = complete_the_partsort(a, n, axis)
    return a


def test_transpose():
    "partsort transpose test"
    a = np.arange(12).reshape(4, 3)
    actual = bn.partsort(a.T, 2, -1).T
    desired = bn.slow.partsort(a.T, 2, -1).T
    assert_equal(actual, desired, 'partsort transpose test')


# ---------------------------------------------------------------------------
# rankdata, nanrankdata, push

def test_nonreduce_axis():
    "Test nonreduce axis functions"
    funcs = [bn.rankdata, bn.nanrankdata, bn.push]
    for func in funcs:
        yield reduce_unit_maker, func


def test_push():
    "Test push"
    ns = (0, 1, 2, 3, 4, 5)
    a = np.array([np.nan, 1, 2, np.nan, np.nan, np.nan, np.nan, 3, np.nan])
    for n in ns:
        actual = bn.push(a.copy(), n=n)
        desired = bn.slow.push(a.copy(), n=n)
        assert_array_equal(actual, desired, "failed on n=%s" % str(n))


# ---------------------------------------------------------------------------
# Test with arrays that are not C ordered

def test_strides():
    "test nonreducer_axis functions with non-C ordered arrays"
    for func in bn.get_functions('nonreduce_axis'):
        yield unit_maker_strides, func


def unit_maker_strides(func, decimal=5):
    "Test that bn.xxx gives the same output as bn.slow.xxx."
    fmt = '\nfunc %s | input %s (%s) | shape %s | axis %s\n'
    fmt += '\nInput array:\n%s\n'
    fmt += '\nStrides: %s\n'
    fmt += '\nFlags: \n%s\n'
    name = func.__name__
    func0 = eval('bn.slow.%s' % name)
    for i, a in enumerate(arrays_strides()):
        if a.ndim == 0:
            axes = [None]  # numpy can't handle e.g. np.nanmean(9, axis=-1)
        else:
            axes = list(range(-1, a.ndim)) + [None]
        for axis in axes:
            # do not use a.copy() here because it will C order the array
            if name in ('partsort', 'argpartsort', 'push'):
                if axis is None:
                    if name == 'push':
                        continue
                    n = min(2, a.size)
                else:
                    n = min(2, a.shape[axis])
                actual = func(a, n, axis=axis)
                desired = func0(a, n, axis=axis)
            else:
                actual = func(a, axis=axis)
                desired = func0(a, axis=axis)
            tup = (name, 'a'+str(i), str(a.dtype), str(a.shape),
                   str(axis), a, a.strides, a.flags)
            err_msg = fmt % tup
            assert_array_almost_equal(actual, desired, decimal, err_msg)
            err_msg += '\n dtype mismatch %s %s'


# ---------------------------------------------------------------------------
# Test argument parsing

def test_arg_parsing():
    "test argument parsing in nonreduce_axis"
    for func in bn.get_functions('nonreduce_axis'):
        name = func.__name__
        if name in ('partsort', 'argpartsort'):
            yield unit_maker_parse, func
        elif name in ('push'):
            yield unit_maker_parse, func
        elif name in ('rankdata', 'nanrankdata'):
            yield unit_maker_parse_rankdata, func
        else:
            fmt = "``%s` is an unknown nonreduce_axis function"
            raise ValueError(fmt % name)
        yield unit_maker_raises, func


def unit_maker_parse(func, decimal=5):
    "test argument parsing."

    name = func.__name__
    func0 = eval('bn.slow.%s' % name)

    a = np.array([1., 2, 3])

    fmt = '\n%s' % func
    fmt += '%s\n'
    fmt += '\nInput array:\n%s\n' % a

    actual = func(a, 1)
    desired = func0(a, 1)
    err_msg = fmt % "(a, 1)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    actual = func(a, 1, axis=0)
    desired = func0(a, 1, axis=0)
    err_msg = fmt % "(a, 1, axis=0)"
    assert_array_almost_equal(actual, desired, decimal, err_msg)

    if name != 'push':

        actual = func(a, 2, None)
        desired = func0(a, 2, None)
        err_msg = fmt % "(a, 2, None)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

        actual = func(a, 1, axis=None)
        desired = func0(a, 1, axis=None)
        err_msg = fmt % "(a, 1, axis=None)"
        assert_array_almost_equal(actual, desired, decimal, err_msg)

        # regression test: make sure len(kwargs) == 0 doesn't raise
        args = (a, 1, -1)
        kwargs = {}
        func(*args, **kwargs)

    else:

        # regression test: make sure len(kwargs) == 0 doesn't raise
        args = (a, 1)
        kwargs = {}
        func(*args, **kwargs)


def unit_maker_raises(func):
    "test argument parsing raises in nonreduce_axis"
    a = np.array([1., 2, 3])
    assert_raises(TypeError, func)
    assert_raises(TypeError, func, axis=a)
    assert_raises(TypeError, func, a, axis=0, extra=0)
    assert_raises(TypeError, func, a, axis=0, a=a)
    if func.__name__ in ('partsort', 'argpartsort'):
        assert_raises(TypeError, func, a, 0, 0, 0, 0, 0)
        assert_raises(TypeError, func, a, axis='0')
