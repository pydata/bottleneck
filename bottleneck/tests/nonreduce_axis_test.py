import numpy as np
from numpy.testing import assert_equal, assert_array_equal

import bottleneck as bn
from .reduce_test import unit_maker as reduce_unit_maker

DTYPES = [np.float64, np.float32, np.int64, np.int32]
nan = np.nan


# ---------------------------------------------------------------------------
# partsort, argpartsort

def test_partsort():
    "test partsort"
    for func in (bn.partsort, bn.partsort2):
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
    funcs = [bn.rankdata, bn.rankdata2, bn.nanrankdata, bn.nanrankdata2,
             bn.push]
    for func in funcs:
        yield reduce_unit_maker, func


def test_push_2():
    "Test push #2."
    ns = (np.inf, -1, 0, 1, 2, 3, 4, 5, 1.1, 1.5, 1.9)
    arr = np.array([np.nan, 1, 2, np.nan, np.nan, np.nan, np.nan, 3, np.nan])
    for n in ns:
        actual = bn.push(arr.copy(), n=n)
        desired = bn.slow.push(arr.copy(), n=n)
        assert_array_equal(actual, desired, "failed on n=%s" % str(n))
