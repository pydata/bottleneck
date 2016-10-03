
import numpy as np
import bottleneck as bn
from .autotimeit import autotimeit

__all__ = ['bench']


def bench(dtype='float64', axis=-1,
          shapes=[(100,), (1000,), (1000, 1000), (1000, 1000)],
          nans=[False, True, False, True],
          order='C',
          functions=None):
    """
    Bottleneck benchmark.

    Parameters
    ----------
    dtype : str, optional
        Data type string such as 'float64', which is the default.
    axis : int, optional
        Axis along which to perform the calculations that are being
        benchmarked. The default is the last axis (axis=-1).
    shapes : list, optional
        A list of tuple shapes of input arrays to use in the benchmark.
    nans : list, optional
        A list of the bools (True or False), one for each tuple in the
        `shapes` list, that tells whether the input arrays should be randomly
        filled with one-fifth NaNs.
    order : {'C', 'F'}, optional
        Whether to store multidimensional data in C- or Fortran-contiguous
        (row- or column-wise) order in memory.
    functions : {list, None}, optional
        A list of strings specifying which functions to include in the
        benchmark. By default (None) all functions are included in the
        benchmark.

    Returns
    -------
    A benchmark report is printed to stdout.

    """

    if len(shapes) != len(nans):
        raise ValueError("`shapes` and `nans` must have the same length")

    dtype = str(dtype)
    axis = str(axis)

    tab = '    '

    # Header
    print('Bottleneck performance benchmark')
    print("%sBottleneck %s; Numpy %s" % (tab, bn.__version__, np.__version__))
    print("%sSpeed is NumPy time divided by Bottleneck time" % tab)
    tup = (tab, dtype, axis)
    print("%sNaN means approx one-fifth NaNs; %s and axis=%s are used" % tup)

    print('')
    header = [" "*15]
    for nan in nans:
        if nan:
            header.append("NaN".center(11))
        else:
            header.append("no NaN".center(11))
    print("".join(header))
    header = ["".join(str(shape).split(" ")).center(11) for shape in shapes]
    header = [" "*16] + header
    print("".join(header))

    suite = benchsuite(shapes, dtype, axis, nans, order, functions)
    for test in suite:
        name = test["name"].ljust(12)
        fmt = tab + name + "%7.1f" + "%11.1f"*(len(shapes) - 1)
        speed = timer(test['statements'], test['setups'])
        print(fmt % tuple(speed))


def timer(statements, setups):
    speed = []
    if len(statements) != 2:
        raise ValueError("Two statements needed.")
    for setup in setups:
        with np.errstate(invalid='ignore'):
            t0 = autotimeit(statements[0], setup)
            t1 = autotimeit(statements[1], setup)
        speed.append(t1 / t0)
    return speed


def getarray(shape, dtype, nans=False, order='C'):
    a = np.arange(np.prod(shape), dtype=dtype)
    if nans and issubclass(a.dtype.type, np.inexact):
        a[::5] = np.nan
    else:
        rs = np.random.RandomState(shape)
        rs.shuffle(a)
    return np.array(a.reshape(*shape), order=order)


def benchsuite(shapes, dtype, axis, nans, order, functions):

    suite = []

    def getsetups(setup, shapes, nans, order):
        template = """
        from bottleneck.benchmark.bench import getarray
        a = getarray(%s, 'DTYPE', %s, '%s')
        %s"""
        setups = []
        for shape, nan in zip(shapes, nans):
            setups.append(template % (str(shape), str(nan), order, setup))
        return setups

    # non-moving window functions
    funcs = bn.get_functions("reduce", as_string=True)
    funcs += ['rankdata', 'nanrankdata']
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        run['statements'] = ["bn_func(a, AXIS)", "sl_func(a, AXIS)"]
        setup = """
            from bottleneck import %s as bn_func
            try: from numpy import %s as sl_func
            except ImportError: from bottleneck.slow import %s as sl_func
            if "%s" == "median": from bottleneck.slow import median as sl_func
        """ % (func, func, func, func)
        run['setups'] = getsetups(setup, shapes, nans, order)
        suite.append(run)

    # partition, argpartition
    funcs = ['partition', 'argpartition']
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        run['statements'] = ["bn_func(a, n, AXIS)",
                             "sl_func(a, n, AXIS)"]
        setup = """
            from bottleneck import %s as bn_func
            from bottleneck.slow import %s as sl_func
            if AXIS is None: n = a.size
            else: n = a.shape[AXIS] - 1
            n = max(n / 2, 0)
        """ % (func, func)
        run['setups'] = getsetups(setup, shapes, nans, order)
        suite.append(run)

    # replace, push
    funcs = ['replace', 'push']
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        if func == 'replace':
            run['statements'] = ["bn_func(a, nan, 0)",
                                 "slow_func(a, nan, 0)"]
        elif func == 'push':
            run['statements'] = ["bn_func(a, 5, AXIS)",
                                 "slow_func(a, 5, AXIS)"]
        else:
            raise ValueError('Unknow function name')
        setup = """
            from numpy import nan
            from bottleneck import %s as bn_func
            from bottleneck.slow import %s as slow_func
        """ % (func, func)
        run['setups'] = getsetups(setup, shapes, nans, order)
        suite.append(run)

    # moving window functions
    funcs = bn.get_functions('move', as_string=True)
    for func in funcs:
        if functions is not None and func not in functions:
            continue
        run = {}
        run['name'] = func
        run['statements'] = ["bn_func(a, w, 1, AXIS)",
                             "sw_func(a, w, 1, AXIS)"]
        setup = """
            from bottleneck.slow.move import %s as sw_func
            from bottleneck import %s as bn_func
            w = a.shape[AXIS] // 5
        """ % (func, func)
        run['setups'] = getsetups(setup, shapes, nans, order)
        if axis != 'None':
            suite.append(run)

    # Strip leading spaces from setup code
    for i, run in enumerate(suite):
        for j in range(len(run['setups'])):
            t = run['setups'][j]
            t = '\n'.join([z.strip() for z in t.split('\n')])
            suite[i]['setups'][j] = t

    # Set dtype and axis in setups
    for i, run in enumerate(suite):
        for j in range(len(run['setups'])):
            t = run['setups'][j]
            t = t.replace('DTYPE', dtype)
            t = t.replace('AXIS', axis)
            suite[i]['setups'][j] = t

    # Set dtype and axis in statements
    for i, run in enumerate(suite):
        for j in range(2):
            t = run['statements'][j]
            t = t.replace('DTYPE', dtype)
            t = t.replace('AXIS', axis)
            suite[i]['statements'][j] = t

    return suite
