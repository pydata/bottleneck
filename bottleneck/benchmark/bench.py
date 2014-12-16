
# For support of python 2.5
from __future__ import absolute_import, with_statement

import numpy as np
import bottleneck as bn
from .autotimeit import autotimeit

__all__ = ['bench']


def bench(dtype='float64', axis=-1,
          shapes=[(10,), (1000, 1000), (10,), (1000, 1000)],
          nans=[False, False, True, True]):
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
        filled with one-third NaNs.

    Returns
    -------
    A benchmark report is printed to stdout.

    """

    try:
        import scipy as sp
        SCIPY = True
    except ImportError:
        SCIPY = False

    if len(shapes) != len(nans):
        raise ValueError("`shapes` and `nans` must have the same length")

    dtype = str(dtype)
    axis = str(axis)

    tab = '    '

    # Header
    print('Bottleneck performance benchmark')
    print("%sBottleneck  %s" % (tab, bn.__version__))
    print("%sNumpy (np)  %s" % (tab, np.__version__))
    if SCIPY:
        print("%sScipy (sp)  %s" % (tab, sp.__version__))
    else:
        print("%sScipy (sp)  Cannot import, skipping scipy benchmarks" % tab)
    print("%sSpeed is NumPy or SciPy time divided by Bottleneck time" % tab)
    tup = (tab, dtype, axis)
    print("%sNaN means approx one-third NaNs; %s and axis=%s are used" % tup)

    print('')
    header = [" "*14]
    for nan in nans:
        if nan:
            header.append("NaN".center(11))
        else:
            header.append("no NaN".center(11))
    print("".join(header))
    header = ["".join(str(shape).split(" ")).center(11) for shape in shapes]
    header = [" "*16] + header
    print("".join(header))

    suite = benchsuite(shapes, dtype, axis, nans)
    for test in suite:
        name = test["name"].ljust(12)
        fmt = tab + name + "%7.1f" + "%11.1f"*(len(shapes) - 1)
        if test['scipy_required'] and not SCIPY:
            print("%s%s" % (name, "requires SciPy"))
        else:
            speed = timer(test['statements'], test['setups'])
            print(fmt % tuple(speed))

    print('')
    print('Reference functions:')
    for test in suite:
        print("%s%s" % (test["name"].ljust(15), test['ref']))


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


def getarray(shape, dtype, nans=False):
    arr = np.arange(np.prod(shape), dtype=dtype)
    if nans and issubclass(arr.dtype.type, np.inexact):
        arr[::3] = np.nan
    else:
        rs = np.random.RandomState(shape)
        rs.shuffle(arr)
    return arr.reshape(*shape)


def benchsuite(shapes, dtype, axis, nans):

    suite = []

    def getsetups(setup, shapes, nans):
        template = """import numpy as np
        import bottleneck as bn
        from bottleneck.benchmark.bench import getarray
        a = getarray(%s, 'DTYPE', %s)
        %s"""
        setups = []
        for shape, nan in zip(shapes, nans):
            setups.append(template % (str(shape), str(nan), setup))
        return setups

    # numpy functions
    funcs = ['nansum', 'nanmean', 'nanstd', 'nanmax']
    for func in funcs:
        run = {}
        run['name'] = func
        run['ref'] = "np.%s" % func
        run['scipy_required'] = False
        run['statements'] = ["bn_func(a, axis=AXIS)", "np_func(a, axis=AXIS)"]
        setup = """
            from bottleneck import %s as bn_func
            from numpy import %s as np_func
        """ % (func, func)
        run['setups'] = getsetups(setup, shapes, nans)
        suite.append(run)

    # partsort, argpartsort
    funcs = ['partsort', 'argpartsort']
    for func in funcs:
        run = {}
        run['name'] = func
        if func == 'argpartsort':
            pre = 'arg'
        else:
            pre = ''
        run['ref'] = "np.%ssort, n=max(a.shape[%s]/2,1)" % (pre, axis)
        run['scipy_required'] = False
        run['statements'] = ["bn_func(a, n=n, axis=AXIS)",
                             "np_func(a, axis=AXIS)"]
        setup = """
            from bottleneck import %s as bn_func
            from numpy import %ssort as np_func
            if AXIS is None: n = a.size
            else: n = a.shape[AXIS]
            n = max(n / 2, 1)
        """ % (func, pre)
        run['setups'] = getsetups(setup, shapes, nans)
        suite.append(run)

    # replace
    run = {}
    run['name'] = "replace"
    run['ref'] = "np.putmask based (see bn.slow.replace)"
    run['scipy_required'] = False
    run['statements'] = ["bn_func(a, np.nan, 0)",
                         "slow_func(a, np.nan, 0)"]
    setup = """
        from bottleneck import replace as bn_func
        from bottleneck.slow import replace as slow_func
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # moving window function that benchmark against sp.ndimage.convolve1d
    funcs = ['move_sum', 'move_nansum', 'move_mean', 'move_nanmean',
             'move_std', 'move_nanstd']
    for func in funcs:
        run = {}
        run['name'] = func
        run['ref'] = "sp.ndimage.convolve1d based, "
        run['ref'] += "window=a.shape[%s] // 5" % axis
        run['scipy_required'] = True
        code = ["bn_func(a, window=w, axis=AXIS)",
                "sp_func(a, window=w, axis=AXIS, method='filter')"]
        run['statements'] = code
        setup = """
            from bottleneck.slow.move import %s as sp_func
            from bottleneck import %s as bn_func
            w = a.shape[AXIS] // 5
        """ % (func, func)
        run['setups'] = getsetups(setup, shapes, nans)
        if axis != 'None':
            suite.append(run)

    # move_median
    run = {}
    func = 'move_median'
    run['name'] = func
    run['ref'] = "for loop with np.median"
    run['scipy_required'] = False
    code = ["bn_func(a, window=w, axis=AXIS)",
            "sl_func(a, window=w, axis=AXIS, method='loop')"]
    run['statements'] = code
    setup = """
        from bottleneck.slow.move import %s as sl_func
        from bottleneck import %s as bn_func
        w = a.shape[AXIS] // 5
    """ % (func, func)
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # runs
    # -----------------------------------------------------------------------
    # does not yet run

    # median
    run = {}
    run['name'] = "median"
    run['ref'] = "np.median"
    run['scipy_required'] = False
    code = "bn.median(a, axis=AXIS)"
    run['statements'] = [code, "np.median(a, axis=AXIS)"]
    setup = """
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # nanmedian
    run = {}
    run['name'] = "nanmedian"
    run['ref'] = "local copy of sp.stats.nanmedian"
    run['scipy_required'] = False
    code = "bn.nanmedian(a, axis=AXIS)"
    run['statements'] = [code, "scipy_nanmedian(a, axis=AXIS)"]
    setup = """
        from bottleneck.slow.func import scipy_nanmedian
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # nanargmax
    run = {}
    run['name'] = "nanargmax"
    run['ref'] = "np.nanargmax"
    run['scipy_required'] = False
    code = "bn.nanargmax(a, axis=AXIS)"
    run['statements'] = [code, "np.nanargmax(a, axis=AXIS)"]
    setup = """
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # ss
    run = {}
    run['name'] = "ss"
    run['ref'] = "scipy.stats.ss"
    run['scipy_required'] = False
    code = "bn.ss(a, axis=AXIS)"
    run['statements'] = [code, "scipy_ss(a, axis=AXIS)"]
    setup = """
        from bottleneck.slow.func import scipy_ss
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # rankdata
    run = {}
    run['name'] = "rankdata"
    run['ref'] = "scipy.stats.rankdata based (axis support added)"
    run['scipy_required'] = True
    code = "bn.rankdata(a, axis=AXIS)"
    run['statements'] = [code, "bn.slow.rankdata(a, axis=AXIS)"]
    setup = """
        ignore = bn.slow.rankdata(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # anynan
    run = {}
    run['name'] = "anynan"
    run['ref'] = "np.isnan(arr).any(axis)"
    run['scipy_required'] = False
    code = "bn.anynan(a, axis=AXIS)"
    run['statements'] = [code, "np.isnan(a).any(axis=AXIS)"]
    setup = """
    """
    run['setups'] = getsetups(setup, shapes, nans)
    #suite.append(run)

    # move_max
    run = {}
    run['name'] = "move_max"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    code = "bn.move_max(a, window=w, axis=AXIS)"
    run['statements'] = [code, "scipy_move_max(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_max as scipy_move_max
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_max(a, window=w, axis=AXIS, method='filter')
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        pass#suite.append(run)

    # move_nanmax
    run = {}
    run['name'] = "move_nanmax"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    code = "bn.move_nanmax(a, window=w, axis=AXIS)"
    run['statements'] = [code, "scipy_move_nanmax(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_nanmax as scipy_move_nanmax
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_nanmax(a, window=w, axis=AXIS, method='filter')
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        pass#suite.append(run)

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
