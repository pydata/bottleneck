
# For support of python 2.5
from __future__ import absolute_import, with_statement

import numpy as np
import bottleneck as bn
from .autotimeit import autotimeit

__all__ = ['bench']


def bench(mode='fast', dtype='float64', axis=1,
          shapes=[(10, 10), (100, 100), (1000, 1000), (10, 10), (100, 100),
                  (1000, 1000)],
          nans=[False, False, False, True, True, True]):
    """
    Bottleneck benchmark.

    Parameters
    ----------
    mode : {'fast', 'faster'}, optional
        Whether to benchmark the high-level functions ('fast') such as
        bn.median or the low-level functions ('faster') such as
        bn.func.median_2d_float64_axis0. By default the high-level functions
        are benchmarked.
    dtype : str, optional
        Data type string such as 'float64', which is the default.
    axis : int, optional
        Axis along which to perform the calculations that are being
        benchmarked. The default is the second axis (axis=1).
    shapes : list, optional
        A list of tuple shapes of input arrays to use in the benchmark.
    nans : list, optional
        A list of the bools (True or False), one for each tuple in the
        `shapes` list, that tells whether the input arrays should be filled
        with one-third NaNs.

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
    print("%sNaN means one-third NaNs; %s and axis=%s are used" % tup)
    if mode == 'fast':
        print("%sHigh-level functions used (mode='fast')" % tab)
    elif mode == 'faster':
        print("%sLow-level functions used (mode='faster')" % tab)

    print('')
    header = [" "*14]
    for nan in nans:
        if nan:
            header.append("NaN".center(11))
        else:
            header.append("no NaN".center(11))
    print("".join(header))
    header = ["".join(str(shape).split(" ")).center(11) for shape in shapes]
    header = [" "*14] + header
    print("".join(header))

    suite = benchsuite(mode, shapes, dtype, axis, nans)
    for test in suite:
        name = test["name"].ljust(12)
        fmt = name + "%10.2f" + "%11.2f"*(len(shapes) - 1)
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


def benchsuite(mode, shapes, dtype, axis, nans):

    if mode not in ('fast', 'faster'):
        raise ValueError("`mode` must be 'fast' or 'faster'")

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

    # median
    run = {}
    run['name'] = "median"
    run['ref'] = "np.median"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.median(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.median(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.median_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nanmedian
    run = {}
    run['name'] = "nanmedian"
    run['ref'] = "local copy of sp.stats.nanmedian"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nanmedian(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "scipy_nanmedian(a, axis=AXIS)"]
    setup = """
        from bottleneck.slow.func import scipy_nanmedian
        func, a = bn.func.nanmedian_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nansum
    run = {}
    run['name'] = "nansum"
    run['ref'] = "np.nansum"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nansum(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.nansum(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.nansum_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nanmax
    run = {}
    run['name'] = "nanmax"
    run['ref'] = "np.nanmax"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nanmax(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.nanmax(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.nanmax_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nanmean
    run = {}
    run['name'] = "nanmean"
    run['ref'] = "local copy of sp.stats.nanmean"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nanmean(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.nanmean(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.nanmean_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nanstd
    run = {}
    run['name'] = "nanstd"
    run['ref'] = "local copy of sp.stats.nanstd"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nanstd(a, axis=AXIS)"
    else:
        code = "func(a, 0)"
    run['statements'] = [code, "np.nanstd(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.nanstd_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # nanargmax
    run = {}
    run['name'] = "nanargmax"
    run['ref'] = "np.nanargmax"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.nanargmax(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.nanargmax(a, axis=AXIS)"]
    setup = """
        func, a = bn.func.nanargmax_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # ss
    run = {}
    run['name'] = "ss"
    run['ref'] = "scipy.stats.ss"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.ss(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "scipy_ss(a, axis=AXIS)"]
    setup = """
        from bottleneck.slow.func import scipy_ss
        func, a = bn.func.ss_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # rankdata
    run = {}
    run['name'] = "rankdata"
    run['ref'] = "scipy.stats.rankdata based (axis support added)"
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.rankdata(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "bn.slow.rankdata(a, axis=AXIS)"]
    setup = """
        ignore = bn.slow.rankdata(a, axis=AXIS)
        func, a = bn.func.rankdata_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # partsort
    run = {}
    run['name'] = "partsort"
    run['ref'] = "np.sort, n=max(a.shape[%s]/2,1)" % axis
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.partsort(a, n=n, axis=AXIS)"
    else:
        code = "func(a, n)"
    run['statements'] = [code, "np.sort(a, axis=AXIS)"]
    setup = """
        if AXIS is None: n = a.size
        else: n = a.shape[AXIS]
        n = max(n / 2, 1)
        func, a = bn.func.partsort_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # argpartsort
    run = {}
    run['name'] = "argpartsort"
    run['ref'] = "np.argsort, n=max(a.shape[%s]/2,1)" % axis
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.argpartsort(a, n=n, axis=AXIS)"
    else:
        code = "func(a, n)"
    run['statements'] = [code, "np.argsort(a, axis=AXIS)"]
    setup = """
        if AXIS is None: n = a.size
        else: n = a.shape[AXIS]
        n = max(n / 2, 1)
        func, a = bn.func.argpartsort_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # replace
    run = {}
    run['name'] = "replace"
    run['ref'] = "np.putmask based (see bn.slow.replace)"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.replace(a, np.nan, 0)"
    else:
        code = "func(a, np.nan, 0)"
    run['statements'] = [code, "replace(a, np.nan, 0)"]
    setup = """
        from bottleneck.slow.func import replace
        func = bn.func.replace_selector(a)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # anynan
    run = {}
    run['name'] = "anynan"
    run['ref'] = "np.isnan(arr).any(axis)"
    run['scipy_required'] = False
    if mode == 'fast':
        code = "bn.anynan(a, axis=AXIS)"
    else:
        code = "func(a)"
    run['statements'] = [code, "np.isnan(a).any(axis=AXIS)"]
    setup = """
        func, a = bn.func.anynan_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    suite.append(run)

    # move_sum
    run = {}
    run['name'] = "move_sum"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_sum(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_sum(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_sum as scipy_move_sum
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_sum(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_sum_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_nansum
    run = {}
    run['name'] = "move_nansum"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nansum(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_nansum(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_nansum as scipy_move_nansum
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_nansum(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_nansum_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_mean
    run = {}
    run['name'] = "move_mean"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_mean(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_mean(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_mean as scipy_move_mean
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_mean(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_mean_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_nanmean
    run = {}
    run['name'] = "move_nanmean"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nanmean(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_nanmean(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_nanmean as scipy_move_nanmean
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_nanmean(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_nanmean_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_std
    run = {}
    run['name'] = "move_std"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_std(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w, 0)"
    run['statements'] = [code, "scipy_move_std(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_std as scipy_move_std
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_std(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_std_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_nanstd
    run = {}
    run['name'] = "move_nanstd"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nanstd(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w, 0)"
    run['statements'] = [code, "scipy_move_nanstd(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_nanstd as scipy_move_nanstd
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_nanstd(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_nanstd_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_max
    run = {}
    run['name'] = "move_max"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_max(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_max(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_max as scipy_move_max
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_max(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_max_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
    if axis != 'None':
        suite.append(run)

    # move_nanmax
    run = {}
    run['name'] = "move_nanmax"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s] // 5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nanmax(a, window=w, axis=AXIS)"
    else:
        code = "func(a, w)"
    run['statements'] = [code, "scipy_move_nanmax(a, window=w, axis=AXIS)"]
    setup = """
        from bottleneck.slow.move import move_nanmax as scipy_move_nanmax
        w = a.shape[AXIS] // 5
        ignore = bn.slow.move_nanmax(a, window=w, axis=AXIS, method='filter')
        func, a = bn.move.move_nanmax_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup, shapes, nans)
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
