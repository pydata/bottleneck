
import numpy as np
try:
    import scipy as sp
    SCIPY = True
except ImportError:
    SCIPY = False
import bottleneck as bn
from autotimeit import autotimeit

__all__ = ['bench']

def bench(mode='fast', dtype='float64', axis=0):

    dtype = str(dtype)
    axis = str(axis)

    tab = '    '

    # Header
    print 'Bottleneck performance benchmark'
    print "%sBottleneck  %s" % (tab, bn.__version__)
    print "%sNumpy (np)  %s" % (tab, np.__version__)
    if SCIPY:
        print "%sScipy (sp)  %s" % (tab, sp.__version__)
    else:
        print "%sScipy (sp)  Cannot import, skipping scipy benchmarks" % tab
    print "%sSpeed is NumPy or SciPy time divided by Bottleneck time" % tab
    tup = (tab, dtype, axis)
    print "%sNaN means one-third NaNs; %s and axis=%s are used" % tup 
    if mode == 'fast':
        print "%sHigh-level functions used (mode='fast')" % tab
    elif mode == 'faster':    
        print "%sLow-level functions used (mode='faster')" % tab
    
    h1 = "%s no NaN   no NaN     no NaN     NaN      NaN        NaN"
    h2 = "%s(10,10) (100,100) (1000,1000) (10,10) (100,100) (1000,1000)"
    print
    print h1 % (16*" ")
    print h2 % (16*" ")

    suite = benchsuite(mode, dtype, axis)
    fmt = "%s%6.2f   %6.2f     %6.2f    %6.2f   %6.2f     %6.2f"
    for test in suite:
        name = test["name"].ljust(16)
        if test['scipy_required'] and not SCIPY:
            print "%s%s" % (name, "requires SciPy")
        else:
            s = timer(test['statements'], test['setups'])
            print fmt % (name, s[0], s[1], s[2], s[3], s[4], s[5])

    print
    print 'Reference functions:'
    for test in suite:
        print "%s%s" % (test["name"].ljust(16), test['ref'])

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
    
def benchsuite(mode, dtype, axis):

    if mode not in ('fast', 'faster'):
        raise ValueError("`mode` must be 'fast' or 'faster'")

    suite = []
   
    def getsetups(setup):
        template = """import numpy as np
        import bottleneck as bn
        from bottleneck.benchmark.bench import getarray
        N = %d
        a = getarray((N,N), 'DTYPE', %s)
        %s"""
        setups = []
        setups.append(template % (10, str(False), setup))
        setups.append(template % (100, str(False), setup))
        setups.append(template % (1000, str(False), setup))
        setups.append(template % (10, str(True), setup))
        setups.append(template % (100, str(True), setup))
        setups.append(template % (1000, str(True), setup))
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
    run['setups'] = getsetups(setup)
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
    run['setups'] = getsetups(setup)
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
    run['setups'] = getsetups(setup)
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
    run['setups'] = getsetups(setup)
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
    run['statements'] = [code, "scipy_nanmean(a, axis=AXIS)"] 
    setup = """
        from bottleneck.slow.func import scipy_nanmean
        func, a = bn.func.nanmean_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup)
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
    run['statements'] = [code, "scipy_nanstd(a, axis=AXIS)"] 
    setup = """
        from bottleneck.slow.func import scipy_nanstd
        func, a = bn.func.nanstd_selector(a, axis=AXIS)
    """
    run['setups'] = getsetups(setup)
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
    run['setups'] = getsetups(setup)
    suite.append(run)
    
    # move_nanmean
    run = {}
    run['name'] = "move_nanmean"
    run['ref'] = "sp.ndimage.convolve1d based, "
    run['ref'] += "window=a.shape[%s]/5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nanmean(a, window=w, axis=AXIS)"
    else:
        code = "func(a, 5)"
    run['statements'] = [code, "scipy_move_nanmean(a, window=w, axis=AXIS)"] 
    setup = """
        from bottleneck.slow.move import move_nanmean as scipy_move_nanmean
        w = a.shape[AXIS] / 5
        func, a = bn.move.move_nanmean_selector(a, window=w, axis=AXIS)
    """
    run['setups'] = getsetups(setup)
    if axis != 'None':
        suite.append(run)
    
    # move_max
    run = {}
    run['name'] = "move_max"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s]/5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_max(a, window=w, axis=AXIS)"
    else:
        code = "func(a, 5)"
    run['statements'] = [code, "scipy_move_max(a, window=w, axis=AXIS)"] 
    setup = """
        from bottleneck.slow.move import move_max as scipy_move_max
        w = a.shape[AXIS] / 5
        func, a = bn.move.move_max_selector(a, window=w, axis=AXIS)
    """
    run['setups'] = getsetups(setup)
    if axis != 'None':
        suite.append(run)
    
    # move_nanmax
    run = {}
    run['name'] = "move_nanmax"
    run['ref'] = "sp.ndimage.maximum_filter1d based, "
    run['ref'] += "window=a.shape[%s]/5" % axis
    run['scipy_required'] = True
    if mode == 'fast':
        code = "bn.move_nanmax(a, window=w, axis=AXIS)"
    else:
        code = "func(a, 5)"
    run['statements'] = [code, "scipy_move_nanmax(a, window=w, axis=AXIS)"] 
    setup = """
        from bottleneck.slow.move import move_nanmax as scipy_move_nanmax
        w = a.shape[AXIS] / 5
        func, a = bn.move.move_nanmax_selector(a, window=w, axis=AXIS)
    """
    run['setups'] = getsetups(setup)
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
