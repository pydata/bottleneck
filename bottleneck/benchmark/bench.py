
import numpy as np
try:
    import scipy as sp
    SCIPY = True
except ImportError:
    SCIPY = False
import bottleneck as bn
from autotimeit import autotimeit

__all__ = ['bench']

def bench(mode='fast'):

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
    print "%sNaN means all NaNs; axis=0 and float64 are used" % tab

    suite = benchsuite(mode)
    for test in suite:
        if test['scipy_required'] and not SCIPY:
            print test["name"] + "requires SciPy"
        else:
            print test["name"]
        speed = timer(test['statements'], test['setups'])
        results = []
        for i, name in enumerate(test['setups']):
            results.append("%8.2f  %s" % (speed[i], name))
        speed = -np.array(speed)
        index = speed.argsort()
        results = [results[idx] for idx in index]
        print '\n'.join(results)

def timer(statements, setups):
    speed = []
    if len(statements) != 2:
        raise ValueError("Two statements needed.")
    for name in setups:
        with np.errstate(invalid='ignore'):
            t0 = autotimeit(statements[0], setups[name])
            t1 = autotimeit(statements[1], setups[name])
        speed.append(t1 / t0)
    return speed

def geta(shape, dtype, nans=False):
    arr = np.arange(np.prod(shape), dtype=dtype).reshape(*shape)
    if nans:
        arr[:] = np.nan
    return arr 
    
def benchsuite(mode='fast'):

    if mode not in ('fast', 'faster'):
        raise ValueError("`mode` must be 'fast' or 'faster'")
    
    suite = []
    
    # median
    run = {}
    run['scipy_required'] = False
    if mode == 'fast':
        run['name'] = "median vs np.median"
        code = "bn.median(a, axis=0)"
    else:
        run['name'] = "median_selector vs np.median"
        code = "func(a)"
    run['statements'] = [code, "np.median(a, axis=0)"] 
    setup = """
        import numpy as np
        import bottleneck as bn
        from bottleneck.benchmark.bench import geta
        N = %d
        a = geta((N,N), 'float64')
        func, a = bn.func.median_selector(a, axis=0)
    """
    setups = {}
    setups["(10,10)         "] = setup % 10
    setups["(100,100)       "] = setup % 100
    setups["(1000,1000)     "] = setup % 1000
    run['setups'] = setups
    suite.append(run)

    # nanmax
    run = {}
    run['scipy_required'] = False
    if mode == 'fast':
        run['name'] = "nanmax vs np.nanmax"
        code = "bn.nanmax(a, axis=0)"
    else:
        run['name'] = "nanmax_selector vs np.nanmax"
        code = "func(a)"
    run['statements'] = [code, "np.nanmax(a, axis=0)"] 
    setup = """
        import numpy as np
        import bottleneck as bn
        from bottleneck.benchmark.bench import geta
        N = %d
        a = geta((N,N), 'float64', %s)
        func, a = bn.func.nanmax_selector(a, axis=0)
    """
    setups = {}
    setups["(10,10)         "] = setup % (10, str(False))
    setups["(10,10)      NaN"] = setup % (10, str(True))
    setups["(100,100)       "] = setup % (100, str(False))
    setups["(100,100)    NaN"] = setup % (100, str(True))
    setups["(1000,1000)     "] = setup % (1000, str(False))
    setups["(1000,1000)  NaN"] = setup % (1000, str(True))
    run['setups'] = setups
    suite.append(run)
    
    # nanmin
    run = {}
    run['scipy_required'] = False
    if mode == 'fast':
        run['name'] = "nanmin vs np.nanmin"
        code = "bn.nanmin(a, axis=0)"
    else:
        run['name'] = "nanmin_selector vs np.nanmin"
        code = "func(a)"
    run['statements'] = [code, "np.nanmin(a, axis=0)"] 
    setup = """
        import numpy as np
        import bottleneck as bn
        from bottleneck.benchmark.bench import geta
        N = %d
        a = geta((N,N), 'float64', %s)
        func, a = bn.func.nanmin_selector(a, axis=0)
    """
    setups = {}
    setups["(10,10)         "] = setup % (10, str(False))
    setups["(10,10)      NaN"] = setup % (10, str(True))
    setups["(100,100)       "] = setup % (10, str(False))
    setups["(100,100)    NaN"] = setup % (10, str(True))
    setups["(1000,1000)     "] = setup % (1000, str(False))
    setups["(1000,1000)  NaN"] = setup % (1000, str(True))
    run['setups'] = setups
    suite.append(run)
    
    # nanmean
    run = {}
    run['scipy_required'] = False
    if mode == 'fast':
        run['name'] = "nanmean vs local copy of sp.stats.nanmean"
        code = "bn.nanmean(a, axis=0)"
    else:
        run['name'] = "nanmean_selector vs local copy of sp.stats.nanmean"
        code = "func(a)"
    run['statements'] = [code, "scipy_nanmean(a, axis=0)"] 
    setup = """
        import numpy as np
        import bottleneck as bn
        from bottleneck.slow.func import scipy_nanmean
        from bottleneck.benchmark.bench import geta
        N = %d
        a = geta((N,N), 'float64', %s)
        func, a = bn.func.nanmean_selector(a, axis=0)
    """
    setups = {}
    setups["(10,10)         "] = setup % (10, str(False))
    setups["(10,10)      NaN"] = setup % (10, str(True))
    setups["(100,100)       "] = setup % (100, str(False))
    setups["(100,100)    NaN"] = setup % (100, str(True))
    setups["(1000,1000)     "] = setup % (1000, str(False))
    setups["(1000,1000)  NaN"] = setup % (1000, str(True))
    run['setups'] = setups
    suite.append(run)

    # nanstd
    run = {}
    run['scipy_required'] = False
    if mode == 'fast':
        run['name'] = "nanstd vs local copy of sp.stats.nanstd"
        code = "bn.nanstd(a, axis=0)"
    else:
        run['name'] = "nanstd_selector vs local copy of sp.stats.nanstd"
        code = "func(a, 0)"
    run['statements'] = [code, "scipy_nanstd(a, axis=0)"] 
    setup = """
        import numpy as np
        import bottleneck as bn
        from bottleneck.slow.func import scipy_nanstd
        from bottleneck.benchmark.bench import geta
        N = %d
        a = geta((N,N), 'float64', %s)
        func, a = bn.func.nanstd_selector(a, axis=0)
    """
    setups = {}
    setups["(10,10)         "] = setup % (10, str(False))
    setups["(10,10)      NaN"] = setup % (10, str(True))
    setups["(100,100)       "] = setup % (100, str(False))
    setups["(100,100)    NaN"] = setup % (100, str(True))
    setups["(1000,1000)     "] = setup % (1000, str(False))
    setups["(1000,1000)  NaN"] = setup % (1000, str(True))
    run['setups'] = setups
    suite.append(run)
    
    # Strip leading spaces from setup code
    for i, run in enumerate(suite):
        for s in run['setups']:
            t = run['setups'][s]
            t = '\n'.join([z.strip() for z in t.split('\n')])
            suite[i]['setups'][s] = t
  
    return suite 
