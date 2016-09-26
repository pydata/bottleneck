
import numpy as np
import bottleneck as bn
from .autotimeit import autotimeit

__all__ = ['bench_detailed']


def bench_detailed(function='nansum', fraction_nan=0.0):
    """
    Benchmark a single function.

    Parameters
    ----------
    function : str, optional
        Name of function, as a string, to benchmark. Default ('nansum') is
        to benchmark bn.nansum.
    fraction_nan : float, optional
        Fraction of array elements that should, on average, be NaN. The
        default (0.0) is not to set any elements to NaN.

    Returns
    -------
    A benchmark report is printed to stdout.

    """

    if fraction_nan < 0 or fraction_nan > 1:
        raise ValueError("`fraction_nan` must be between 0 and 1, inclusive")

    tab = '    '

    # Header
    print('%s benchmark' % function)
    print("%sBottleneck %s; Numpy %s" % (tab, bn.__version__, np.__version__))
    print("%sSpeed is NumPy time divided by Bottleneck time" % tab)
    if fraction_nan == 0:
        print("%sNone of the array elements are NaN" % tab)
    else:
        print("%s%.1f%% of the array elements are NaN (on average)"
              % (tab, fraction_nan * 100))
    print("")

    print("   Speed  Call                     Array")
    suite = benchsuite(function, fraction_nan)
    for test in suite:
        name = test["name"]
        speed = timer(test['statements'], test['setup'], test['repeat'])
        print("%8.1f  %s   %s" % (speed, name[0].ljust(22), name[1]))


def timer(statements, setup, repeat):
    if len(statements) != 2:
        raise ValueError("Two statements needed.")
    with np.errstate(invalid='ignore'):
        t0 = autotimeit(statements[0], setup, repeat=repeat)
        t1 = autotimeit(statements[1], setup, repeat=repeat)
    speed = t1 / t0
    return speed


def benchsuite(function, fraction_nan):

    repeat_array_sig = [

     (10, "rand(1)",             "(a)",    "(a, 1)",         "(a, np.nan, 0)"),
     (10, "rand(10)",            "(a)",    "(a, 2)",         "(a, np.nan, 0)"),
     (6,  "rand(100)",           "(a)",    "(a, 20)",        "(a, np.nan, 0)"),
     (3,  "rand(1000)",          "(a)",    "(a, 200)",       "(a, np.nan, 0)"),
     (2,  "rand(1000000)",       "(a)",    "(a, 200)",       "(a, np.nan, 0)"),

     (6,  "rand(10, 10)",        "(a)",    "(a, 2)",         "(a, np.nan, 0)"),
     (3,  "rand(100, 100)",      "(a)",    "(a, 20)",        "(a, np.nan, 0)"),
     (2,  "rand(1000, 1000)",    "(a)",    "(a, 200)",       "(a, np.nan, 0)"),

     (6,  "rand(10, 10)",        "(a, 1)", None,             None),
     (3,  "rand(100, 100)",      "(a, 1)", None,             None),
     (3,  "rand(1000, 1000)",    "(a, 1)", None,             None),
     (2,  "rand(100000, 2)",     "(a, 1)", "(a, 2)",         None),

     (6,  "rand(10, 10)",        "(a, 0)", None,             None),
     (3,  "rand(100, 100)",      "(a, 0)", None,             None),
     (2,  "rand(1000, 1000)",    "(a, 0)", None,             None),

     (2,  "rand(100, 100, 100)", "(a, 0)", None,             None),
     (2,  "rand(100, 100, 100)", "(a, 1)", None,             None),
     (2,  "rand(100, 100, 100)", "(a, 2)", "(a, 20)",        "(a, np.nan, 0)"),

     (10, "array(1.0)",          "(a)",    None,             "(a, 0, 2)"),

     ]

    # what kind of function signature do we need to use?
    if function in bn.get_functions('reduce', as_string=True):
        index = 0
    elif function in bn.get_functions('move', as_string=True):
        index = 1
    elif function in ['rankdata', 'nanrankdata']:
        index = 0
    elif function in ['partition', 'argpartition', 'push']:
        index = 1
    elif function == 'replace':
        index = 2
    else:
        raise ValueError("`function` (%s) not recognized" % function)

    setup = """
        import numpy as np
        from bottleneck import %s as bn_fn
        try: from numpy import %s as sl_fn
        except ImportError: from bottleneck.slow import %s as sl_fn
        if "%s" == "median": from bottleneck.slow import median as sl_fn
        if "%s" == "nanmedian": from bottleneck.slow import nanmedian as sl_fn
        from numpy.random import rand
        from numpy import array
        a = %s
        if %s != 0: a[a < %s] = np.nan
    """
    setup = '\n'.join([s.strip() for s in setup.split('\n')])

    # create benchmark suite
    f = function
    suite = []
    for instructions in repeat_array_sig:
        signature = instructions[index + 2]
        if signature is None:
            continue
        repeat = instructions[0]
        array = instructions[1]
        run = {}
        run['name'] = [f + signature, array]
        run['statements'] = ["bn_fn" + signature, "sl_fn" + signature]
        run['setup'] = setup % (f, f, f, f, f, array,
                                fraction_nan, fraction_nan)
        run['repeat'] = repeat
        suite.append(run)

    return suite
