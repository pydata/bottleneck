
import numpy as np
import bottleneck as bn
from .autotimeit import autotimeit

__all__ = ['bench2']


def bench2(function='nansum'):
    "Bottleneck benchmark of C rewrite."

    tab = '    '

    # Header
    print('Bottleneck performance benchmark of C rewrite')
    print("%sBottleneck %s; Numpy %s" % (tab, bn.__version__, np.__version__))
    print("%sSpeed is Bottleneck Cython time divided by Bottleneck C time"
          % tab)
    print("")

    print("   Speed  Call                     Input")
    suite = benchsuite(function)
    for test in suite:
        name = test["name"]
        speed = timer(test['statements'], test['setup'])
        print("%7.2f   %s   %s" % (speed, name[0].ljust(22), name[1]))


def timer(statements, setup):
    if len(statements) != 2:
        raise ValueError("Two statements needed.")
    with np.errstate(invalid='ignore'):
        t0 = autotimeit(statements[0], setup)
        t1 = autotimeit(statements[1], setup)
    speed = t1 / t0
    return speed


def benchsuite(function):

    suite = []

    setup = """
        import numpy as np
        from bottleneck import %s
        from bottleneck import %s2 as %s_c
        a = np.ones(1)
        b = np.ones(1, dtype=np.float16)
        c = np.ones((2, 2))
        d = np.array(1)
    """ % (function, function, function)

    setup2 = """
        import numpy as np
        from bottleneck import %s
        from bottleneck import %s2 as %s_c
        e = np.random.rand(1000000)
    """ % (function, function, function)

    run = {}
    run['name'] = ["%s(a)" % function, "np.ones(1)"]
    run['statements'] = ["%s_c(a)" % function, "%s(a)" % function]
    run['setup'] = setup
    suite.append(run)

    run = {}
    run['name'] = ["%s(a, None)" % function, "np.ones(1)"]
    run['statements'] = ["%s_c(a, None)" % function, "%s(a, None)" % function]
    run['setup'] = setup
    suite.append(run)

    run = {}
    run['name'] = ["%s(a, 1)" % function, "np.ones((2, 2))"]
    run['statements'] = ["%s_c(c, 1)" % function, "%s(c, 1)" % function]
    run['setup'] = setup
    suite.append(run)

    run = {}
    run['name'] = ["%s(a, axis=None)" % function, "np.ones(1)"]
    run['statements'] = ["%s_c(a, axis=None)" % function, "%s(a, axis=None)" %
                         function]
    run['setup'] = setup
    suite.append(run)

    run = {}
    run['name'] = ["%s(a)" % function, "np.random.rand(1000000)"]
    run['statements'] = ["%s_c(e)" % function, "%s(e)" % function]
    run['setup'] = setup2
    suite.append(run)

    run = {}
    run['name'] = ["%s(a)" % function, "np.ones(1, dtype=np.float16)"]
    run['statements'] = ["%s_c(b)" % function, "%s(b)" % function]
    run['setup'] = setup
    suite.append(run)

    run = {}
    run['name'] = ["%s(a)" % function, "np.array(1)"]
    run['statements'] = ["%s_c(d)" % function, "%s(d)" % function]
    run['setup'] = setup
    suite.append(run)

    # Strip leading spaces from setup code
    for i, run in enumerate(suite):
        t = run['setup']
        t = '\n'.join([z.strip() for z in t.split('\n')])
        suite[i]['setup'] = t

    return suite
