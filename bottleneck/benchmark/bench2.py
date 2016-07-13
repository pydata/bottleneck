
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
        t0 = autotimeit(statements[0], setup, repeat=6)
        t1 = autotimeit(statements[1], setup, repeat=6)
    speed = t1 / t0
    return speed


def benchsuite(function):

    setup = """
        import numpy as np
        from bottleneck import %s
        from bottleneck import %s2 as %s_c
        a=%s
    """

    sig_array = [
                 ("%s%s(a, 1)", "np.ones((1, 1))"),
                 ("%s%s(a, 1)", "np.random.rand(1000000, 2)"),
                 ("%s%s(a)", "np.ones(1)"),
                 ("%s%s(a, None)", "np.ones(1)"),
                 ("%s%s(a, axis=None)", "np.ones(1)"),
                 ("%s%s(a)", "np.ones(1, dtype=np.float16)"),
                 ("%s%s(a)", "np.random.rand(1000000)"),
                 ("%s%s(a)", "np.array(1)"),
                ]

    f = function
    suite = []
    for signature, array in sig_array:
        run = {}
        run['name'] = [signature % (f, ''), array]
        run['statements'] = [signature % (f, "_c"), signature % (f, "")]
        run['setup'] = setup % (f, f, f, array)
        suite.append(run)

    # Strip leading spaces from setup code
    for i, run in enumerate(suite):
        t = run['setup']
        t = '\n'.join([z.strip() for z in t.split('\n')])
        suite[i]['setup'] = t

    return suite
