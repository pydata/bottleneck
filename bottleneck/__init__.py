# flak8: noqa

from . import slow

# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import Bottleneck so that you can
# rebuild using the templates. So try to import the compiled Bottleneck
# functions to the top level, but move on if not successful.
try:
    from .reduce import nansum
    from .reduce2 import nansum as nansum2
    from .reduce3 import nansum as nansum3
except:
    pass

from bottleneck.version import __version__
from bottleneck.benchmark.bench import bench

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print("No Bottleneck unit testing available.")
