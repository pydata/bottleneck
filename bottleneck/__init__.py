# flak8: noqa

from . import slow

# If you bork the build (e.g. by messing around with the templates),
# you still want to be able to import Bottleneck so that you can
# rebuild using the templates. So try to import the compiled Bottleneck
# functions to the top level, but move on if not successful.
try:
    from .reduce import (nansum, nanmean, nanstd, nanvar, nanmin, nanmax,
                         median, nanmedian, ss, nanargmin, nanargmax, anynan,
                         allnan)
except:
    pass
try:
    from .nonreduce import replace
except:
    pass
try:
    from .nonreduce_axis import partsort, argpartsort, rankdata, nanrankdata
except:
    pass
try:
    from .move import (move_sum, move_mean, move_std, move_min,
                       move_max, move_median)
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
