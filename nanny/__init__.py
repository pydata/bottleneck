
from func import nansum, nanmax, nanmin, nanmean, nanstd
from nanny.version import __version__
from nanny.bench.bench import *

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No Nanny unit testing available."
