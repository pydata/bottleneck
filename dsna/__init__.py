
from func import nansum, nanmax, nanmin, nanmean, nanstd, nanvar
from dsna.version import __version__
from dsna.bench.bench import *

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No Nanny unit testing available."
