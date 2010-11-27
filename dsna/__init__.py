
from func import sum, max, min, mean, std, var
from move import move_sum
from group import group_mean

from dsna.version import __version__
from dsna.bench.bench import *

try:
    from numpy.testing import Tester
    test = Tester().test
    del Tester
except (ImportError, ValueError):
    print "No dsna unit testing available."
