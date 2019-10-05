try:
    from .reduce import (
        nansum,
        nanmean,
        nanstd,
        nanvar,
        nanmin,
        nanmax,
        median,
        nanmedian,
        ss,
        nanargmin,
        nanargmax,
        anynan,
        allnan,
    )
    from .nonreduce import replace
    from .nonreduce_axis import partition, argpartition, rankdata, nanrankdata, push
    from .move import (
        move_sum,
        move_mean,
        move_std,
        move_var,
        move_min,
        move_max,
        move_argmin,
        move_argmax,
        move_median,
        move_rank,
    )

    from . import slow

except ImportError:
    raise ImportError(
        "bottleneck modules failed to import, likely due to a "
        "mismatch in NumPy versions. Either upgrade numpy to "
        "1.16+ or install with:\n\t"
        "pip install --no-build-isolation --no-cache-dir "
        "bottleneck"
    )

from bottleneck.benchmark.bench import bench
from bottleneck.benchmark.bench_detailed import bench_detailed
from bottleneck.tests.util import get_functions

from ._pytesttester import PytestTester

test = PytestTester(__name__)
del PytestTester

from ._version import get_versions  # noqa: E402

__version__ = get_versions()["version"]
del get_versions
