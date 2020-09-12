try:
    from . import slow
    from .move import (
        move_argmax,
        move_argmin,
        move_max,
        move_mean,
        move_median,
        move_min,
        move_rank,
        move_std,
        move_sum,
        move_var,
    )
    from .nonreduce import replace
    from .nonreduce_axis import argpartition, nanrankdata, partition, push, rankdata
    from .reduce import (
        allnan,
        anynan,
        median,
        nanargmax,
        nanargmin,
        nanmax,
        nanmean,
        nanmedian,
        nanmin,
        nanstd,
        nansum,
        nanvar,
        ss,
    )

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
from ._version import get_versions  # noqa: E402

test = PytestTester(__name__)
del PytestTester


__version__ = get_versions()["version"]  # type: ignore
del get_versions
