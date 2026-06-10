import sys

import numpy as np
import pytest

import bottleneck as bn


@pytest.mark.thread_unsafe
@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="resource module not available on windows"
)
def test_memory_leak():
    import resource

    arr = np.arange(1).reshape((1, 1))

    starting = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    for _ in range(1000):
        for axis in [None, 0, 1]:
            bn.nansum(arr, axis=axis)
            bn.nanargmax(arr, axis=axis)
            bn.nanargmin(arr, axis=axis)
            bn.nanmedian(arr, axis=axis)
            bn.nansum(arr, axis=axis)
            bn.nanmean(arr, axis=axis)
            bn.nanmin(arr, axis=axis)
            bn.nanmax(arr, axis=axis)
            bn.nanvar(arr, axis=axis)

    ending = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    diff = ending - starting
    diff_bytes = diff * resource.getpagesize()
    print(diff_bytes)
    # For 1.3.0 release, this had value of ~100kB
    assert diff_bytes == 0


def test_refcount_leak():
    # see https://github.com/pydata/bottleneck/issues/521
    rs = np.random.RandomState(0)
    arr = np.asfortranarray(rs.rand(8, 8))
    start_rc = sys.getrefcount(arr)

    for _ in range(1000):
        bn.rankdata(arr)

    assert sys.getrefcount(arr) == start_rc


@pytest.mark.thread_unsafe
@pytest.mark.parametrize(
    "func, arr",
    [
        pytest.param(bn.nanmin, np.zeros((4, 0), dtype=np.float64), id="nanmin-empty"),
        pytest.param(bn.nanmax, np.zeros((4, 0), dtype=np.int64), id="nanmax-empty"),
        pytest.param(
            bn.nanargmin, np.zeros((4, 0), dtype=np.float64), id="nanargmin-empty"
        ),
        pytest.param(
            bn.nanargmax, np.zeros((4, 0), dtype=np.int64), id="nanargmax-empty"
        ),
        pytest.param(
            bn.nanargmin,
            np.full((4, 2), np.nan, dtype=np.float64),
            id="nanargmin-allnan",
        ),
        pytest.param(
            bn.nanargmax,
            np.full((4, 2), np.nan, dtype=np.float64),
            id="nanargmax-allnan",
        ),
    ],
)
def test_reducer_error_path_leak(func, arr):
    # The single-axis reducers build their output array before checking for an
    # empty reduction axis or an all-NaN slice, then raised without releasing
    # it, leaking one output array per call. A dtype-refcount probe catches this
    # on 3.10-3.12 but not on 3.13+, where the builtin dtype singletons are
    # immortal and their refcount no longer moves, so measure the net allocation
    # directly with tracemalloc instead.
    import gc
    import tracemalloc

    def hammer(rounds):
        for _ in range(rounds):
            with pytest.raises(ValueError):
                func(arr, axis=1)

    hammer(50)  # warm up any one-time caches before sampling
    gc.collect()

    tracemalloc.start()
    before = tracemalloc.take_snapshot()
    rounds = 400
    hammer(rounds)
    gc.collect()
    after = tracemalloc.take_snapshot()
    tracemalloc.stop()

    grew = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    # Each leaked output array is ~140 bytes, so the unfixed code grows by
    # rounds * ~140 bytes here; the fix keeps this near zero. The budget is
    # 16 bytes/call for allocator noise that scales with the loop, plus a
    # fixed 16 KiB because free-threaded builds retain ~7 KiB of
    # interpreter-internal allocations per tracemalloc window regardless of
    # call count (gh-574), which a purely per-call budget cannot absorb.
    assert grew < rounds * 16 + 16 * 1024
