import sys

import numpy as np
import pytest

import bottleneck as bn


@pytest.mark.skipif(
    sys.platform.startswith("win"), reason="resource module not available on windows"
)
def test_memory_leak() -> None:
    import resource

    arr = np.arange(1).reshape((1, 1))

    n_attempts = 3
    results = []

    for _ in range(n_attempts):
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
        # For 1.3.0 release, this had value of ~100kB
        if diff_bytes:
            results.append(diff_bytes)
        else:
            break

    assert len(results) < n_attempts
