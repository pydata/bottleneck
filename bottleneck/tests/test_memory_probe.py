"""Check that the reducer memory probe distinguishes noise from real leaks."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

import bottleneck as bn
from bottleneck.tests import memory_test


@pytest.mark.thread_unsafe
@pytest.mark.skipif(sys.platform.startswith("win"), reason="requires resource")
def test_memory_probe_ignores_process_high_water_growth(monkeypatch):
    import resource

    samples = iter([100, 101])
    monkeypatch.setattr(
        resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=next(samples))
    )
    memory_test.test_memory_leak()


@pytest.mark.thread_unsafe
@pytest.mark.skipif(sys.platform.startswith("win"), reason="requires resource")
def test_memory_probe_detects_retained_arrays(monkeypatch):
    import resource

    monkeypatch.setattr(resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=100))
    original = bn.nansum
    retained = []

    def leaking_nansum(arr, axis=None):
        retained.append(np.empty(1024))
        return original(arr, axis=axis)

    monkeypatch.setattr(bn, "nansum", leaking_nansum)
    with pytest.raises(AssertionError):
        memory_test.test_memory_leak()
