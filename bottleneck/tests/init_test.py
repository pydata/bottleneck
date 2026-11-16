"""Exercise failed extension initialization in a fresh interpreter."""

import importlib
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("name", ["move", "reduce", "nonreduce", "nonreduce_axis"])
@pytest.mark.parametrize("failure", ["missing", "invalid"])
def test_failed_numpy_import_does_not_leak_module(name, failure):
    extension = importlib.import_module(f"bottleneck.{name}")
    script = textwrap.dedent(
        """
        import gc
        import importlib.util
        import sys
        import types

        import numpy as np

        name, path, failure = sys.argv[1:]
        api = np._core._multiarray_umath
        original = api._ARRAY_API
        if failure == "missing":
            del api._ARRAY_API
        else:
            api._ARRAY_API = None

        def count_modules():
            gc.collect()
            return sum(
                type(obj) is types.ModuleType
                and obj.__name__ in (name, "bottleneck." + name)
                for obj in gc.get_objects()
            )

        before = count_modules()
        for _ in range(3):
            spec = importlib.util.spec_from_file_location("bottleneck." + name, path)
            try:
                importlib.util.module_from_spec(spec)
            except ImportError:
                pass
            else:
                raise AssertionError("extension initialization should fail")

        api._ARRAY_API = original
        after = count_modules()
        assert after == before, (name, before, after)

        # A failed initialization must not prevent a subsequent valid import.
        module = importlib.util.module_from_spec(spec)
        assert module.__name__ == "bottleneck." + name
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, name, extension.__file__, failure],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
