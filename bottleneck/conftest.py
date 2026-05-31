from importlib.util import find_spec

import pytest

HAS_PYTEST_PARALLEL_RUN = find_spec("pytest_run_parallel") is not None


def pytest_configure(config):
    if HAS_PYTEST_PARALLEL_RUN:
        return

    config.addinivalue_line(
        "markers",
        "parallel_threads(n): run the given test function in parallel "
        "using `n` threads.",
    )
    config.addinivalue_line(
        "markers",
        "thread_unsafe: mark the test function as single-threaded",
    )
    config.addinivalue_line(
        "markers",
        "iterations(n): run the given test function `n` times in each thread",
    )


if not HAS_PYTEST_PARALLEL_RUN:

    @pytest.fixture
    def num_parallel_threads():
        return 1
