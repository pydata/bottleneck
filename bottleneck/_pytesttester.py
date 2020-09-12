"""
Generic test utilities.

Based on scipy._libs._testutils
"""

import os
import sys
from typing import List, Optional

__all__ = ["PytestTester"]


class PytestTester(object):
    """
    Pytest test runner entry point.
    """

    def __init__(self, module_name: str) -> None:
        self.module_name = module_name

    def __call__(
        self,
        label: str = "fast",
        verbose: int = 1,
        extra_argv: Optional[List[str]] = None,
        doctests: bool = False,
        coverage: bool = False,
        tests: Optional[List[str]] = None,
        parallel: Optional[int] = None,
    ) -> bool:
        import pytest

        module = sys.modules[self.module_name]
        module_path = os.path.abspath(module.__path__[0])

        pytest_args = ["-l"]

        if doctests:
            raise ValueError("Doctests not supported")

        if extra_argv:
            pytest_args += list(extra_argv)

        if verbose and int(verbose) > 1:
            pytest_args += ["-" + "v" * (int(verbose) - 1)]

        if coverage:
            pytest_args += ["--cov=" + module_path]

        if label == "fast":
            pytest_args += ["-m", "not slow"]
        elif label != "full":
            pytest_args += ["-m", label]

        if tests is None:
            tests = [self.module_name]

        if parallel is not None and parallel > 1:
            if _pytest_has_xdist():
                pytest_args += ["-n", str(parallel)]
            else:
                import warnings

                warnings.warn(
                    "Could not run tests in parallel because "
                    "pytest-xdist plugin is not available."
                )

        pytest_args += ["--pyargs"] + list(tests)

        if not _have_test_extras():
            warnings.warn(
                "The pytest and/or hypothesis packages are not installed, install with "
                "pip install bottleneck[test]"
            )

        try:
            code = pytest.main(pytest_args)
        except SystemExit as exc:
            code = exc.code

        return code == 0


def _pytest_has_xdist() -> bool:
    """
    Check if the pytest-xdist plugin is installed, providing parallel tests
    """
    # Check xdist exists without importing, otherwise pytests emits warnings
    from importlib.util import find_spec

    return find_spec("xdist") is not None


def _have_test_extras() -> bool:
    """
    Check if the test extra is installed
    """
    from importlib.util import find_spec

    return all(find_spec(x) is not None for x in ["pytest", "hypothesis"])
