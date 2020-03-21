"""Check that functions can handle scalar input"""

from typing import Callable, Union

import hypothesis
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from hypothesis.strategies import integers, floats, one_of

import bottleneck as bn  # noqa: F401

from .util import get_functions

int64_iinfo = np.iinfo(np.int64)


scalars = one_of(
    [integers(min_value=int64_iinfo.min, max_value=int64_iinfo.max), floats()]
)


@hypothesis.given(scalar=scalars)
@pytest.mark.parametrize(
    "func",
    get_functions("reduce") + get_functions("nonreduce_axis"),
    ids=lambda x: x.__name__,
)
def test_scalar_input(
    func: Callable[[np.array], Union[int, float, np.array]], scalar: Union[int, float]
) -> None:
    """Test that bn.xxx gives the same output as bn.slow.xxx for scalar input."""
    if func.__name__ in ("partition", "argpartition", "push"):
        return
    func0 = eval("bn.slow.%s" % func.__name__)
    msg = "\nfunc %s | input %s\n"
    actual_raised = False
    desired_raised = False

    try:
        actual = func(scalar)
    except ValueError:
        actual_raised = True

    try:
        desired = func0(scalar)
    except ValueError:
        desired_raised = True

    if desired_raised or actual_raised:
        assert actual_raised and desired_raised
    else:
        err_msg = msg % (func.__name__, scalar)
        assert_array_almost_equal(actual, desired, err_msg=err_msg)
