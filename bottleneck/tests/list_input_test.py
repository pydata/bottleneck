"""Check that functions can handle list input"""

import warnings

import hypothesis
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

import bottleneck as bn
from .util import DTYPES, get_functions, hy_lists


def lists(dtypes=DTYPES):
    """Iterator that yields lists to use for unit testing."""
    ss = {}
    ss[1] = {"size": 4, "shapes": [(4,)]}
    ss[2] = {"size": 6, "shapes": [(1, 6), (2, 3)]}
    ss[3] = {"size": 6, "shapes": [(1, 2, 3)]}
    ss[4] = {"size": 24, "shapes": [(1, 2, 3, 4)]}
    for ndim in ss:
        size = ss[ndim]["size"]
        shapes = ss[ndim]["shapes"]
        a = np.arange(size)
        for shape in shapes:
            a = a.reshape(shape)
            for dtype in dtypes:
                yield a.astype(dtype).tolist()


@hypothesis.given(input_list=hy_lists())
@pytest.mark.parametrize(
    "func", get_functions("all"), ids=lambda x: x.__name__,
)
def test_list_input(func, input_list) -> None:
    """Test that bn.xxx gives the same output as bn.slow.xxx for list input."""
    msg = "\nfunc %s | input %s (%s) | shape %s\n"
    msg += "\nInput array:\n%s\n"
    name = func.__name__
    if name == "replace":
        return
    func0 = eval("bn.slow.%s" % name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        actual_raised = False
        desired_raised = False

        try:
            if any(x in func.__name__ for x in ["move", "partition"]):
                actual = func(input_list, 2)
            else:
                actual = func(input_list)
        except ValueError:
            actual_raised = True

        try:
            if any(x in func.__name__ for x in ["move", "partition"]):
                desired = func0(input_list, 2)
            else:
                desired = func0(input_list)
        except ValueError:
            desired_raised = True

    if actual_raised and desired_raised:
        return

    assert not (actual_raised or desired_raised)

    a = np.array(input_list)
    tup = (name, "a", str(a.dtype), str(a.shape), a)
    err_msg = msg % tup
    assert_array_almost_equal(actual, desired, err_msg=err_msg)
