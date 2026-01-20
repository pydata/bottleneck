"""Check that functions can handle scalar input"""

import pytest
from numpy.testing import assert_array_almost_equal

import bottleneck as bn


@pytest.mark.parametrize(
    "func",
    bn.get_functions("reduce") + bn.get_functions("nonreduce_axis"),
    ids=lambda x: x.__name__,
)
def test_scalar_input(func, args=tuple()):
    """Test that bn.xxx gives the same output as bn.slow.xxx for scalar input."""
    if func.__name__ in ("partition", "argpartition", "push"):
        return
    func0 = eval(f"bn.slow.{func.__name__}")
    msg = "\nfunc %s | input %s\n"
    a = -9
    argsi = [a] + list(args)
    actual = func(*argsi)
    desired = func0(*argsi)
    err_msg = msg % (func.__name__, a)
    assert_array_almost_equal(actual, desired, err_msg=err_msg)
