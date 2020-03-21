"""Test functions."""

import warnings

import numpy as np
from numpy.testing import assert_equal

from .util import hy_array_gen, get_functions


@pytest.mark.parametrize("func", get_functions("all"), ids=lambda x: x.__name__)
@hypothesis.given(array=hy_array_gen)
def test_modification(func, array):
    """Test that bn.xxx gives the same output as np.xxx."""
    name = func.__name__
    if name == "replace":
        return
    msg = "\nInput array modified by %s.\n\n"
    msg += "input array before:\n%s\nafter:\n%s\n"
    axes: List[Optional[int]] = list(range(-array.ndim, array.ndim))
    if all(x not in name for x in ["push", "move", "sort", "partition"]):
        axes += [None]

    second_arg = 1
    if "partition" in name:
        second_arg = 0

    for axis in axes:
        with np.errstate(invalid="ignore"):
            a1 = array.copy()
            a2 = array.copy()
            if any(x in name for x in ["move", "sort", "partition"]):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    func(a1, second_arg, axis=axis)
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        func(a1, axis=axis)
                except ValueError as e:
                    if name.startswith("nanarg") and "All-NaN slice encountered" in str(
                        e
                    ):
                        continue
            assert_equal(a1, a2, msg % (name, a1, a2))
