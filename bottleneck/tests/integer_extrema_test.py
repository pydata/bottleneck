import numpy as np
import pytest
from numpy.testing import assert_equal

import bottleneck as bn


@pytest.mark.parametrize("name", ["move_min", "move_max", "move_argmin", "move_argmax"])
@pytest.mark.parametrize("offset", [0, 2**53, -(2**53), 2**63 - 16, -(2**63) + 16])
def test_integer_extrema_preserve_comparisons(name, offset):
    """Distinct integers must stay distinct until the output is cast to float."""
    values = np.array([3, 2, 1, 4, 4, 2, 5, 0, 1, 2], dtype=np.int64) + offset
    for data, axis in [
        (values, 0),
        (values[::-1], 0),
        (np.vstack([values, values[::-1]]), 1),
    ]:
        for window in [1, 2, 4, 10]:
            for min_count in [1, window]:
                actual = getattr(bn, name)(data, window, min_count=min_count, axis=axis)
                expected = getattr(bn.slow, name)(
                    data, window, min_count=min_count, axis=axis
                )
                assert_equal(actual, expected)


def test_moving_integer_extrema_keep_rightmost_ties():
    """Real ties use the rightmost value even at the limits of int64."""
    for value in [np.iinfo(np.int64).min, np.iinfo(np.int64).max]:
        values = np.full(8, value, dtype=np.int64)
        for func in [bn.move_argmin, bn.move_argmax]:
            assert_equal(func(values, 4, min_count=1), np.zeros(8))
