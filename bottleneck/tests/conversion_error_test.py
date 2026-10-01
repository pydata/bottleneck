"""Preserve errors raised while converting integer arguments."""

import numpy as np
import pytest

import bottleneck as bn

CASES = [
    (bn.move_sum, {"window": 2}, "window"),
    (bn.move_sum, {"window": 2}, "min_count"),
    (bn.move_sum, {"window": 2}, "axis"),
    (bn.move_var, {"window": 2}, "ddof"),
    (bn.nansum, {}, "axis"),
    (bn.nanvar, {}, "ddof"),
    (bn.rankdata, {}, "axis"),
    (bn.push, {}, "n"),
    (bn.partition, {}, "kth"),
]


class BadIndex:
    def __init__(self, error):
        self.error = error

    def __index__(self):
        raise self.error


@pytest.mark.parametrize("func,kwargs,argument", CASES)
@pytest.mark.parametrize("value", [2**100, -(2**100)])
def test_integer_overflow(func, kwargs, argument, value):
    with pytest.raises(OverflowError):
        func(np.array([1.0, 2.0, 3.0]), **{**kwargs, argument: value})


@pytest.mark.parametrize("func,kwargs,argument", CASES)
def test_index_error_is_preserved(func, kwargs, argument):
    error = RuntimeError("custom index failure")
    with pytest.raises(RuntimeError, match="custom index failure") as caught:
        func(np.array([1.0, 2.0, 3.0]), **{**kwargs, argument: BadIndex(error)})
    assert caught.value is error


@pytest.mark.parametrize("func,kwargs,argument", CASES)
def test_noninteger_still_raises_type_error(func, kwargs, argument):
    with pytest.raises(TypeError):
        func(np.array([1.0, 2.0, 3.0]), **{**kwargs, argument: "invalid"})
