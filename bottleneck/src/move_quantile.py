from ..move import move_quantile as move_quantile_c
from collections.abc import Iterable
import numpy as np

def move_quantile(a, window, q, min_count=None, axis=-1):
    if not isinstance(q, Iterable):
        return move_quantile_c(a, window=window, min_count=min_count, axis=axis, q=q)
    result = np.asarray(
        [move_quantile_c(a=a, window=window, min_count=min_count, axis=axis, q=quantile) for quantile in q]
        )
    return result

move_quantile.__doc__ = move_quantile_c.__doc__