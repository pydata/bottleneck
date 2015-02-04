import numpy as np

__all__ = ['replace']


def replace(arr, old, new):
    "Slow replace (inplace) used for unaccelerated ndim/dtype combinations."
    if type(arr) is not np.ndarray:
        raise TypeError("`arr` must be a numpy array.")
    if not issubclass(arr.dtype.type, np.inexact):
        if old != old:
            # int arrays do not contain NaN
            return
        if int(old) != old:
            raise ValueError("Cannot safely cast `old` to int.")
        if int(new) != new:
            raise ValueError("Cannot safely cast `new` to int.")
    if old != old:
        mask = np.isnan(arr)
    else:
        mask = arr == old
    np.putmask(arr, mask, new)
