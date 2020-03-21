import warnings
from typing import Optional, Union

import numpy as np
from numpy import nanmean, nansum

__all__ = [
    "median",
    "nanmedian",
    "nansum",
    "nanmean",
    "nanvar",
    "nanstd",
    "nanmin",
    "nanmax",
    "nanargmin",
    "nanargmax",
    "ss",
    "anynan",
    "allnan",
]


def nanargmin(a: np.ndarray, axis: Optional[int] = None) -> Union[int, np.ndarray]:
    "Slow nanargmin function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmin(a, axis=axis)


def nanargmax(a: np.ndarray, axis: Optional[int] = None) -> Union[int, np.ndarray]:
    "Slow nanargmax function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanargmax(a, axis=axis)


def nanvar(
    a: np.ndarray, axis: Optional[int] = None, ddof: int = 0
) -> Union[float, np.ndarray]:
    "Slow nanvar function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanvar(a, axis=axis, ddof=ddof)


def nanstd(
    a: np.ndarray, axis: Optional[int] = None, ddof: int = 0
) -> Union[float, np.ndarray]:
    "Slow nanstd function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(a, axis=axis, ddof=ddof)


def nanmin(a: np.ndarray, axis: Optional[int] = None) -> Union[int, float, np.ndarray]:
    "Slow nanmin function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmin(a, axis=axis)


def nanmax(a: np.ndarray, axis: Optional[int] = None) -> Union[int, float, np.ndarray]:
    "Slow nanmax function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmax(a, axis=axis)


def median(a: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    "Slow median function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.median(a, axis=axis)


def nanmedian(a: np.ndarray, axis: Optional[int] = None) -> Union[float, np.ndarray]:
    "Slow nanmedian function used for unaccelerated dtypes."
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmedian(a, axis=axis)


def ss(a: np.ndarray, axis: Optional[int] = None) -> Union[int, float, np.ndarray]:
    "Slow sum of squares used for unaccelerated dtypes."
    a = np.asarray(a)
    y = np.multiply(a, a).sum(axis)
    return y


def anynan(a: np.ndarray, axis: Optional[int] = None) -> Union[bool, np.ndarray]:
    "Slow check for Nans used for unaccelerated dtypes."
    return np.isnan(a).any(axis)


def allnan(a: np.ndarray, axis: Optional[int] = None) -> Union[bool, np.ndarray]:
    "Slow check for all Nans used for unaccelerated dtypes."
    return np.isnan(a).all(axis)
