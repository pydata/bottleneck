import bottleneck as bn
import numpy as np


RAND_ARRAY_CACHE = {}


def get_cached_rand_array(shape, dtype, order):
    key = (shape, dtype, order)
    if key not in RAND_ARRAY_CACHE:
        assert order in ["C", "F"]
        random_state = np.random.RandomState(1234)
        if "int" in shape:
            dtype_info = np.iinfo(dtype)
            arr = random_state.randint(
                dtype_info.min, dtype_info.max, size=shape, dtype=dtype
            )
        else:
            arr = 10000 * random_state.standard_normal(shape).astype(dtype)

        if order == "F":
            arr = np.asfortranarray(arr)

        assert arr.flags[order + "_CONTIGUOUS"]

        RAND_ARRAY_CACHE[key] = arr

    return RAND_ARRAY_CACHE[key].copy(order=order)


class Time1DReductions:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3,), (10 ** 5,), (10 ** 7,)],
    ]
    param_names = ["dtype", "shape"]

    def setup(self, dtype, shape):
        self.arr = get_cached_rand_array(shape, dtype, "C")

    def time_nanmin(self, dtype, shape):
        bn.nanmin(self.arr)

    def time_nanmax(self, dtype, shape):
        bn.nanmin(self.arr)

    def time_nanargmin(self, dtype, shape):
        bn.nanargmin(self.arr)

    def time_nanargmax(self, dtype, shape):
        bn.nanargmax(self.arr)

    def time_nansum(self, dtype, shape):
        bn.nansum(self.arr)

    def time_nanmean(self, dtype, shape):
        bn.nanmean(self.arr)

    def time_nanstd(self, dtype, shape):
        bn.nanstd(self.arr)

    def time_nanvar(self, dtype, shape):
        bn.nanvar(self.arr)

    def time_median(self, dtype, shape):
        bn.median(self.arr)

    def time_nanmedian(self, dtype, shape):
        bn.nanmedian(self.arr)

    def time_ss(self, dtype, shape):
        bn.ss(self.arr)


class Time2DReductions:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3, 10 ** 3)],
        ["C", "F"],
        [None, 0, 1],
    ]
    param_names = ["dtype", "shape", "order", "axis"]

    def setup(self, dtype, shape, order, axis):
        self.arr = get_cached_rand_array(shape, dtype, order)

    def time_nanmin(self, dtype, shape, order, axis):
        bn.nanmin(self.arr, axis=axis)

    def time_nanmax(self, dtype, shape, order, axis):
        bn.nanmin(self.arr, axis=axis)

    def time_nanargmin(self, dtype, shape, order, axis):
        bn.nanargmin(self.arr, axis=axis)

    def time_nanargmax(self, dtype, shape, order, axis):
        bn.nanargmax(self.arr, axis=axis)

    def time_nansum(self, dtype, shape, order, axis):
        bn.nansum(self.arr, axis=axis)

    def time_nanmean(self, dtype, shape, order, axis):
        bn.nanmean(self.arr, axis=axis)

    def time_nanstd(self, dtype, shape, order, axis):
        bn.nanstd(self.arr, axis=axis)

    def time_nanvar(self, dtype, shape, order, axis):
        bn.nanvar(self.arr, axis=axis)

    def time_median(self, dtype, shape, order, axis):
        bn.median(self.arr, axis=axis)

    def time_nanmedian(self, dtype, shape, order, axis):
        bn.nanmedian(self.arr, axis=axis)

    def time_ss(self, dtype, shape, order, axis):
        bn.ss(self.arr, axis=axis)


class TimeAnyNan2D:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3, 10 ** 3)],
        ["C", "F"],
        [None, 0, 1],
        ["fast", "slow"],
    ]
    param_names = ["dtype", "shape", "order", "axis", "case"]

    def setup(self, dtype, shape, order, axis, case):
        self.arr = np.full(shape, 0, dtype=dtype, order=order)

        if "float" in dtype:
            if case == "fast":
                self.arr[:] = np.nan

        assert self.arr.flags[order + "_CONTIGUOUS"]

    def time_anynan(self, dtype, shape, order, axis, case):
        bn.anynan(self.arr, axis=axis)


class TimeAllNan2D:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3, 10 ** 3)],
        ["C", "F"],
        [None, 0, 1],
        ["fast", "slow"],
    ]
    param_names = ["dtype", "shape", "order", "axis", "case"]

    def setup(self, dtype, shape, order, axis, case):
        self.arr = np.full(shape, 0, dtype=dtype, order=order)

        if "float" in dtype:
            if case == "slow":
                self.arr[:] = np.nan

        assert self.arr.flags[order + "_CONTIGUOUS"]

    def time_allnan(self, dtype, shape, order, axis, case):
        bn.allnan(self.arr, axis=axis)
