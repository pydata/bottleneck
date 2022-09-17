import bottleneck as bn
from .reduce import get_cached_rand_array


class Time1DMove:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3,), (10 ** 5,), (10 ** 7,)],
        [10],
    ]
    param_names = ["dtype", "shape", "window"]

    def setup(self, dtype, shape, window):
        self.arr = get_cached_rand_array(shape, dtype, "C")

    def time_move_sum(self, dtype, shape, window):
        bn.move_sum(self.arr, window)

    def time_move_mean(self, dtype, shape, window):
        bn.move_mean(self.arr, window)

    def time_move_std(self, dtype, shape, window):
        bn.move_std(self.arr, window)

    def time_move_var(self, dtype, shape, window):
        bn.move_var(self.arr, window)

    def time_move_min(self, dtype, shape, window):
        bn.move_min(self.arr, window)

    def time_move_max(self, dtype, shape, window):
        bn.move_max(self.arr, window)

    def time_move_argmin(self, dtype, shape, window):
        bn.move_argmin(self.arr, window)

    def time_move_argmax(self, dtype, shape, window):
        bn.move_argmax(self.arr, window)

    def time_move_median(self, dtype, shape, window):
        bn.move_median(self.arr, window)

    def time_move_quantile(self, dtype, shape, window):
        bn.move_quantile(self.arr, window)

    def time_move_rank(self, dtype, shape, window):
        bn.move_rank(self.arr, window)


class Time2DMove:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3, 10 ** 3)],
        ["C", "F"],
        [0, 1],
        [10],
    ]
    param_names = ["dtype", "shape", "order", "axis", "window"]

    def setup(self, dtype, shape, order, axis, window):
        self.arr = get_cached_rand_array(shape, dtype, order)

    def time_move_sum(self, dtype, shape, order, axis, window):
        bn.move_sum(self.arr, window, axis=axis)

    def time_move_mean(self, dtype, shape, order, axis, window):
        bn.move_mean(self.arr, window, axis=axis)

    def time_move_std(self, dtype, shape, order, axis, window):
        bn.move_std(self.arr, window, axis=axis)

    def time_move_var(self, dtype, shape, order, axis, window):
        bn.move_var(self.arr, window, axis=axis)

    def time_move_min(self, dtype, shape, order, axis, window):
        bn.move_min(self.arr, window, axis=axis)

    def time_move_max(self, dtype, shape, order, axis, window):
        bn.move_max(self.arr, window, axis=axis)

    def time_move_argmin(self, dtype, shape, order, axis, window):
        bn.move_argmin(self.arr, window, axis=axis)

    def time_move_argmax(self, dtype, shape, order, axis, window):
        bn.move_argmax(self.arr, window, axis=axis)

    def time_move_median(self, dtype, shape, order, axis, window):
        bn.move_median(self.arr, window, axis=axis)
        
    def time_move_quantile(self, dtype, shape, order, axis, window):
        bn.move_quantile(self.arr, window, axis=axis)

    def time_move_rank(self, dtype, shape, order, axis, window):
        bn.move_rank(self.arr, window, axis=axis)
