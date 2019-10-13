import bottleneck as bn
from .reduce import get_cached_rand_array


class Time1DNonreduceAxis:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3,), (10 ** 5,), (10 ** 7,)],
    ]
    param_names = ["dtype", "shape"]

    def setup(self, dtype, shape):
        self.arr = get_cached_rand_array(shape, dtype, "C")
        self.half = shape[0] // 2

    def time_partition(self, dtype, shape):
        bn.partition(self.arr, self.half)

    def time_argpartition(self, dtype, shape):
        bn.argpartition(self.arr, self.half)

    def time_rankdata(self, dtype, shape):
        bn.rankdata(self.arr)

    def time_nanrankdata(self, dtype, shape):
        bn.nanrankdata(self.arr)

    def time_push(self, dtype, shape):
        bn.push(self.arr)
