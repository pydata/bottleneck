import bottleneck as bn
import numpy as np


class TimeReplace2D:
    params = [
        ["int32", "int64", "float32", "float64"],
        [(10 ** 3, 10 ** 3)],
        ["C", "F"],
    ]
    param_names = ["dtype", "shape", "order"]

    def setup(self, dtype, shape, order):
        self.arr = np.full(shape, 0, dtype=dtype, order=order)

        assert self.arr.flags[order + "_CONTIGUOUS"]

        self.old = 0
        self.new = 1

    def time_replace(self, dtype, shape, order):
        bn.replace(self.arr, self.old, self.new)
