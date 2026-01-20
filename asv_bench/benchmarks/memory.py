import bottleneck as bn
import numpy as np


class Memory:
    def peakmem_nanmedian(self):
        arr = np.arange(1).reshape((1, 1))
        for _ in range(1_000_000):
            bn.nanmedian(arr)
