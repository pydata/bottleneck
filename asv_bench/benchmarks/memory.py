import numpy as np

import bottleneck as bn


class Memory:
    def peakmem_nanmedian(self):
        arr = np.arange(1).reshape((1, 1))
        for i in range(1000000):
            bn.nanmedian(arr)
