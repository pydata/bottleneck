"Generate all pyx files from templates"

import numpy as np

from bottleneck.src.template.func.func import funcpyx
from bottleneck.src.template.move.move import movepyx

def makepyx(bits=None):
    if bits is None:
        if np.int_ == np.int32:
            bits = 32
        elif np.int_ == np.int64:
            bits = 64
    if bits not in (32, 64):        
        raise RuntimeError("`bits` must be 32 or 64")
    funcpyx(bits=bits)
    movepyx(bits=bits)
