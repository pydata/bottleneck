from distutils.core import setup, Extension
import numpy as np

c_ext = Extension("nansum", ["reduce2.c"])

setup(
    ext_modules=[c_ext],
    include_dirs=np.get_include(),
)
