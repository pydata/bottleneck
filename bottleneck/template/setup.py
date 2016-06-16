from distutils.core import setup, Extension
import numpy as np

c_ext = Extension("nansum", ["reduce2.c"], extra_compile_args=[])

setup(
    ext_modules=[c_ext],
    include_dirs=np.get_include(),
)
