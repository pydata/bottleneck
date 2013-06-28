"""
Use to convert func.pyx to a C file.

This setup.py is NOT used to install the Bottleneck package. The Bottleneck
setup.py file is bottleneck/setup.py

The C files are distributed with Bottleneck, so this file is only useful if
you modify nansum.pyx or nanstd.pyx or ...

To convert from cython to C:

$ cd bottleneck/bottleneck/src
$ python func/setup.py build_ext --inplace

"""

import os
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

mod_dir = os.path.dirname(__file__)
ext_modules = [Extension("func", [os.path.join(mod_dir, "func.pyx")],
               include_dirs=[np.get_include()])]

setup(
    name='func',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

os.rename("func.so", os.path.join(mod_dir, "../../func.so"))
