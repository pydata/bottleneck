"""
Use to convert move.pyx to a C file.

This setup.py is NOT used to install the Bottleneck package. The Bottleneck
setup.py file is bottleneck/setup.py

The C files are distributed with bottleneck, so this file is only useful if
you modify move_nansum.pyx or move_nanstd.pyx or ...

To convert from cython to C:

$ cd bottleneck/bottleneck/src
$ python move/setup.py build_ext --inplace

"""

import os
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

mod_dir = os.path.dirname(__file__)
ext_modules = [Extension("move", [os.path.join(mod_dir, "move.pyx")],
               extra_compile_args=["-std=gnu89"],
               include_dirs=[np.get_include()])]

setup(
    name='move',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

os.rename("move.so", os.path.join(mod_dir, "../../move.so"))
