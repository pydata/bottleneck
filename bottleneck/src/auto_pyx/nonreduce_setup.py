"""
Use to convert nonreduce.pyx to a C file.

This setup.py is NOT used to install the Bottleneck package. The Bottleneck
setup.py file is bottleneck/setup.py

The C files are distributed with Bottleneck, so this file is only useful if
you modify nonreduce.pyx

To convert from cython to C:

$ cd bottleneck/bottleneck/src
$ python nonreduce_setup.py build_ext --inplace

"""

import os
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

name = 'nonreduce'
mod_dir = os.path.dirname(__file__)
ext_modules = [Extension(name, [os.path.join(mod_dir, name + ".pyx")],
               include_dirs=[np.get_include()])]

setup(
    name=name,
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)

os.rename(name + ".so", os.path.join(mod_dir, "../../" + name + ".so"))
