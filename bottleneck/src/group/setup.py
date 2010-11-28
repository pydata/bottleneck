"""
Use to convert move.pyx to a C file.

This setup.py is NOT used to install the Bottleneck package. The Bottleneck
setup.py file is bottleneck/setup.py

The C files are distributed with bottleneck, so this file is only useful if
you modify group_sum.pyx or group_std.pyx or ...

To convert from cython to C:

$ cd bottleneck/bottleneck/src
$ python group/setup.py build_ext --inplace

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("group", ["group/group.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  name = 'move',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

