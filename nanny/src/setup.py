"""
Use to convert nansu,.pyx to a C file.

This setup.py is NOT used to install the Nanny package. The Nanny setup.py
file is nanny/setup.py

The C files are distributed with Nanny, so this file is only useful if you
modify nansum.pyx.

To convert from cython to C:

$ cd nany/src
$ python setup.py build_ext --inplace

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("func", ["func.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  name = 'func',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

