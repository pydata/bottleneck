"""
Use to convert nansu,.pyx to a C file.

This setup.py is NOT used to install the DSNA package. The DSNA setup.py
file is dsna/setup.py

The C files are distributed with DSNA, so this file is only useful if you
modify nansum.pyx.

To convert from cython to C:

$ cd dsna/src
$ python setup.py build_ext --inplace

"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("func", ["func/func.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
  name = 'func',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

