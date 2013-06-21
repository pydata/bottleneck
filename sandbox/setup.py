"""
Use to convert *.pyx in the Bottleneck sandbox to a C file and compile it.

This setup.py is NOT used to install the Bottleneck package. The Bottleneck
setup.py file is bottleneck/setup.py

To convert from cython to C:

$ cd bottleneck/sandbox
$ python setup.py build_ext --inplace
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

ext_modules = [Extension("nanmean", ["nanmean.pyx"],
               include_dirs=[numpy.get_include()])]

setup(
    name='nanmean',
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules
)
