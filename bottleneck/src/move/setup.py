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
import sys
import os
import os.path
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# Is the default numpy int 32 or 64 bits?
if np.int_ == np.int32:
    bits = '32'
elif np.int_ == np.int64:
    bits = '64'
else:
    raise ValueError("Your OS does not appear to be 32 or 64 bits.")

mod_dir = os.path.dirname(__file__)
ext_modules = [Extension("move", [os.path.join(mod_dir, "%sbit/move.pyx") % bits],
               extra_compile_args=["-std=gnu89"],
               include_dirs=[np.get_include()])]

setup(
  name = 'move',
  cmdclass = {'build_ext': build_ext},
  ext_modules = ext_modules
)

if sys.version_info[0] >= 3:
	# E.g. Ubuntu has ABI version tagged .so files
	# http://www.python.org/dev/peps/pep-3149/
	import sysconfig
	ext = sysconfig.get_config_var('SO')
else:
	ext = '.so'

os.rename("move{0}".format(ext),
          os.path.join(mod_dir, "../../move{0}".format(ext)))
