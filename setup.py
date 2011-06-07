#!/usr/bin/env python

import os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np


CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description
description = "Fast NumPy array functions written in Cython"
fid = file('README.rst', 'r')
long_description = fid.read()
fid.close()
idx = max(0, long_description.find("Bottleneck is a collection"))
long_description = long_description[idx:]

# Get bottleneck version
ver_file = os.path.join('bottleneck', 'version.py')
fid = file(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')

NAME                = 'Bottleneck'
MAINTAINER          = "Keith Goodman"
MAINTAINER_EMAIL    = "bottle-neck@googlegroups.com"
DESCRIPTION         = description
LONG_DESCRIPTION    = long_description
URL                 = "http://berkeleyanalytics.com/bottleneck"
DOWNLOAD_URL        = "http://pypi.python.org/pypi/Bottleneck"
LICENSE             = "Simplified BSD"
CLASSIFIERS         = CLASSIFIERS
AUTHOR              = "Archipel Asset Management AB"
AUTHOR_EMAIL        = "kwgoodman@gmail.com"
PLATFORMS           = "OS Independent"
MAJOR               = VER[0]
MINOR               = VER[1]
MICRO               = VER[2]
ISRELEASED          = False
VERSION             = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES            = ["bottleneck",
                       "bottleneck/slow",
                       "bottleneck/tests",
                       "bottleneck/benchmark",
                       "bottleneck/src",
                       "bottleneck/src/func",
                       "bottleneck/src/move",
                       "bottleneck/src/template",
                       "bottleneck/src/template/func",
                       "bottleneck/src/template/move"]
PACKAGE_DATA        = {'bottleneck': ['LICENSE']}
REQUIRES            = ["numpy"]


# Is the default numpy int 32 or 64 bits?
if np.int_ == np.int32:
    bits = '32'
elif np.int_ == np.int64:
    bits = '64'
else:
    raise ValueError("Your OS does not appear to be 32 or 64 bits.")

setup(name=NAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      url=URL,
      download_url=DOWNLOAD_URL,
      license=LICENSE,
      classifiers=CLASSIFIERS,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      platforms=PLATFORMS,
      version=VERSION,
      packages=PACKAGES,
      package_data=PACKAGE_DATA,
      requires=REQUIRES,
      ext_package='bottleneck',
      ext_modules=[Extension("func",
                     sources=["bottleneck/src/func/%sbit/func.c" % bits],
                     include_dirs=[np.get_include()]),           
                   Extension("move",
                     sources=["bottleneck/src/move/%sbit/move.c" % bits],
                     include_dirs=[np.get_include()])]
     )                
