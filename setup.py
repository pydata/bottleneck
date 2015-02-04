#!/usr/bin/env python

import os
import sys

try:
    import setuptools
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup
from setuptools.extension import Extension

CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: Cython",
               "Programming Language :: Python",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering"]

# Description
description = "Fast NumPy array functions written in Cython"
fid = open('README.rst', 'r')
long_description = fid.read()
fid.close()
idx = max(0, long_description.find("Bottleneck is a collection"))
long_description = long_description[idx:]

# Get bottleneck version
ver_file = os.path.join('bottleneck', 'version.py')
fid = open(ver_file, 'r')
VER = fid.read()
fid.close()
VER = VER.split("= ")
VER = VER[1].strip()
VER = VER.strip("\"")
VER = VER.split('.')

NAME = 'Bottleneck'
MAINTAINER = "Keith Goodman"
MAINTAINER_EMAIL = "bottle-neck@googlegroups.com"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://berkeleyanalytics.com/bottleneck"
DOWNLOAD_URL = "http://pypi.python.org/pypi/Bottleneck"
LICENSE = "Simplified BSD"
CLASSIFIERS = CLASSIFIERS
AUTHOR = "Berkeley Analytics LLC"
AUTHOR_EMAIL = "kwgoodman@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = VER[0]
MINOR = VER[1]
MICRO = VER[2]
ISRELEASED = False
VERSION = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES = ["bottleneck",
            "bottleneck/slow",
            "bottleneck/tests",
            "bottleneck/benchmark",
            "bottleneck/src",
            "bottleneck/src/template",
            "bottleneck/src/auto_pyx"]
PACKAGE_DATA = {'bottleneck': ['LICENSE']}
REQUIRES = ["numpy"]

metadata = dict(name=NAME,
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
                install_requires = ['numpy']
            )

# Don't attempt to import numpy when it isn't actually needed; this enables pip
# to install numpy before bottleneck:
if not(len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or \
       sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean'))):

    import numpy as np
    metadata['ext_package'] = 'bottleneck'
    metadata['ext_modules'] = \
                              [Extension("reduce", sources=["bottleneck/src/auto_pyx/reduce.c"],
                                         include_dirs=[np.get_include()]),
                               Extension("nonreduce", sources=["bottleneck/src/auto_pyx/nonreduce.c"],
                                         include_dirs=[np.get_include()]),
                               Extension("nonreduce_axis", sources=["bottleneck/src/auto_pyx/nonreduce_axis.c"],
                                         include_dirs=[np.get_include()]),
                               Extension("move", sources=["bottleneck/src/auto_pyx/move.c"],
                                         extra_compile_args=["-std=gnu89"],
                                         include_dirs=[np.get_include()])]

setup(**metadata)
