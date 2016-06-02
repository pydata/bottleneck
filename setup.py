#!/usr/bin/env python

import os
import sys

try:
    import setuptools  # noqa
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

from setuptools import setup, find_packages
from setuptools.extension import Extension


modules = ['reduce', 'reduce2', 'nonreduce', 'nonreduce_axis', 'move']


def prepare_modules():

    # Don't attempt to import numpy until it needed; this
    # enables pip to install numpy before bottleneck
    import numpy as np

    # info needed to compile bottleneck
    kwargs = {}
    for module in modules:
        kwargs[module] = {'include_dirs': [np.get_include()],
                          'extra_compile_args': ['-O2']}
        if module == 'move':
            kwargs[module]['extra_compile_args'].insert(0, '-std=gnu89')

    if CYTHON_AVAILABLE:
        # create the C files
        from bottleneck.template.template import make_pyx
        make_pyx()
        return cythonize([Extension("bottleneck.%s" % module,
                                    sources=["bottleneck/%s.pyx" % module],
                                    **kwargs[module])
                          for module in modules])
    else:
        # assume the presence of shipped C files
        return [Extension("bottleneck.%s" % module,
                sources=["bottleneck/%s.c" % module],
                **kwargs[module])
                for module in modules]


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
AUTHOR = "Keith Goodman"
AUTHOR_EMAIL = "kwgoodman@gmail.com"
PLATFORMS = "OS Independent"
MAJOR = VER[0]
MINOR = VER[1]
MICRO = VER[2]
ISRELEASED = False
VERSION = '%s.%s.%s' % (MAJOR, MINOR, MICRO)
PACKAGES = find_packages()
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
                install_requires=['numpy'])


if not(len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
       sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean',
                       'build_sphinx'))):
    metadata['ext_modules'] = prepare_modules()
elif sys.argv[1] == 'build_sphinx':
    # create intro.rst (from readme file) for sphinx manual
    readme = 'README.rst'
    intro = os.path.join('doc', 'source', 'intro.rst')
    with open(readme, 'r') as infile, open(intro, 'w') as outfile:
        txt = infile.readlines()[2:]  # skip travis build status
        outfile.write(''.join(txt))

setup(**metadata)
