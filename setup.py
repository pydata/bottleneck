#!/usr/bin/env python

import os
import sys

try:
    import setuptools  # noqa
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()

from setuptools import setup, find_packages
from setuptools.extension import Extension


def prepare_modules():

    # Don't attempt to import numpy until it needed; this
    # enables pip to install numpy before bottleneck
    import numpy as np
    from bottleneck.src.template import make_c_files

    make_c_files()
    ext_list = [Extension("bottleneck.reduce",
                          sources=["bottleneck/src/reduce.c"],
                          include_dirs=[np.get_include()],
                          extra_compile_args=['-O2'])]
    ext_list += [Extension("bottleneck.move",
                           sources=["bottleneck/src/move.c",
                                    "bottleneck/src/move_median/move_median.c"],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O2'])]
    ext_list += [Extension("bottleneck.nonreduce",
                           sources=["bottleneck/src/nonreduce.c"],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O2'])]
    ext_list += [Extension("bottleneck.nonreduce_axis",
                           sources=["bottleneck/src/nonreduce_axis.c"],
                           include_dirs=[np.get_include()],
                           extra_compile_args=['-O2'])]

    return ext_list


def get_long_description():
    with open('README.rst', 'r') as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("Bottleneck is a collection"))
    long_description = long_description[idx:]
    return long_description


def get_version_str():
    ver_file = os.path.join('bottleneck', 'version.py')
    with open(ver_file, 'r') as fid:
        version = fid.read()
    version = version.split("= ")
    version = version[1].strip()
    version = version.strip("\"")
    return version


CLASSIFIERS = ["Development Status :: 4 - Beta",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: BSD License",
               "Operating System :: OS Independent",
               "Programming Language :: C",
               "Programming Language :: Python",
               "Programming Language :: Python :: 3",
               "Topic :: Scientific/Engineering"]


metadata = dict(name='Bottleneck',
                maintainer="Keith Goodman",
                maintainer_email="bottle-neck@googlegroups.com",
                description="Fast NumPy array functions written in C",
                long_description=get_long_description(),
                url="https://github.com/kwgoodman/bottleneck",
                download_url="http://pypi.python.org/pypi/Bottleneck",
                license="Simplified BSD",
                classifiers=CLASSIFIERS,
                platforms="OS Independent",
                version=get_version_str(),
                packages=find_packages(),
                package_data={'bottleneck': ['LICENSE']},
                requires=['numpy'],
                install_requires=['numpy'])


if not(len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
       sys.argv[1] in ('--help-commands', 'egg_info', '--version', 'clean',
                       'build_sphinx'))):
    # build bottleneck
    metadata['ext_modules'] = prepare_modules()
elif sys.argv[1] == 'build_sphinx':
    # create intro.rst (from readme file) for sphinx manual
    readme = 'README.rst'
    intro = os.path.join('doc', 'source', 'intro.rst')
    with open(readme, 'r') as infile, open(intro, 'w') as outfile:
        txt = infile.readlines()[2:]  # skip travis build status
        outfile.write(''.join(txt))

setup(**metadata)
