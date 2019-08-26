#!/usr/bin/env python

import os
import sys

from setuptools import setup, find_packages
from setuptools.extension import Extension
from setuptools.command.build_ext import build_ext as _build_ext
import versioneer


# workaround for installing bottleneck when numpy is not present
class build_ext(_build_ext):
    # taken from: stackoverflow.com/questions/19919905/
    # how-to-bootstrap-numpy-installation-in-setup-py#21621689
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process
        if sys.version_info < (3,):
            import __builtin__ as builtins
        else:
            import builtins
        builtins.__NUMPY_SETUP__ = False
        import numpy
        # place numpy includes first, see gh #156
        self.include_dirs.insert(0, numpy.get_include())


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = build_ext

# Add our template path to the path so that we don't have a circular reference
# of working install to be able to re-compile
sys.path.append(os.path.join(os.path.dirname(__file__), 'bottleneck/src'))


def prepare_modules():
    from bn_template import make_c_files
    make_c_files()
    ext = [Extension("bottleneck.reduce",
                     sources=["bottleneck/src/reduce.c"],
                     extra_compile_args=['-O2'])]
    ext += [Extension("bottleneck.move",
                      sources=["bottleneck/src/move.c",
                               "bottleneck/src/move_median/move_median.c"],
                      extra_compile_args=['-O2'])]
    ext += [Extension("bottleneck.nonreduce",
                      sources=["bottleneck/src/nonreduce.c"],
                      extra_compile_args=['-O2'])]
    ext += [Extension("bottleneck.nonreduce_axis",
                      sources=["bottleneck/src/nonreduce_axis.c"],
                      extra_compile_args=['-O2'])]
    return ext


def get_long_description():
    with open('README.rst', 'r') as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("Bottleneck is a collection"))
    long_description = long_description[idx:]
    return long_description


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
                maintainer="Christopher Whelan",
                maintainer_email="bottle-neck@googlegroups.com",
                description="Fast NumPy array functions written in C",
                long_description=get_long_description(),
                url="https://github.com/pydata/bottleneck",
                download_url="http://pypi.python.org/pypi/Bottleneck",
                license="Simplified BSD",
                classifiers=CLASSIFIERS,
                platforms="OS Independent",
                version=versioneer.get_version(),
                packages=find_packages(),
                package_data={'bottleneck': ['LICENSE']},
                requires=['numpy'],
                install_requires=['numpy'],
                cmdclass=cmdclass,
                setup_requires=['numpy'],
                ext_modules=prepare_modules(),
                zip_safe=False)


setup(**metadata)
