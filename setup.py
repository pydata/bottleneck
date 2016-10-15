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
from setuptools.command.build_ext import build_ext as _build_ext


# workaround for installing bottleneck when numpy is not present
class build_ext(_build_ext):
    # taken from: stackoverflow.com/questions/19919905/
    # how-to-bootstrap-numpy-installation-in-setup-py#21621689
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy
        self.include_dirs.append(numpy.get_include())


def prepare_modules():
    from bottleneck.src.template import make_c_files
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
                install_requires=['numpy'],
                cmdclass={'build_ext': build_ext},
                setup_requires=['numpy'])


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
        txt = infile.readlines()[4:]  # skip travis, appveyor build status
        outfile.write(''.join(txt))

setup(**metadata)
