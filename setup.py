#!/usr/bin/env python

import os
import shutil
import sys
from distutils.command.config import config as _config

from setuptools import Command, find_packages, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension

import versioneer


class config(_config):
    def run(self):
        from bn_config import create_config_h

        create_config_h(self)


class clean(Command):
    user_options = [("all", "a", "")]

    def initialize_options(self):
        self.all = True
        self.delete_dirs = []
        self.delete_files = []

        for root, dirs, files in os.walk("bottleneck"):
            for d in dirs:
                if d == "__pycache__":
                    self.delete_dirs.append(os.path.join(root, d))

            if "__pycache__" in root:
                continue

            for f in files:
                if f.endswith(".pyc") or f.endswith(".so"):
                    self.delete_files.append(os.path.join(root, f))

                if f.endswith(".c") and "template" in f:
                    generated_file = os.path.join(root, f.replace("_template", ""))
                    if os.path.exists(generated_file):
                        self.delete_files.append(generated_file)

        config_h = "bottleneck/src/bn_config.h"
        if os.path.exists(config_h):
            self.delete_files.append(config_h)

        if os.path.exists("build"):
            self.delete_dirs.append("build")

    def finalize_options(self):
        pass

    def run(self):
        for delete_dir in self.delete_dirs:
            shutil.rmtree(delete_dir)
        for delete_file in self.delete_files:
            os.unlink(delete_file)


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
        self.include_dirs.append("bottleneck/src")

    def build_extensions(self):
        from bn_template import make_c_files

        self.run_command("config")
        dirpath = "bottleneck/src"
        modules = ["reduce", "move", "nonreduce", "nonreduce_axis"]
        make_c_files(dirpath, modules)

        _build_ext.build_extensions(self)


cmdclass = versioneer.get_cmdclass()
cmdclass["build_ext"] = build_ext
cmdclass["clean"] = clean
cmdclass["config"] = config

# Add our template path to the path so that we don't have a circular reference
# of working install to be able to re-compile
sys.path.append(os.path.join(os.path.dirname(__file__), "bottleneck/src"))


def prepare_modules():
    base_includes = [
        "bottleneck/src/bottleneck.h",
        "bottleneck/src/bn_config.h",
        "bottleneck/src/iterators.h",
    ]
    ext = [
        Extension(
            "bottleneck.reduce",
            sources=["bottleneck/src/reduce.c"],
            depends=base_includes,
            extra_compile_args=["-O2"],
        )
    ]
    ext += [
        Extension(
            "bottleneck.move",
            sources=[
                "bottleneck/src/move.c",
                "bottleneck/src/move_median/move_median.c",
            ],
            depends=base_includes + ["bottleneck/src/move_median/move_median.h"],
            extra_compile_args=["-O2"],
        )
    ]
    ext += [
        Extension(
            "bottleneck.nonreduce",
            sources=["bottleneck/src/nonreduce.c"],
            depends=base_includes,
            extra_compile_args=["-O2"],
        )
    ]
    ext += [
        Extension(
            "bottleneck.nonreduce_axis",
            sources=["bottleneck/src/nonreduce_axis.c"],
            depends=base_includes,
            extra_compile_args=["-O2"],
        )
    ]
    return ext


def get_long_description():
    with open("README.rst", "r") as fid:
        long_description = fid.read()
    idx = max(0, long_description.find("Bottleneck is a collection"))
    long_description = long_description[idx:]
    return long_description


CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Financial and Insurance Industry",
    "License :: OSI Approved :: BSD License",
    "Operating System :: OS Independent",
    "Programming Language :: C",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.5",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering",
]


metadata = dict(
    name="Bottleneck",
    maintainer="Christopher Whelan",
    maintainer_email="bottle-neck@googlegroups.com",
    description="Fast NumPy array functions written in C",
    long_description=get_long_description(),
    long_description_content_type="text/x-rst",
    url="https://github.com/pydata/bottleneck",
    download_url="http://pypi.python.org/pypi/Bottleneck",
    license="Simplified BSD",
    classifiers=CLASSIFIERS,
    platforms="OS Independent",
    version=versioneer.get_version(),
    packages=find_packages(),
    package_data={
        "bottleneck": ["LICENSE"],
        "bottleneck.tests": ["data/*/*"],
    },
    requires=["numpy"],
    install_requires=["numpy"],
    extras_require={"doc": ["numpydoc", "sphinx", "gitpython"]},
    cmdclass=cmdclass,
    setup_requires=["numpy"],
    ext_modules=prepare_modules(),
    zip_safe=False,
)


setup(**metadata)
