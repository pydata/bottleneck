#!/usr/bin/env python

import os
import shutil
import sys
import sysconfig
from distutils.command.config import config as _config

from setuptools import Command, setup
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.extension import Extension

import versioneer

# restrict LIMITED_API usage:
# - require BN_LIMITED_API=1
# - LIMITED_API is not compatible with free-threading (as of CPython 3.14)
USE_PY_LIMITED_API = os.getenv(
    "BN_LIMITED_API", "0"
) == "1" and not sysconfig.get_config_var("Py_GIL_DISABLED")
ABI3_TARGET_VERSION = "".join(str(_) for _ in sys.version_info[:2])
ABI3_TARGET_HEX = hex(sys.hexversion & 0xFFFF00F0)


define_macros = [
    # keep in sync with runtime requirements (pyproject.toml)
    ("NPY_NO_DEPRECATED_API", "NPY_1_21_API_VERSION"),
]
if USE_PY_LIMITED_API:
    define_macros.append(("Py_LIMITED_API", ABI3_TARGET_HEX))


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
            define_macros=define_macros,
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
            define_macros=define_macros,
            extra_compile_args=["-O2"],
        )
    ]
    ext += [
        Extension(
            "bottleneck.nonreduce",
            sources=["bottleneck/src/nonreduce.c"],
            depends=base_includes,
            define_macros=define_macros,
            extra_compile_args=["-O2"],
        )
    ]
    ext += [
        Extension(
            "bottleneck.nonreduce_axis",
            sources=["bottleneck/src/nonreduce_axis.c"],
            depends=base_includes,
            define_macros=define_macros,
            extra_compile_args=["-O2"],
        )
    ]
    return ext


if USE_PY_LIMITED_API:
    options = {"bdist_wheel": {"py_limited_api": f"cp{ABI3_TARGET_VERSION}"}}
else:
    options = {}

setup(
    version=versioneer.get_version(),
    package_data={
        "bottleneck.tests": ["data/*/*"],
    },
    cmdclass=cmdclass,
    ext_modules=prepare_modules(),
    options=options,
)
