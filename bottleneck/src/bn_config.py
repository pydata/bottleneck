""" Based on numpy's approach to exposing compiler features via a config header.
Unfortunately that file is not exposed, so re-implement the portions we need.
"""
import os
from numpy.distutils.command.autodist import check_gcc_function_attribute

OPTIONAL_FUNCTION_ATTRIBUTES = [("HAVE_ATTRIBUTE_OPTIMIZE_OPT_3",
                                 '__attribute__((optimize("O3")))')]


def create_config_h(config):
    dirname = os.path.dirname(__file__)
    config_h = os.path.join(dirname, 'bn_config.h')

    if (
        os.path.exists(config_h) and
        os.stat(__file__).st_mtime < os.stat(config_h).st_mtime
    ):
        return

    output = []

    for config_attr, func_attr in OPTIONAL_FUNCTION_ATTRIBUTES:
        if check_gcc_function_attribute(config, func_attr, config_attr.lower()):
            output.append((config_attr, "1"))
        else:
            output.append((config_attr, "0"))

    with open(config_h, 'w') as f:
        for setting in output:
            f.write("#define {} {}\n".format(*setting))
