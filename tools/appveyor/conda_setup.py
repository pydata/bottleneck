# -*- coding: utf-8 -*-


from __future__ import absolute_import
import logging
from os import environ

from conda_wrapper import CondaWrapper


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with CondaWrapper(
        environ["PYTHON_VERSION"], environ["CONDA_HOME"], environ["CONDA_VENV"]
    ) as conda:
        conda.configure()
        conda.update()
        conda.create(*environ["DEPS"].split(" "))
    logging.shutdown()
