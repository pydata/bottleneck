# -*- coding: utf-8 -*-


from __future__ import absolute_import

import logging

from os import environ

from conda_manager import CondaManager


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    with CondaManager(environ['PYTHON_VERSION'], environ['PYTHON_ARCH'],
            environ['CONDA_HOME'], environ['CONDA_VENV']) as conda:
        conda.configure()
        conda.update()
        conda.create(*environ['DEPS'].split(' '))
    logging.shutdown()
