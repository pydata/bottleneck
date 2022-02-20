# -*- coding: utf-8 -*-

from __future__ import absolute_import
import sys
import logging
from subprocess import check_output


if sys.version_info[0] == 2:

    def decode(string):
        return string


else:

    def decode(string):
        return string.decode()


class CondaWrapper(object):
    """Manage the AppVeyor Miniconda installation through Python.

    AppVeyor has pre-installed Python 2.7.x as well as Miniconda (2 and 3).
    Thus we only need to configure that properly and create the desired
    environment.
    """

    def __init__(self, version, home, venv, **kw_args):
        super(CondaWrapper, self).__init__(**kw_args)
        self.logger = logging.getLogger(
            "{}.{}".format(__name__, self.__class__.__name__)
        )
        self.version = version
        self.home = home
        self.venv = venv

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # False reraises the exception

    def configure(self):
        self.logger.info("Configuring '%s'...", self.home)
        cmd = [
            "conda",
            "config",
            "--set",
            "always_yes",
            "yes",
            "--set",
            "changeps1",
            "no",
        ]
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        self.logger.info("Done.")

    def update(self):
        self.logger.info("Updating '%s'...", self.home)
        cmd = ["conda", "update", "-q", "conda"]
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        self.logger.info("Done.")

    def create(self, *args):
        self.logger.info("Creating environment '%s'...", self.venv)
        cmd = [
            "conda",
            "create",
            "-q",
            "-n",
            self.venv,
            "python=" + self.version,
        ] + list(args)
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        cmd = ["activate", self.venv]
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        # consider only for debugging
        cmd = ["conda", "info", "-a"]
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        cmd = ["conda", "list"]
        msg = check_output(cmd, shell=True)
        self.logger.debug(decode(msg))
        self.logger.info("Done.")
