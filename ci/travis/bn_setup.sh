#!/usr/bin/env bash

set -ev # exit on first error, print commands

if [ "${RUN}" = "style" ]; then
    flake8 bottleneck
elif [ "${RUN}" = "sdist" ]; then
    python setup.py sdist
    conda uninstall -n "${NAME}" cython
    ARCHIVE=`ls dist/*.tar.gz`
    pip install --verbose "${ARCHIVE[0]}"
    python "tools/test-installed-bottleneck.py"
else
    pip install --verbose "."
    python "tools/test-installed-bottleneck.py"
fi
