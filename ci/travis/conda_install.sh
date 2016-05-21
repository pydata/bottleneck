#!/usr/bin/env bash

set -uev # error if unset variables, exit on first error, print commands

if [ "${PYTHON_ARCH}" == "32" ]; then
  set CONDA_FORCE_32BIT=1
fi
NAME="test-${PYTHON_VERSION}-${PYTHON_ARCH}bit"
conda create -q -n "${NAME}" "${DEPS}" python="${PYTHON_VERSION}"
source activate "${NAME}"
conda info -a
conda list
