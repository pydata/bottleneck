#!/usr/bin/env bash

set -ev # exit on first error, print commands

if [ "${PYTHON_ARCH}" == "32" ]; then
  set CONDA_FORCE_32BIT=1
fi
NAME="test-python-${PYTHON_VERSION}-${PYTHON_ARCH}bit"
# split dependencies into separate packages
IFS=" " DEPS=(${DEPS})
conda create -q -n "${NAME}" "${DEPS[@]}" python="${PYTHON_VERSION}"

set +v # we dont want to  see commands in the conda script

source activate "${NAME}"
conda info -a
conda list
