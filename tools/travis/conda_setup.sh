#!/usr/bin/env bash

set -ev # exit on first error, print commands

CONDA_URL="http://repo.continuum.io/miniconda"

if [ "${PYTHON_VERSION:0:1}" == "2" ]; then
    CONDA="Miniconda2"
else
    CONDA="Miniconda3"
fi
if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    CONDA_OS="MacOSX"
else
    CONDA_OS="Linux"
fi
if [ "${PYTHON_ARCH}" == "64" ]; then
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-x86_64.sh"
else
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-x86.sh"
fi
echo "Downloading '${URL}'..."

set +e
travis_retry wget "${URL}" -O miniconda.sh
set -e

chmod +x miniconda.sh
./miniconda.sh -b -p "${HOME}/miniconda"
export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
