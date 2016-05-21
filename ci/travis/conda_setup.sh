#!/usr/bin/env bash

set -uev # error if unset variables, exit on first error, print commands

MINICONDA_URL="http://repo.continuum.io/miniconda"

if [ "${PYTHON_VERSION:0:1}" == "2" ]; then
    MINICONDA="Miniconda2"
else
    MINICONDA="Miniconda3"
fi
if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    MINICONDA_OS="MacOSX"
else
    MINICONDA_OS="Linux"
fi
if [ "${PYTHON_ARCH}" == "64" ]; then
    URL="${MINICONDA_URL}/${MINICONDA}-latest-${MINICONDA_OS}-x86_64.sh"
else
    URL="${MINICONDA_URL}/${MINICONDA}-latest-${MINICONDA_OS}-x86.sh"
fi
travis_retry wget "${URL}" -O miniconda.sh
chmod +x miniconda.sh
./miniconda.sh -b -p "${HOME}/miniconda"
export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r
conda config --set always_yes yes --set changeps1 no
conda update -q conda
