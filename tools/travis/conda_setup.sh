#!/usr/bin/env bash

set -ev # exit on first error, print commands

CONDA_URL="http://repo.continuum.io/miniconda"
MINIFORGE_URL="https://github.com/conda-forge/miniforge/"
MINIFORGE_VERSION="4.8.2-1"

if [ "${TRAVIS_CPU_ARCH}" == "amd64" ]; then
    ARCH="x86_64"
elif [ "${TRAVIS_CPU_ARCH}" == "arm64" ]; then
    ARCH="aarch64"
else
    ARCH="${TRAVIS_CPU_ARCH}"
fi

if [ "${ARCH}" == "aarch64" ]; then
    CONDA="Miniforge"
else
    CONDA="Miniconda"
fi

if [ "${PYTHON_VERSION:0:1}" == "2" ]; then
    CONDA="${CONDA}2"
else
    CONDA="${CONDA}3"
fi

if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    CONDA_OS="MacOSX"
else
    CONDA_OS="Linux"
fi

if [ "${ARCH}" == "aarch64" ]; then
    URL="${MINIFORGE_URL}/releases/download/${MINIFORGE_VERSION}/${CONDA}-${MINIFORGE_VERSION}-${CONDA_OS}-${ARCH}.sh"
elif [ "${PYTHON_ARCH}" == "64" ]; then
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-${ARCH}.sh"
else
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-x86.sh"
fi
echo "Downloading '${URL}'..."

set +e
travis_retry wget "${URL}" -O installer.sh
set -e


chmod +x installer.sh
./installer.sh -b -p "${HOME}/miniconda"

export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r

conda config --set always_yes yes --set changeps1 no
conda update -q conda
