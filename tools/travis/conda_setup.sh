#!/usr/bin/env bash

set -ev # exit on first error, print commands

CONDA_URL="http://repo.continuum.io/miniconda"
ARCHICONDA_URL="https://github.com/Archiconda"

if [ `uname -m` == 'aarch64' ]; then
    MINICONDA_DIR="$HOME/archiconda3"
    IS_SUDO="sudo"
elif [ "${PYTHON_VERSION:0:1}" == "2" ]; then
    CONDA="Miniconda2"
else
    CONDA="Miniconda3"
fi
if [ "${TRAVIS_OS_NAME}" == "osx" ]; then
    CONDA_OS="MacOSX"
else
    CONDA_OS="Linux"
fi
if [ `uname -m` == 'aarch64' ]; then
    URL="${ARCHICONDA_URL}/build-tools/releases/download/0.2.2/Archiconda3-0.2.2-Linux-aarch64.sh"
elif [ "${PYTHON_ARCH}" == "64" ]; then
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-x86_64.sh"
else
    URL="${CONDA_URL}/${CONDA}-latest-${CONDA_OS}-x86.sh"
fi
echo "Downloading '${URL}'..."

set +e
if [ `uname -m` == 'aarch64' ]; then
    travis_retry wget "${URL}" -O archiconda.sh
else
    travis_retry wget "${URL}" -O miniconda.sh
fi
set -e

if [ `uname -m` == 'aarch64' ]; then
    chmod +x archiconda.sh
    $IS_SUDO apt-get install python-dev
    $IS_SUDO apt-get install python3-pip
    bash archiconda.sh -b -p $HOME/miniconda;
    $IS_SUDO chmod -R 777 /home/travis/.cache/pip
    $IS_SUDO cp -r $HOME/miniconda/bin/* /usr/bin/;
    $IS_SUDO rm /usr/bin/lsb_release
else
    chmod +x miniconda.sh
    ./miniconda.sh -b -p "${HOME}/miniconda"
fi
export PATH="${HOME}/miniconda/bin:${PATH}"
hash -r
if [ `uname -m` == 'aarch64' ]; then
    $IS_SUDO conda config --set always_yes yes --set changeps1 no
    $IS_SUDO conda update -n base conda
else
    conda config --set always_yes yes --set changeps1 no
    conda update -q conda
fi
