#!/usr/bin/env bash

set -ev # exit on first error, print commands

COVERAGE=""
if [ "${TEST_RUN}" == "coverage" ]; then
    COVERAGE="--coverage"
    export CFLAGS="-fprofile-arcs -ftest-coverage"
    export LDFLAGS="-fprofile-arcs"
fi

if [ "${TEST_RUN}" == "style" ]; then
    flake8
else
    if [ "${TEST_RUN}" == "sdist" ]; then
        python setup.py sdist
        ARCHIVE=`ls dist/*.tar.gz`
        pip install "${ARCHIVE[0]}"
    elif [ "${TEST_RUN}" != "coverage" ]; then
	# CFLAGS gets ignored by PEP 518, so do coverage from inplace build
        pip install "."
    fi
    python setup.py build_ext --inplace
    if [ "${TEST_RUN}" == "doc" ]; then
	make doc
    elif [ "${TEST_RUN}" == "coverage" ]; then
	py.test --cov=bottleneck --cov-branch
	find build -iname "*.gc*" -exec mv {} . \;
#	cd build/temp.linux-x86_64-3.7/
#	cp -r bottleneck ../..
	#	cd ../..
	echo $(ls)
	echo $(ls bottleneck)
	echo $(ls bottleneck/src)
	bash <(curl -s https://codecov.io/bash)
    else
	# Workaround for https://github.com/travis-ci/travis-ci/issues/6522
	python "tools/test-installed-bottleneck.py"
    fi
fi
