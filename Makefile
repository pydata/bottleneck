# Bottleneck Makefile

PYTHON=python

srcdir := bottleneck

help:
	@echo "Available tasks:"
	@echo "help    -->  This help page"
	@echo "all     -->  clean, build, flake8, test"
	@echo "build   -->  Build the Python C extensions"
	@echo "clean   -->  Remove all the build files for a fresh start"
	@echo "test    -->  Run unit tests"
	@echo "flake8  -->  Check for pep8 errors"
	@echo "readme  -->  Update benchmark results in README.rst"
	@echo "bench   -->  Run performance benchmark"
	@echo "detail  -->  Detailed benchmarks for all functions"
	@echo "sdist   -->  Make source distribution"
	@echo "doc     -->  Build Sphinx manual"
	@echo "pypi    -->  Upload to pypi"

all: clean build test flake8

build:
	${PYTHON} setup.py build_ext --inplace

test:
	${PYTHON} -c "import bottleneck;bottleneck.test()"

flake8:
	flake8

black:
	black . --exclude "(build/|dist/|\.git/|\.mypy_cache/|\.tox/|\.venv/\.asv/|env|\.eggs)"

readme:
	PYTHONPATH=`pwd`:PYTHONPATH ${PYTHON} tools/update_readme.py

bench:
	${PYTHON} -c "import bottleneck; bottleneck.bench()"

detail:
	${PYTHON} -c "import bottleneck; bottleneck.bench_detailed('all')"

sdist: clean
	${PYTHON} setup.py sdist
	git status

pypi: clean
	${PYTHON} setup.py sdist upload -r pypi

# doc directory exists so use phony
.PHONY: doc
doc: clean build
	rm -rf build/sphinx
	${PYTHON} setup.py build_sphinx

clean:
	rm -rf build dist Bottleneck.egg-info
	find . -name \*.pyc -delete
	rm -f MANIFEST
	rm -rf ${srcdir}/*.html ${srcdir}/build
	rm -rf ${srcdir}/*.c
	rm -rf ${srcdir}/*.so
	rm -rf ${srcdir}/bn_config.h
