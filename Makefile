# Bottleneck Makefile

PYTHON=python

srcdir := bottleneck/

help:
	@echo "Available tasks:"
	@echo "help    -->  This help page"
	@echo "all     -->  clean, build, flake8, test"
	@echo "build   -->  Build the Cython extension modules"
	@echo "clean   -->  Remove all the build files for a fresh start"
	@echo "test    -->  Run unit tests"
	@echo "flake8  -->  Check for pep8 errors"
	@echo "readme  -->  Update benchmark results in README.rst"
	@echo "coverage-->  Unit test coverage (doesn't check compiled functions)"
	@echo "bench   -->  Run performance benchmark"
	@echo "sdist   -->  Make source distribution"

all: clean build flake8 test

build:
	${PYTHON} setup.py build_ext --inplace

test:
	${PYTHON} -c "import bottleneck;bottleneck.test()"

flake8:
	flake8 bottleneck

readme:
	PYTHONPATH=`pwd`:PYTHONPATH ${PYTHON} tools/update_readme.py

coverage:
	rm -rf .coverage
	${PYTHON} -c "import bottleneck; bottleneck.test(coverage=True)"

bench:
	${PYTHON} -c "import bottleneck; bottleneck.bench()"

sdist: pyx
	rm -f MANIFEST
	${PYTHON} setup.py sdist
	git status

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf build dist Bottleneck.egg-info
	find . -name \*.pyc -delete
	rm -rf ${srcdir}/*.c ${srcdir}/*.html ${srcdir}/build ${srcdir}/*.so ${srcdir}/*.pyx
