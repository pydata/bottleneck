# Bottleneck Makefile

PYTHON=python

srcdir := bottleneck/src/auto_pyx

help:
	@echo "Available tasks:"
	@echo "help    -->  This help page"
	@echo "pyx     -->  Create Cython pyx files from templates"
	@echo "cfiles  -->  Convert pyx files to C files"
	@echo "build   -->  Build the Cython extension modules"
	@echo "clean   -->  Remove all the build files for a fresh start"
	@echo "test    -->  Run unit tests"
	@echo "coverage-->  Unit test coverage (doesn't check compiled functions)"
	@echo "all     -->  clean, pyx, build, test"
	@echo "bench   -->  Run performance benchmark"
	@echo "sdist   -->  Make source distribution"

all: clean pyx cfiles build test

pyx:
	${PYTHON} -c "from bottleneck.src.template.template import make_pyx; make_pyx();"

cfiles:
	cython ${srcdir}/reduce.pyx
	cython ${srcdir}/nonreduce_axis.pyx
	cython ${srcdir}/move.pyx

build: reduce nonreduce_axis move

reduce:
	rm -rf ${srcdir}/../reduce.so
	${PYTHON} ${srcdir}/reduce_setup.py build_ext --inplace

nonreduce_axis:
	rm -rf ${srcdir}/../nonreduce_axis.so
	${PYTHON} ${srcdir}/nonreduce_axis_setup.py build_ext --inplace

move:
	rm -rf ${srcdir}/../move.so
	${PYTHON} ${srcdir}/move_setup.py build_ext --inplace

test:
	${PYTHON} -c "import bottleneck;bottleneck.test()"

coverage:
	rm -rf .coverage
	python -c "import bottleneck; bottleneck.test(coverage=True)"

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
	rm -rf ${srcdir}/*.c ${srcdir}/*.html ${srcdir}/build ${srcdir}/../../*.so
