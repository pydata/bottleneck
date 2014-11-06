# Bottleneck Makefile

PYTHON=python

srcdir := bottleneck/src

help:
	@echo "Available tasks:"
	@echo "help    -->  This help page"
	@echo "cfiles  -->  Convert pyx files to C files"
	@echo "build   -->  Build the Cython extension modules"
	@echo "clean   -->  Remove all the build files for a fresh start"
	@echo "test    -->  Run unit tests"
	@echo "coverage-->  Unit test coverage (doesn't check compiled functions)"
	@echo "all     -->  clean, pyx, build, test"
	@echo "bench   -->  Run performance benchmark"
	@echo "sdist   -->  Make source distribution"

all: clean cfiles build test

cfiles:
	cython ${srcdir}/reduce.pyx

build: reduce

reduce:
	rm -rf ${srcdir}/../reduce.so
	${PYTHON} ${srcdir}/reduce_setup.py build_ext --inplace

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
	rm -rf ${srcdir}/*~ ${srcdir}/*.so ${srcdir}/*.c ${srcdir}/*.o ${srcdir}/*.html ${srcdir}/build ${srcdir}/../*.so
	rm -rf ${srcdir}/*.c
