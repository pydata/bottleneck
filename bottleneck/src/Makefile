# Bottleneck Makefile 

help:
	@echo "Available tasks:"
	@echo "help   -->  This help page"
	@echo "pyx    -->  Create Cython pyx files from templates"
	@echo "cfiles -->  Convert pyx files to C files"
	@echo "build  -->  Build the Cython extension modules"
	@echo "clean  -->  Remove all the build files for a fresh start"
	@echo "test   -->  Run unit tests"
	@echo "all    -->  clean, pyx, build, test"
	@echo "bench  -->  Run performance benchmark"
	@echo "sdist  -->  Make source distribution"

all: clean pyx cfiles build test

pyx:
	python -c "from bottleneck.src.makepyx import makepyx; makepyx()"

cfiles:
	cython func/32bit/func.pyx
	cython func/64bit/func.pyx
	cython move/32bit/move.pyx
	cython move/64bit/move.pyx

build: funcs moves
	
funcs:
	rm -rf ../func.so
	python func/setup.py build_ext --inplace
	
moves:
	rm -rf ../move.so
	python move/setup.py build_ext --inplace
		
test:
	python -c "import bottleneck;bottleneck.test(extra_argv=['--processes=4'])"

bench:
	python -c "import bottleneck; bottleneck.bench()"

sdist:
	rm -f ../../MANIFEST
	git status
	find ../.. -name *.c
	cd ../..; python setup.py sdist

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build ../*.so
	rm -rf func/32bit/*.c func/64bit/*.c
	rm -rf move/32bit/*.c move/64bit/*.c
	rm -rf func/32bit/*.pyx  func/64bit/*.pyx
	rm -rf move/32bit/*.pyx  move/64bit/*.pyx
