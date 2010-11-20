# Simple makefile to quickly access handy build commands for Cython
# extensions.

help:
	@echo "Numpy/Cython tasks.  Available tasks:"
	@echo "cython  -> build the Cython extension module."
	@echo "clean -> remove all the build files for a freah start."
	@echo "all  -> clean, build, test."


all: clean build test

build:
	python setup.py build_ext --inplace
	cp func.so ../func.so

test:
	python -c "import nanny; nanny.test()"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build ../*.so
