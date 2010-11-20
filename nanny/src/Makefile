# Simple makefile to quickly access handy build commands for Cython
# extensions.

help:
	@echo "Available tasks:"
	@echo "help  --> This help page"
	@echo "build --> Build the Cython extension module"
	@echo "clean --> Remove all the build files for a freah start"
	@echo "test  --> Run unit tests"
	@echo "all   --> clean, build, test"
	@echo "bench --> Run performance benchmark"


all: clean build test

build:
	python setup.py build_ext --inplace
	cp func.so ../func.so

test:
	python -c "import nanny; nanny.test()"

bench:
	python -c "import nanny; nanny.benchit()"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build ../*.so
