# Simple makefile to quickly access handy build commands for Cython
# extensions.

help:
	@echo "Available tasks:"
	@echo "help  --> This help page"
	@echo "pyx   --> Create Cython pyx file from templates"
	@echo "build --> Build the Cython extension modules"
	@echo "clean --> Remove all the build files for a freah start"
	@echo "test  --> Run unit tests"
	@echo "all   --> clean, build, test"
	@echo "bench --> Run performance benchmark"


all: clean pyx build test

pyx:
	python -c "from bottleneck.src.template.func.func import funcpyx; \
    funcpyx()"
	python -c "from bottleneck.src.template.move.move import movepyx; \
    movepyx()"
	python -c "from bottleneck.src.template.group.group import grouppyx; \
    grouppyx()"

build: funcs moves groups
	
funcs:
	rm -rf func/func.c ../func.so
	python func/setup.py build_ext --inplace
	mv func.so ../func.so
	
moves:
	rm -rf move/move.c ../move.so
	python move/setup.py build_ext --inplace
	mv move.so ../move.so

groups:
	rm -rf group/group.c ../group.so
	python group/setup.py build_ext --inplace
	mv group.so ../group.so

test:
	python -c "import bottleneck; bottleneck.test()"

bench:
	python -c "import bottleneck; bottleneck.benchit()"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build ../*.so func/*.c move/*.c
