# Simple makefile to quickly access handy build commands for Cython
# extensions.

help:
	@echo "Available tasks:"
	@echo "help  --> This help page"
	@echo "pyx   --> Create Cython pyx files from templates"
	@echo "pyx32 --> Create Cython pyx files from templates (32 bit OS)"
	@echo "pyx64 --> Create Cython pyx files from templates (64 bit OS)"
	@echo "build --> Build the Cython extension modules"
	@echo "clean --> Remove all the build files for a freah start"
	@echo "test  --> Run unit tests"
	@echo "all   --> clean, build, test"
	@echo "all32 --> clean, build, test for 32 bit OS"
	@echo "all64 --> clean, build, test for 64 bit OS"
	@echo "bench --> Run performance benchmark"


all: clean pyx build test
all32: clean pyx32 build test
all64: clean pyx64 build test

pyx:
	python -c "from bottleneck.src.makepyx import makepyx; makepyx(bits=None)"

pyx32:
	python -c "from bottleneck.src.makepyx import makepyx; makepyx(bits=32)"

pyx64:
	python -c "from bottleneck.src.makepyx import makepyx; makepyx(bits=64)"


build: funcs moves
	
funcs:
	rm -rf func/func.c ../func.so
	python func/setup.py build_ext --inplace
	mv func.so ../func.so
	
moves:
	rm -rf move/move.c ../move.so
	python move/setup.py build_ext --inplace
	mv move.so ../move.so

test:
	python -c "import bottleneck; bottleneck.test()"

bench:
	python -c "import bottleneck; bottleneck.bench()"

# Phony targets for cleanup and similar uses

.PHONY: clean
clean:
	rm -rf *~ *.so *.c *.o *.html build ../*.so func/*.c move/*.c
