.PHONY: lib

LLVM_CONFIG := $(shell llvm-config --libs --cflags --ldflags)

lib:
	gcc prefilter.c -shared -o libprefilter.so -fPIC ${LLVM_CONFIG}
