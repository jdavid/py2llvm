.PHONY: lib

CC ?= gcc
LLVM_CONFIG := $(shell llvm-config --libs --cflags --ldflags)

lib:
	${CC} prefilter.c -shared -o libprefilter.so -fPIC ${LLVM_CONFIG}
