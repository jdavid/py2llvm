.PHONY: build install

build:
	python setup.py build_ext --inplace

install:
	python3 -m venv venv
	./venv/bin/pip install -U pip
	./venv/bin/pip install .[test]
