.PHONY: install

install:
	python3 -m venv venv
	./venv/bin/pip install -U pip
	./venv/bin/pip install .[test]
