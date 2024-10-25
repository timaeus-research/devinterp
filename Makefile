.PHONY:  docs docs-auto test

VENV_NAME := .venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

docs-prep:
	pip install devinterp[docs]
	cd docs && python generate_docs.py && cd ..

docs:
	make docs-prep
	make -C docs html
	# sphinx-apidoc -o docs ./src/devinterp ./src/devinterp/mechinterp --force 
	# sphinx-build -b html -E -a docs docs/_build/html

docs-auto:
	make docs-prep
	sphinx-autobuild docs docs/_build/html


publish-docs:
	cp -rf docs/_build/html/* ../devinterp-docs/public

test:
	. .venv/bin/activate && pytest tests/

%:
	@: