.PHONY:  docs docs-auto test

# Try to find uv, otherwise fall back to system paths
UV := $(shell command -v uv 2>/dev/null || echo ~/.local/bin/uv)
VENV_NAME := ../../.venv
PYTHON := $(VENV_NAME)/bin/python
PIP := $(VENV_NAME)/bin/pip

docs-prep:
	cd docs && $(UV) run python generate_docs.py && cd ..

docs:
	make docs-prep
	make -C docs html SPHINXBUILD="$(UV) run sphinx-build"
	# sphinx-apidoc -o docs ./src/devinterp ./src/devinterp/mechinterp --force 
	# sphinx-build -b html -E -a docs docs/_build/html

docs-auto:
	make docs-prep
	$(UV) run sphinx-autobuild docs docs/_build/html


publish-docs:
	cp -rf docs/_build/html/* ../devinterp-docs/public

test:
	. .venv/bin/activate && pytest tests/

%:
	@: