```
cd docs
sphinx-apidoc -o . ../src/devinterp ../src/devinterp/mechinterp --force sphinx-autobuild . _build
sphinx-build -b html -E -a . _build/html
```