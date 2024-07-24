```
cd docs
sphinx-apidoc -o . ../src/devinterp ../src/devinterp/mechinterp --force 
sphinx-autobuild . _build
```