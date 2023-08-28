# DevInterp

![DevInterp](assets/devinterp_logo.png)

[![Pypi](https://img.shields.io/pypi/v/devinterp)](https://pypi.org/project/devinterp/)

## A Python Library for Developmental Interpretability Research

DevInterp is a Python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).

## Getting Started

For those new to developmental interpretability and our library, we suggest starting with our **[main tutorial](https://www.github.com/timaeus/devinterp/docs)**. It provides an in-depth understanding of the library's functionalities and key features.

To see DevInterp in action, check out our [demo notebook](https://www.github.com/timaeus/devinterp/demo/tutorial.ipynb), which provides a hands-on approach to applying our tool to real problems.

## Installation

To install DevInterp, simply run:

```bash
pip install devinterp
```

You can then import the library in your Python environment with `import devinterp`.

## Local Development

### Manual Setup
This project uses [Poetry](https://python-poetry.org/docs/#installation) for package management. Install as follows (this will also setup your virtual environment):

```bash
poetry config virtualenvs.in-project true
poetry install --with dev
```

Optionally, if you want Jupyter Lab you can run `poetry run pip install jupyterlab` (to install in the same virtual environment), and then run with `poetry run jupyter lab`.

Then the library can be imported as `import devinterp`.
### Testing

- All tests via `make test`
- Unit tests only via `make unit-test`
- Acceptance tests only via `make acceptance-test`

### Formatting

This project uses `pycln`, `isort` and `black` for formatting, pull requests are checked in github actions.

- Format all files via `make format`
- Only check the formatting via `make check-format`

## Citation

If you use DevInterp in your research, please cite us as:

```bibtex
@misc{timaeusdevinterp2023,
    title = {DevInterp},
    author = {Timaeus},
    url = {https://github.com/timaeus-research/devinterp},
    year = {2023}
}
```

If you're using DevInterp for your research, we'd love to hear from you! 
