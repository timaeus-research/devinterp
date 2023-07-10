# DevInterp

![DevInterp](assets/rm_devinterp_logo.png)

[![Pypi](https://img.shields.io/pypi/v/devinterp)](https://pypi.org/project/devinterp/)

## A Python Library for Developmental Interpretability Research

DevInterp is a Python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).

## Getting Started

For those new to developmental interpretability and our library, we suggest starting with our **[main tutorial](https://www.github.com/timaeus/devinterp/docs/tutorial)**. It provides an in-depth understanding of the library's functionalities and key features.

To see DevInterp in action, check out our [demo notebook](https://www.github.com/timaeus/devinterp/docs/demo.ipynb), which provides a hands-on approach to applying our tool to real problems.

## Installation

To install DevInterp, simply run:

```bash
pip install devinterp
```

You can then import the library in your Python environment with `import devinterp`.

## Local Development

### Manual Setup

To set up the project for local development, clone the repository from GitHub:

```bash
git clone https://github.com/timaeus-research/devinterp.git
cd devinterp
pip install -r requirements.txt
```

This installs all necessary dependencies for DevInterp.

### Testing

When contributing to DevInterp, please make sure to add corresponding unit tests and confirm that your changes haven't introduced any regressions by running existing tests. 

```bash
python -m unittest discover tests
```

### Formatting

We use `black` and `isort` for code formatting. Please make sure your contributions adhere to these standards.

```bash
black .
isort .
```

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
