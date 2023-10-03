# DevInterp

## A Python Library for Developmental Interpretability Research

DevInterp is a WIP Python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).

## Installation

To install `devinterp`, simply run:

```bash
pip install devinterp
```

**Requirements**: Python 3.9 or higher.

## Getting Started

To see DevInterp in action, check out our example notebooks:

- [Normal Crossing Demo](https://www.github.com/timaeus/devinterp/examples/normal_crossing.ipynb)
- [MNIST Demo](https://www.github.com/timaeus/devinterp/examples/mnist.ipynb)

### Minimal Example

```python
from devinterp.slt import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD

learning_coeff = estimate_learning_coeff_with_summary(model, trainloader, ...)
```

## Features

- Estimate the learning coefficient.
  - Supported optimizers: `SGHMC`, `SGLD`, `SGNHT`, `SVGD`

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute.
