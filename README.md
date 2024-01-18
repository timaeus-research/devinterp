# DevInterp

[![PyPI version](https://badge.fury.io/py/devinterp.svg)](https://badge.fury.io/py/devinterp) ![Python version](https://img.shields.io/pypi/pyversions/devinterp) ![Contributors](https://img.shields.io/github/contributors/timaeus-research/devinterp)



## A Python Library for Developmental Interpretability Research

DevInterp is a python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).

> :warning: This library is still in early development. Don't expect things to work on a first attempt. We are actively working on improving the library and adding new features. If you have questions or suggestions, please feel free to open an issue or submit a pull request.

## Installation

To install `devinterp`, simply run:

```bash
pip install devinterp
```

**Requirements**: Python 3.8 or higher.

## Getting Started

To see DevInterp in action, check out our example notebooks:

- [Introduction](https://www.github.com/timaeus-research/devinterp/blob/main/examples/introduction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/introduction.ipynb)
- [Normal Crossing degeneracy quantification](https://www.github.com/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb)
- [Toy Models of Superposition phase transition detection](https://www.github.com/timaeus-research/devinterp/blob/main/examples/tms.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/tms.ipynb)
- [MNIST eSGD vs SGD comparison](https://www.github.com/timaeus-research/devinterp/blob/main/examples/mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/mnist.ipynb)

### Minimal Example

```python
from devinterp.slt import estimate_learning_coeff, estimate_learning_coeff_with_summary
from devinterp.optim import SGLD

# Assuming you have a PyTorch Module and DataLoader
learning_coeff = estimate_learning_coeff(model, trainloader, ...)

# If you want to see mean, std, and learning coeff estimate per chain
learning_coeff_summary = estimate_learning_coeff_with_summary(model, trainloader, ...)

```

For more advanced usage, see [the Diagnostics notebook](https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb) and for a quick guide on picking hyperparameters, see [the Calibration notebook.](https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb)
## Known Issues
TODO 

Variations of LLC (eval on validation set, full dataset, gradient minibatch, new minibatch) -> Blocked on refactoring/cleaning up ICL

Variation on initial loss: optional argument to LLC callback for an init loss, otherwise fallback to batch (with warning) 

Custom forward pass or documentation that we encourage using a wrapper instead 

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute. For questions, please [hop into the DevInterp Discord](https://discord.gg/UwjWKCZZYR)!


## Credits & Citations

The main contributers to this package are Jesse Hoogland, Stan van Wingerden, and George Wang. Zach Furman, Matthew Farrugia-Roberts and Edmund Lau also had valuable contributions or provided useful advice.

If this package was useful in your work, please cite it as:

`@misc{devinterp2024,
    title = {DevInterp},
    author = {Jesse Hoogland, Stan van Wingerden, and George Wang},
    year = {202},
    howpublished = {\url{https://github.com/timaeus-research/devinterp}},
}`