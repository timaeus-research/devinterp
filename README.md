# DevInterp

[![PyPI version](https://badge.fury.io/py/devinterp.svg)](https://badge.fury.io/py/devinterp) ![Python version](https://img.shields.io/pypi/pyversions/devinterp) ![Contributors](https://img.shields.io/github/contributors/timaeus-research/devinterp) [![Docs](https://img.shields.io/badge/Read_the_Docs!-white?style=flat&logo=Read-the-Docs&logoColor=black&link=https%3A%2F%2Ftimaeus-research.github.io%2Fdevinterp%2F)](https://timaeus-research.github.io/devinterp/)


## A Python Library for Developmental Interpretability Research

DevInterp is a python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).


> :warning: This library is still in early development. Don't expect things to work on a first attempt. We are actively working on improving the library and adding new features. 

## Installation

 To install `devinterp`, simply run `pip install devinterp`.

### Minimal Example

```python

from devinterp.slt import sample, LLCEstimator
from devinterp.optim import SGLD

# Assuming you have a PyTorch Module and DataLoader
llc_estimator = LLCEstimator(...)
sample(model, trainloader, ..., callbacks = [llc_estimator])

llc_mean = llc_estimator.sample()["llc/mean"]

```

## Advanced Usage

To see DevInterp in action, check out our example notebooks:

- [Introduction](https://www.github.com/timaeus-research/devinterp/blob/main/examples/introduction.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/introduction.ipynb)
- [Normal Crossing Demo](https://www.github.com/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb)
- [Toy Models of Superposition](https://www.github.com/timaeus-research/devinterp/blob/main/examples/tms.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/tms.ipynb)
- [MNIST eSGD vs SGD comparison](https://www.github.com/timaeus-research/devinterp/blob/main/examples/mnist.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/mnist.ipynb)

For more advanced usage, see [the Diagnostics notebook](https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb) and for a quick guide on picking hyperparameters, see [the Calibration notebook.](https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb). Documentation can be [found here](https://timaeus-research.github.io/devinterp/). [![Docs](https://img.shields.io/badge/Read_the_Docs!-white?style=flat&logo=Read-the-Docs&logoColor=black&link=https%3A%2F%2Ftimaeus-research.github.io%2Fdevinterp%2F)](https://timaeus-research.github.io/devinterp/)

For papers that either inspired or used the DevInterp package, [click here](https://devinterp.com/publications).

## Known Issues

- We currently calculate the LLC taking the initial loss to be the loss after one sampling step. This is slightly wrong (it should be the loss before sampling), and there are a bunch of other reasonable and similarly compute-friendly alternative choices that can be made.

- Similarly, we now sample using minibatches that are passed along from the dataloader to sample(). This choice is obscured by the repo, and we should offer alternatives.

- The current implementation does not work with transformers out-of-the-box. This can be fixed by adding a wrapper to your model, for example passing unpack(model) to sample() where unpack is defined by:
```python
class unpack(nn.Module):
 def __init__(model: nn.Module):
      self.model = model

 def forward(data: Tuple[torch.Tensor, torch.Tensor]):
      return self.model(*data)
```
- LLC Estimation is currently more of an art than a science. It will take some time and pain to get it work reliably.

If you run into issues not mentioned here, please first check the github issues, then ask in [the DevInterp Discord](https://discord.gg/UwjWKCZZYR):, and only then make a new github issue.

## Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines on how to contribute. 

## Credits & Citations

This package was created by [Timaeus](https://timaeus.co). The main contributors to this package are Jesse Hoogland, Stan van Wingerden, and George Wang. Zach Furman, Matthew Farrugia-Roberts and Edmund Lau also made valuable contributions or provided useful advice.

If this package was useful in your work, please cite it as:

```BibTeX
   @misc{devinterp2024,
      title = {DevInterp},
      author = {Jesse Hoogland, Stan van Wingerden, and George Wang},
      year = {2024},
      howpublished = {\url{https://github.com/timaeus-research/devinterp}},
   }
```
