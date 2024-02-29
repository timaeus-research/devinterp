.. devinterp documentation master file, created by
   sphinx-quickstart on Fri Jan 19 13:45:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to devinterp's documentation!
=====================================


DevInterp is a python library for conducting research on developmental interpretability, which is a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately *controlling* the development of structure over training.

Read more about `developmental interpretability here <https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability>`_!

For an overview of papers that either used or inspired this package, `click here <https://devinterp.com/publications>`_.

For questions, it's easiest to `join the DevInterp discord <https://discord.gg/UwjWKCZZYR>`_!

.. warning:: This library is still in early development. Don't expect things to work on a first attempt. We are actively working on improving the library and adding new features.

Installation
=====================================

To install :code:`devinterp`, simply run :bash:`pip install devinterp`.

**Requirements**: Python 3.8 or higher.

Minimal Example
=====================================

.. code-block:: python

   from devinterp.slt import sample, LLCEstimator
   from devinterp.optim import SGLD

   # Assuming you have a PyTorch Module and DataLoader
   llc_estimator = LLCEstimator(..., temperature=optimal_temperature(trainloader))
   sample(model, trainloader, ..., callbacks = [llc_estimator])
   
   llc_mean = llc_estimator.sample()["llc/mean"]

Examples
=====================================

To see DevInterp in action, check out our example notebooks:

.. |colab1| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/introduction.ipynb
.. |colab2| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb
.. |colab3| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/tms.ipynb
.. |colab4| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/mnist.ipynb
.. |colab5| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb
.. |colab6| image:: https://colab.research.google.com/assets/colab-badge.svg 
   :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb

- `Introduction <https://www.github.com/timaeus-research/devinterp/blob/main/examples/introduction.ipynb>`_ |colab1|
- `Normal Crossing degeneracy quantification <https://www.github.com/timaeus-research/devinterp/blob/main/examples/normal_crossing.ipynb>`_ |colab2|
- `Toy Models of Superposition phase transition detection <https://www.github.com/timaeus-research/devinterp/blob/main/examples/tms.ipynb>`_ |colab3|
- `MNIST eSGD vs SGD comparison <https://www.github.com/timaeus-research/devinterp/blob/main/examples/mnist.ipynb>`_ |colab4|

For more advanced usage, see `the Diagnostics notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb>`_ |colab5| and for a quick guide on picking hyperparameters, see `the calibration notebook. <https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb>`_ |colab6|


Known Issues
=====================================

- The current implementation does not work with transformers out-of-the-box. This can be fixed by adding a wrapper to your model, for example passing :python:`unpack(model)` to :func:`~devinterp.slt.sampler.sample` where :python:`unpack` is defined by:

.. code-block:: python

   class unpack(nn.Module):
    def __init__(model: nn.Module):
         self.model = model

    def forward(data: Tuple[torch.Tensor, torch.Tensor]):
         return self.model(*data)

- LLC Estimation is currently more of an art than a science. It will take some time and pain to get it work reliably.

If you run into issues not mentioned here, please first `check the GitHub issues <https://github.com/timaeus-research/devinterp/issues>`_, then `ask in the DevInterp discord <https://discord.gg/UwjWKCZZYR>`_, and only then make a new github issue.

Credits & Citations
=====================================

This package was created by `Timaeus <https://timaeus.co>`_. The main contributors to this package are Stan van Wingerden, Jesse Hoogland, and George Wang. Zach Furman, Matthew Farrugia-Roberts and Edmund Lau also made valuable contributions or provided useful advice.

If this package was useful in your work, please cite it as:

.. code-block:: 

   @misc{devinterp2024,
      title = {DevInterp},
      author = {Stan van Wingerden, Jesse Hoogland, and George Wang},
      year = {2024},
      howpublished = {\url{https://github.com/timaeus-research/devinterp}},
   }

Table of Contents
=====================================

.. toctree::
   :maxdepth: 4

   self
   SLT Observables <devinterp.slt>
   Sampling Methods <devinterp.optim>