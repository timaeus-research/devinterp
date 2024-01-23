.. devinterp documentation master file, created by
   sphinx-quickstart on Fri Jan 19 13:45:44 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to devinterp's documentation!
=====================================


DevInterp is a python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately *controlling* the development of structure over training.

`Read more about developmental interpretability <https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability>`_.

.. warning:: This library is still in early development. Don't expect things to work on a first attempt. We are actively working on improving the library and adding new features. If you have questions or suggestions, please feel free to open an issue or submit a pull request.

Installation
************

To install `devinterp`, simply run:

.. code-block:: bash

   pip install devinterp


**Requirements**: Python 3.8 or higher.

Getting Started
***************

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

For more advanced usage, see `the Diagnostics notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/diagnostics.ipynb>`_ |colab5| and for a quick guide on picking hyperparameters, see `the calibration notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb>`_ |colab5|


Minimal Example
***************

.. code-block:: python

   from devinterp.slt import sample, LLCEstimator
   from devinterp.optim import SGLD

   # Assuming you have a PyTorch Module and DataLoader
   llc_estimator = LLCEstimator(...)
   learning_coeff = sample(model, trainloader, ..., callbacks = [llc_estimator])
   
   llc_mean = llc_estimator.sample()["llc/mean"]
   llc_std = llc_estimator.sample()["llc/std"]
   loss_mean = llc_estimator.sample()["loss/mean"]


Known Issues
************

TODO 

Variations of LLC (eval on validation set, full dataset, gradient minibatch, new minibatch) -> Blocked on refactoring/cleaning up ICL

Variation on initial loss: optional argument to LLC callback for an init loss, otherwise fallback to batch (with warning) 

Custom forward pass or documentation that we encourage using a wrapper instead 

Contributing
************

See `CONTRIBUTING.md <https://github.com/timaeus-research/devinterp/blob/main/CONTRIBUTING.md>`_ for guidelines on how to contribute. For questions, please `hop into the DevInterp Discord <https://discord.gg/UwjWKCZZYR>`_!


Credits & Citations
********************

The main contributers to this package are Jesse Hoogland, Stan van Wingerden, and George Wang. Zach Furman, Matthew Farrugia-Roberts and Edmund Lau also had valuable contributions or provided useful advice.

If this package was useful in your work, please cite it as:

.. code-block:: 

   @misc{devinterp2024,
      title = {DevInterp},
      author = {Jesse Hoogland, Stan van Wingerden, and George Wang},
      year = {2024},
      howpublished = {\url{https://github.com/timaeus-research/devinterp}},
   }

.. toctree::
   :maxdepth: 4

   SLT Observables <devinterp.slt>
   Sampling Methods <devinterp.optim>
