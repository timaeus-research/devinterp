Welcome to DevInterp's documentation!
=====================================

DevInterp is a Python library for conducting research on developmental interpretability,
a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp
proposes tools for detecting, locating, and ultimately *controlling* the development of
structure over training.

Read more about `developmental interpretability here <https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability>`_!

For questions, `join the DevInterp discord <https://discord.gg/UwjWKCZZYR>`_!

.. warning:: This library is under active development. The API may change between releases.

Installation
============

.. code-block:: bash

   pip install devinterp

**Requirements**: Python 3.8 or higher. PyTorch is a dependency.


Quick Start
===========

Compute the Local Learning Coefficient
---------------------------------------

.. code-block:: python

   from devinterp.slt.llc import llc

   result = llc(
       model=model,
       dataset=dataset,              # HuggingFace Dataset with "input_ids"
       observables={"train": dataset},
       lr=0.001,
       n_beta=30,
       num_chains=4,
       num_draws=200,
   )
   print(result["llc_mean"])         # scalar LLC
   print(result["loss_trace"])       # (num_chains, num_draws) loss trace


Sample with Observables
-----------------------

.. code-block:: python

   from devinterp.slt.sampling import sample

   tree = sample(
       model=model,
       dataset=train_data,
       observables={
           "train": train_data,
           "code": (code_data, 5),   # (dataset, batches_per_draw)
       },
       lr=0.001,
       n_beta=30,
       num_chains=4,
       num_draws=200,
   )
   # tree is an xr.DataTree backed by Zarr with full per-token loss traces


Concepts
========

Posterior Sampling with SGLD
----------------------------

The core workflow:

1. Start at a checkpoint :math:`\hat{w}^*`
2. Take SGLD steps (SGD + noise) using one dataset for gradients
3. Evaluate losses on multiple datasets (observables) at each draw
4. Store the full per-token loss chains as Zarr datasets
5. Compute observables (LLC, susceptibilities, BIF) from these chains

The SGLD noise allows exploring low-loss directions while staying near the original
checkpoint. This samples from the local posterior distribution around the checkpoint.

Local Learning Coefficient (LLC)
--------------------------------

The **LLC** measures model complexity by counting "effective parameters" in a region of
weight space:

.. math::

   \hat{\lambda}(\hat{w}^*) = n\beta \cdot (\bar{L}_n - L_n(\hat{w}^*))

Unlike parameter count or Hessian rank, LLC accounts for **singularities** -- regions where
multiple parameter configurations produce identical outputs. This makes it suitable for
neural networks.

**Why LLC matters:**

- **Detect phase transitions** during training (sudden capability changes)
- **Predict generalization** via the Free Energy formula
- **Compare checkpoints** across training

Susceptibilities
----------------

**Susceptibilities** measure how a model component responds to distribution shifts. For
example, how does an attention head's behavior change when shifting from general text toward
code or math?

This is computed by sampling with different **weight restrictions** (parameter subsets) and
measuring the covariance between sampling loss and observable loss.

See `Structural Inference: Interpreting Small Language Models with Susceptibilities
<https://arxiv.org/abs/2504.18274>`_ (Baker et al., 2025) for details.

Bayesian Influence Functions (BIF)
----------------------------------

**BIF** computes pairwise correlations between observable loss traces across sequences from
SGLD sampling results. This reveals which sequences influence each other's loss under
posterior sampling, providing a measure of functional similarity.


Architecture
============

Each analysis has two entry points:

- **High-level** (``llc()``, ``bif()``, ``susceptibilities()``): runs sampling and
  post-processing in one call
- **Low-level** (``compute_llc()``, ``compute_bif()``, ``compute_susceptibilities()``):
  takes a pre-computed ``xr.DataTree`` from ``sample()``, useful when you want to run
  sampling once and compute multiple analyses

The sampling pipeline stores full per-token losses to Zarr via ``sample()``, and
post-processing functions operate on the resulting ``xr.DataTree``.


Model Requirements
==================

The current API assumes **autoregressive language models** with fixed-length tokenized
sequences:

- Model must accept ``input_ids`` and return logits (HuggingFace models, TransformerLens
  ``HookedTransformer``, or any model returning a tensor or object with ``.logits``)
- Dataset must be a HuggingFace ``Dataset`` with an ``"input_ids"`` column of
  uniform-length sequences
- Loss is next-token cross-entropy

For non-standard models, ``sample_single_chain()`` in ``devinterp.slt.sampler`` accepts a
custom ``evaluate`` callable.


Known Issues
============

- LLC estimation is sensitive to hyperparameters. Always vary ``lr``, ``n_beta``, and
  ``num_draws`` to check robustness.
- Hyperparameters do change what we observe. Observables should theoretically be independent
  of hyperparameters, but in practice estimates are sensitive.

Further Reading
===============

- `You're Measuring Model Complexity Wrong <https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong>`_ - Introduction to LLC and phase transitions
- `Structural Inference with Susceptibilities <https://arxiv.org/abs/2504.18274>`_ - Susceptibility framework for interpretability (Baker et al., 2025)
- Lau et al. (2023) - Local learning coefficient estimator
- Watanabe (2009) - Algebraic Geometry and Statistical Learning Theory


Credits & Citations
===================

.. TODO: Update credits and citation for v2 release. The current citation
   reflects the original devinterp authors. The v2 sampling/susceptibilities/BIF
   pipeline was ported from aether and needs proper attribution.

This package was created by `Timaeus <https://timaeus.co>`_.

.. code-block:: bibtex

   @misc{devinterpcode,
     title = {DevInterp},
     author = {van Wingerden, Stan and Hoogland, Jesse and Wang, George and Zhou, William},
     year = {2024},
     howpublished = {\url{https://github.com/timaeus-research/devinterp}},
   }


Guides
======

.. toctree::
   :maxdepth: 2

   sampling
   output_formats


API Reference
=============

.. toctree::
   :maxdepth: 2

   SLT Analysis <source/devinterp.slt>
   Sampling Methods <source/devinterp.optim>
   Utilities <source/devinterp.utils>
