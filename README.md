# DevInterp

[![PyPI version](https://badge.fury.io/py/devinterp.svg)](https://badge.fury.io/py/devinterp) ![Python version](https://img.shields.io/pypi/pyversions/devinterp) ![Contributors](https://img.shields.io/github/contributors/timaeus-research/devinterp) [![Docs](https://img.shields.io/badge/docs-devinterp.timaeus.co-blue?style=flat)](https://devinterp.timaeus.co/)

DevInterp is [Timaeus](https://timaeus.co)' open source research package, built to allow external researchers to do SLT/DevInterp-style research on Large Language Models.

## Features

- **SGLD Sampling** with per-token loss storage to xarray/Zarr
- **Local Learning Coefficient (LLC)** estimation from sampling results
- **Susceptibilities** measuring first-order posterior response to data perturbations, optionally restricted to specific model components
- **Bayesian Influence Functions (BIF)** as posterior correlations (or covariances) between per-sample losses
- **Weight restrictions** for sampling over parameter subsets (e.g., individual attention heads)

## Installation

`devinterp` is distributed through PyPI. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv add devinterp
```

## Example

See the [Quickstart Notebook](examples/quickstart.ipynb) ([open in Colab](https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/quickstart.ipynb)) or the [Quickstart Script](examples/quickstart.py) for examples of how to compute LLCs and susceptibilities on Qwen2.5-0.5B (GPU required).

## Quick Start

### Sampling with Observables

```python
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
```

### Computing the Local Learning Coefficient

```python
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
print(result["llc_per_chain"])    # (num_chains,) per-chain LLC
print(result["loss_trace"])       # (num_chains, num_steps) per-step loss, num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
```

### Computing Susceptibilities

```python
from devinterp.slt.susceptibilities import susceptibilities
from devinterp.slt.weight_restrictions import create_param_masks

result = susceptibilities(
    model=model,
    dataset=train_data,
    observables={"train": train_data, "code": code_data},
    weight_restrictions={
        "full": None,
        "l0h0": create_param_masks(model, "l0h0"),
        "l0h1": create_param_masks(model, "l0h1"),
    },
    sampling_task="train",
    lr=0.001,
    n_beta=30,
)
# result is a DataTree with /susceptibilities and /context subtrees
```

`create_param_masks` supports 85+ HuggingFace model types and TransformerLens.
Restriction patterns: `"full"`, `"l0"`, `"l0h1"`, `"l0g0"` (GQA group), `"l0 attn"`, `"l0 mlp"`, `"embed"`, `"unembed"`.

### Computing Bayesian Influence Functions

```python
from devinterp.slt.bif import bif

result = bif(
    model=model,
    dataset=train_data,
    observables={"train": train_data, "code": code_data},
    lr=0.001,
    n_beta=30,
    num_chains=4,
    num_draws=200,
    correlation_method="token",  # or "sequence"
)
# result["influences"] contains pairwise correlation matrix
```

## Architecture

Each analysis has two entry points:

- **High-level** (`llc()`, `bif()`, `susceptibilities()`): runs sampling and post-processing in one call
- **Low-level** (`compute_llc()`, `compute_bif()`): takes a pre-computed `xr.DataTree` from `sample()`, useful when you want to run sampling once and compute multiple analyses. `compute_susceptibilities()` takes a `dict[str, xr.DataTree]` (one tree per weight restriction), since susceptibilities require a separate sampling run for each restriction.

The sampling pipeline stores full per-token losses to Zarr via `sample()`, and post-processing functions operate on the resulting `xr.DataTree`.

## Model Requirements

The current API assumes **autoregressive language models** with fixed-length tokenized sequences:

- Model must accept `input_ids` and return logits (HuggingFace models, TransformerLens HookedTransformer, or any model returning a tensor or object with `.logits`)
- Dataset must be a HuggingFace `Dataset` with an `"input_ids"` column of uniform-length sequences
- Loss defaults to next-token cross-entropy

For non-standard losses, pass `loss_fn=...` to `sample()`, `bif()`, `llc()`, or `susceptibilities()`. The function takes `(model, input_ids)` and must return per-token loss of shape `(batch, seq_len-1)`. For more exotic control, `sample_single_chain()` in `devinterp.slt.sampler` accepts a custom `evaluate` callable.

## Migrating from v1

The v2 API replaces the callback-based sampling with a data-centric pipeline. Key changes:

```python
# v1 (old)
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim import SGLD

result = estimate_learning_coeff_with_summary(
    model, loader,
    sampling_method=SGLD,
    sampling_method_kwargs={"lr": 0.001, "nbeta": 30},
    num_chains=4, num_draws=200,
)
llc = result["llc/mean"]

# v2 (new)
from devinterp.slt.llc import llc

result = llc(
    model=model,
    dataset=dataset,                # HF Dataset, not DataLoader
    observables={"train": dataset},
    lr=0.001, n_beta=30,
    num_chains=4, num_draws=200,
)
llc_value = float(result["llc_mean"])
```

**What changed:**
- `estimate_learning_coeff` / `LLCEstimator` / `SamplerCallback` → `llc()` and `compute_llc()`
- `DataLoader` → HuggingFace `Dataset` with `"input_ids"` column
- `sampling_method_kwargs={"nbeta": ...}` → `n_beta=...` as a direct parameter
- Results are `xr.Dataset` / `xr.DataTree`, not dicts with string keys
- New capabilities: `susceptibilities()`, `bif()`, observables, weight restrictions, per-token loss storage

## Hyperparameter selection

All sampling is sensitive to hyperparameters. Our [Sampling Hyperparameter Guide](https://timaeus.co/research/2026-04-21-sampling-guide) covers the three primary knobs — step size (`lr`), inverse temperature (`n_beta`), and localization strength (`localization`) — along with burn-in, steps between draws, and chain count, and walks through diagnosing common failure modes (non-convergence, spikes, NaNs, low signal-to-noise) from the loss traces.


## Further Reading

Blog Posts:
- [Spectroscopy at Scale: Finding Interpretable Structure in Pythia-1.4B](https://timaeus.co/research/2026-04-21-spectroscopy-main) (2026)
- [Guide for Sampling Hyperparameter Selection](https://timaeus.co/research/2026-04-21-sampling-guide) (2026)

Papers:
- [Structural Inference with Susceptibilities](https://arxiv.org/abs/2504.18274) (2025)
- [Towards Spectroscopy: Susceptibility Clusters in Language Models](https://arxiv.org/abs/2601.12703) (2026)
- [The Local Learning Coefficient: A Singularity-Aware Complexity Measure](https://arxiv.org/pdf/2308.12108) (2023)

Background:
- [Algebraic Geometry and Statistical Learning Theory](https://www.cambridge.org/core/books/algebraic-geometry-and-statistical-learning-theory/9C8FD1BDC817E2FC79117C7F41544A3A#fndtn-information), Watanabe (2009)
- [Interpreting the Ising Model](https://timaeus.co/research/2026-04-21-spectroscopy-ising) (2026)
- [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong) (2024)

## Credits & Citations

This package was created by [Timaeus](https://timaeus.co). Most of the sampling, LLC, susceptibility, and BIF implementations were developed internally; this package is a port of that joint work.

If this package was useful in your work, please cite it as:

```BibTeX
@misc{devinterp2026,
  title   = {DevInterp},
  author  = {Snell, William and Wind, Johan Sokrates and Snikkers, Billy
             and Fraser, Sandy and Newgas, Adam and Hoogland, Jesse
             and Wang, George and Gordon, Andrew and Zhou, William
             and van Wingerden, Stan},
  year    = {2026},
  version = {2.0},
  howpublished = {\url{https://github.com/timaeus-research/devinterp}},
}
```

The authors would like to thank Zach Furman, Matthew Farrugia-Roberts, Rohan Hitchcock, and Edmund Lau for useful advice.

## About Timaeus

Timaeus is a non-profit advancing AI safety through research in Singular Learning Theory (SLT). We use SLT to understand how training data shapes AI behavior, combining deep mathematical insights from algebraic geometry and statistical physics with empirical research to develop interpretability tools for how capabilities and values emerge during neural network training. This foundational work enables us to build interventions that ensure models are aligned with human values.
