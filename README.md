# DevInterp

[![PyPI version](https://badge.fury.io/py/devinterp.svg)](https://badge.fury.io/py/devinterp) ![Python version](https://img.shields.io/pypi/pyversions/devinterp) ![Contributors](https://img.shields.io/github/contributors/timaeus-research/devinterp) [![Docs](https://img.shields.io/badge/Read_the_Docs!-white?style=flat&logo=Read-the-Docs&logoColor=black&link=https%3A%2F%2Ftimaeus-research.github.io%2Fdevinterp%2F)](https://devinterp.timaeus.co/)


## A Python Library for Developmental Interpretability Research

DevInterp is a python library for conducting research on developmental interpretability, a novel AI safety research agenda rooted in Singular Learning Theory (SLT). DevInterp proposes tools for detecting, locating, and ultimately _controlling_ the development of structure over training.

[Read more about developmental interpretability](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/towards-developmental-interpretability).

## Features

- **SGLD Sampling** with per-token loss storage to xarray/Zarr
- **Local Learning Coefficient (LLC)** estimation from sampling results
- **Susceptibilities** measuring model response to distribution shifts across weight-restricted components
- **Bayesian Influence Functions (BIF)** computing pairwise correlations between observable loss traces
- **Observable framework** for evaluating multiple probe datasets during sampling
- **Weight restrictions** for sampling over parameter subsets (e.g., individual attention heads)

## Installation

To install `devinterp`, simply run `pip install devinterp`. (Note: This has PyTorch as a dependency.)

## Example

See [`examples/quickstart.py`](examples/quickstart.py) for a runnable script that computes LLC and susceptibilities on Qwen2.5-0.5B.

## Quick Start

### Compute the Local Learning Coefficient

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
print(result["loss_trace"])       # (num_chains, num_draws) loss trace
```

### Sample with Observables

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

### Compute Susceptibilities

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

`create_param_masks` supports 95+ HuggingFace model types and TransformerLens.
Restriction patterns: `"full"`, `"l0"`, `"l0h1"`, `"l0g0"` (GQA group), `"l0 attn"`, `"l0 mlp"`, `"embed"`, `"unembed"`.

### Compute BIF

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
- **Low-level** (`compute_llc()`, `compute_bif()`, `compute_susceptibilities()`): takes a pre-computed `xr.DataTree` from `sample()`, useful when you want to run sampling once and compute multiple analyses

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

## Known Issues

- LLC estimation is sensitive to hyperparameters. Always vary `lr`, `n_beta`, and `num_draws` to check robustness.
- Hyperparameters do change what we observe, and we don't have ground truth. Observables should theoretically be independent of hyperparameters, but in practice estimates are sensitive.

If you run into issues, please first check the GitHub issues, then ask in [the DevInterp Discord](https://discord.gg/UwjWKCZZYR).

## Further Reading

- [You're Measuring Model Complexity Wrong](https://www.lesswrong.com/posts/6g8cAftfQufLmFDYT/you-re-measuring-model-complexity-wrong) - Introduction to LLC and phase transitions
- [Structural Inference with Susceptibilities](https://arxiv.org/abs/2504.18274) - Susceptibility framework for interpretability (Baker et al., 2025)
- [Announcing Timaeus](https://www.lesswrong.com/posts/TjaeCWvLZtEDAS5Ex/announcing-timaeus) - Timaeus research program
- Lau et al. (2023) - Local learning coefficient estimator
- Watanabe (2009) - Algebraic Geometry and Statistical Learning Theory

## Credits & Citations

<!-- TODO: Update credits and citation for v2 release. The current citation
     reflects the original devinterp authors. The v2 sampling/susceptibilities/BIF
     pipeline was ported from aether and needs proper attribution. -->

This package was created by [Timaeus](https://timaeus.co).

If this package was useful in your work, please cite it as:

```BibTeX
@misc{devinterpcode,
  title = {DevInterp},
  author = {van Wingerden, Stan and Hoogland, Jesse and Wang, George and Zhou, William},
  year = {2024},
  howpublished = {\url{https://github.com/timaeus-research/devinterp}},
}
```
