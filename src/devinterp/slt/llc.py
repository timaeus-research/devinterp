"""Local Learning Coefficient (LLC) computation from sampling results.

Computes LLC from the stored per-draw losses, without needing callbacks.

    LLC = n_beta * (mean_sampling_loss - init_loss)

Two entry points:
- llc(): high-level, takes model + dataset, runs sampling + LLC
- compute_llc(): low-level, takes pre-computed sampling DataTree
"""

from __future__ import annotations

import torch
import xarray as xr
from datasets import Dataset

from devinterp.slt.lm_loss import LossFn
from devinterp.slt.sampler import ParamMasks
from devinterp.slt.sampling import ObservableSpec, sample


def llc(
    model: torch.nn.Module,
    dataset: Dataset,
    observables: dict[str, ObservableSpec],
    *,
    lr: float,
    n_beta: float,
    param_masks: ParamMasks | None = None,
    loss_fn: LossFn | None = None,
    **kwargs,
) -> xr.Dataset:
    """Sample and compute LLC in one call.

    Args:
        model: PyTorch model.
        dataset: HuggingFace Dataset with "input_ids" column.
        observables: Dict mapping names to datasets (or (dataset, batches_per_draw) tuples).
        lr: SGLD learning rate.
        n_beta: SGLD inverse temperature.
        param_masks: Which parameters to optimize. None for full model.
        loss_fn: Optional custom per-token loss `(model, input_ids) -> (batch, seq-1)`.
            Defaults to cross-entropy on the model's logits.
        **kwargs: Additional arguments passed to sample() (num_chains, num_draws,
            batch_size, output_path, etc.)

    Returns:
        xr.Dataset with llc_mean, llc_std, llc_per_chain, loss_trace, init_loss.
    """
    samples = sample(
        model=model,
        dataset=dataset,
        observables=observables,
        param_masks=param_masks,
        lr=lr,
        n_beta=n_beta,
        loss_fn=loss_fn,
        **kwargs,
    )
    return compute_llc(samples)


def compute_llc(samples: xr.DataTree) -> xr.Dataset:
    """Compute LLC from a sampling DataTree.

    Matches aether's `calculate` action with `function: llc`: averages
    `sampling_loss_micro` over every step (including burn-in) and every
    micro-batch, then subtracts mean `init_loss` and scales by `n_beta`.

    Args:
        samples: DataTree output from sample(), containing sampling_loss_micro,
            init_loss, and n_beta.

    Returns:
        xr.Dataset with:
            llc_mean: scalar, mean LLC across chains
            llc_std: scalar, std LLC across chains
            llc_per_chain: (chain,) LLC per chain
            llc_scalar: scalar LLC matching aether's calculate action
            loss_trace: (chain, step) mean loss per chain per step
            init_loss: scalar, mean init loss
    """
    ds = samples.dataset
    n_beta = float(ds.metadata["config"]["sampler"]["n_beta"])
    init_loss_mean = float(ds.init_loss.values.mean())

    # Per-chain per-step mean loss (average over batch_sampling and token_pos)
    loss_trace = ds.sampling_loss_micro.mean(dim=["batch_sampling", "token_pos"])

    # LLC per chain = n_beta * (mean_loss_per_chain - init_loss)
    llc_per_chain = n_beta * (loss_trace.mean(dim="step") - init_loss_mean)  # (chain,)

    # Scalar LLC matching aether's calculate action reduction order
    # (all dims reduced at once for bitwise parity)
    llc_scalar = n_beta * (float(ds.sampling_loss_micro.mean()) - init_loss_mean)

    return xr.Dataset(
        {
            "llc_mean": llc_per_chain.mean(),
            "llc_std": llc_per_chain.std(),
            "llc_per_chain": llc_per_chain,
            "llc_scalar": llc_scalar,
            "loss_trace": loss_trace,
            "init_loss": init_loss_mean,
        }
    )
