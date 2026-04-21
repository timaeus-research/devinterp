"""Bayesian Influence Function (BIF) computation.

Computes pairwise correlations between observable loss traces across
sequences from SGLD sampling results.

Two entry points:
- bif(): high-level, takes model + dataset, runs sampling + BIF
- compute_bif(): low-level, takes pre-computed sampling DataTree
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Literal, Sequence

import numpy as np
import torch
import xarray as xr
from datasets import Dataset
from tqdm import tqdm

from devinterp.slt.covariance import (
    batch_corrcoef,
    batch_cov,
    xr_corrcoef_with_torch_backend,
)
from devinterp.slt.lm_loss import LossFn
from devinterp.slt.sampler import ParamMasks
from devinterp.slt.sampling import ObservableSpec, sample

CorrelationMethod = Literal["token", "sequence"]
ChainReductionMethod = Literal["stack", "mean"]


def bif(
    model: torch.nn.Module,
    dataset: Dataset,
    observables: dict[str, ObservableSpec],
    *,
    lr: float,
    n_beta: float,
    param_masks: ParamMasks | None = None,
    correlation_method: CorrelationMethod = "token",
    reduce_chain_dimension_method: ChainReductionMethod = "stack",
    average_tokenwise_bif: bool = False,
    compute_covariance: bool = False,
    bif_batch_size: int = 32,
    bif_device: str | torch.device | None = None,
    loss_fn: LossFn | None = None,
    **kwargs,
) -> xr.Dataset:
    """Sample and compute BIF in one call.

    Args:
        model: PyTorch model.
        dataset: HuggingFace Dataset with "input_ids" column.
        observables: Dict mapping names to datasets (or (dataset, batches_per_draw) tuples).
        lr: SGLD learning rate.
        n_beta: SGLD inverse temperature.
        param_masks: Which parameters to optimize. None for full model.
        correlation_method: "token" or "sequence".
        reduce_chain_dimension_method: "stack" (recommended) or "mean".
        average_tokenwise_bif: Average token-wise BIF to scalar per pair.
        compute_covariance: Compute covariance instead of correlation.
        bif_batch_size: Batch size for BIF block processing.
        bif_device: Torch device for BIF computation. None for auto-detect.
        loss_fn: Optional custom per-token loss `(model, input_ids) -> (batch, seq-1)`.
            Defaults to cross-entropy on the model's logits.
        **kwargs: Additional arguments passed to sample() (num_chains, num_draws,
            batch_size, output_path, etc.)

    Returns:
        xr.Dataset with "influences" and "input_ids" variables.
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
    return compute_bif(
        samples,
        correlation_method=correlation_method,
        reduce_chain_dimension_method=reduce_chain_dimension_method,
        average_tokenwise_bif=average_tokenwise_bif,
        compute_covariance=compute_covariance,
        batch_size=bif_batch_size,
        device=bif_device,
    )


def compute_bif(
    samples: xr.DataTree,
    *,
    correlation_method: CorrelationMethod = "token",
    reduce_chain_dimension_method: ChainReductionMethod = "stack",
    loss_keys: Literal["all"] | list[str] = "all",
    batch_index_range_1: Literal["all"] | Sequence[int] = "all",
    batch_index_range_2: Literal["all"] | Sequence[int] = "all",
    average_tokenwise_bif: bool = False,
    compute_covariance: bool = False,
    batch_size: int = 32,
    device: str | torch.device | None = None,
) -> xr.Dataset:
    """Compute BIF from a sampling DataTree.

    Args:
        samples: DataTree output from sample().
        correlation_method: "token" for token-wise, "sequence" for sequence-level.
        reduce_chain_dimension_method: "stack" (recommended) or "mean".
        loss_keys: Which observables to include. "all" auto-discovers.
        batch_index_range_1: Batch indices for first operand.
        batch_index_range_2: Batch indices for second operand.
        average_tokenwise_bif: Average token-wise BIF to scalar per pair.
        compute_covariance: Compute covariance instead of correlation.
        batch_size: Batch size for block processing.
        device: Torch device. None for auto-detect.

    Returns:
        xr.Dataset with "influences" and "input_ids" variables.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = samples.dataset
    resolved_keys = _resolve_loss_keys(ds, loss_keys)

    losses = _concatenate_observables(ds, resolved_keys, var_prefix="loss")
    losses = _adapt_to_correlation_method(losses, correlation_method)
    losses = _reduce_chains(losses, reduce_chain_dimension_method)

    losses_1 = (
        losses
        if batch_index_range_1 == "all"
        else losses.sel(batch=list(batch_index_range_1))
    )
    losses_2 = (
        losses
        if batch_index_range_2 == "all"
        else losses.sel(batch=list(batch_index_range_2))
    )

    bif_dataset = _compute_correlations(
        losses_1,
        losses_2,
        correlation_method=correlation_method,
        average_tokenwise_bif=average_tokenwise_bif,
        compute_covariance=compute_covariance,
        batch_size=batch_size,
        device=device,
    )

    # Attach input_ids
    input_ids = _concatenate_observables(ds, resolved_keys, var_prefix="input_ids")
    # Remove chain/draw dims (input_ids are fixed across draws)
    while input_ids.ndim > 2:
        input_ids = input_ids.isel({input_ids.dims[0]: 0})
    bif_dataset["input_ids"] = input_ids

    # Drop scalar coordinates from sampling metadata
    scalar_coords = [c for c in bif_dataset.coords if bif_dataset[c].ndim == 0]
    if scalar_coords:
        bif_dataset = bif_dataset.drop_vars(scalar_coords)

    return bif_dataset


# ─── Data preparation ────────────────────────────────────────────────────────


def _resolve_loss_keys(
    ds: xr.Dataset, loss_keys: Literal["all"] | list[str]
) -> list[str]:
    all_ids = [
        str(v)[len("loss_") :]
        for v in ds.data_vars
        if str(v).startswith("loss_") and str(v) != "sampling_loss"
    ]
    if loss_keys == "all":
        return sorted(all_ids)
    missing = [k for k in loss_keys if k not in all_ids]
    if missing:
        raise KeyError(f"Loss key(s) {missing} not found. Available: {all_ids}")
    return list(loss_keys)


def _normalize_dims(arr: xr.DataArray, obs_id: str) -> xr.DataArray:
    """Rename disambiguated dims back to standard names."""
    rename = {}
    for dim in arr.dims:
        d = str(dim)
        if d == f"batch_{obs_id}":
            rename[d] = "batch"
    if "token_pos" in arr.dims:
        rename["token_pos"] = "target_position"
    if "token" in arr.dims:
        rename["token"] = "position"
    return arr.rename(rename) if rename else arr


def _concatenate_observables(
    ds: xr.Dataset, keys: list[str], var_prefix: str
) -> xr.DataArray:
    """Concatenate observable arrays along batch, normalizing dim names."""
    arrays = [_normalize_dims(ds[f"{var_prefix}_{key}"], key) for key in keys]
    return xr.concat(arrays, dim="batch").load()


def _adapt_to_correlation_method(
    losses: xr.DataArray, method: CorrelationMethod
) -> xr.DataArray:
    if method == "sequence":
        losses = losses.mean(dim="target_position", keep_attrs=True)
        expected = ("chain", "draw", "batch")
    elif method == "token":
        expected = ("chain", "draw", "batch", "target_position")
    else:
        raise ValueError(f"Unknown correlation method: {method}")

    assert losses.dims == expected, f"Expected dims {expected}, got {losses.dims}"

    # Ensure integer coords
    updates = {
        dim: xr.DataArray(np.arange(losses.sizes[dim]), dims=(dim,))
        for dim in expected
        if dim not in losses.coords or losses[dim].dims != (dim,)
    }
    return losses.assign_coords(updates) if updates else losses


def _reduce_chains(losses: xr.DataArray, method: ChainReductionMethod) -> xr.DataArray:
    if method == "stack":
        return losses.stack(chain_draw=("chain", "draw"))
    elif method == "mean":
        losses = losses.mean(dim="chain", keep_attrs=True).rename(
            {"draw": "chain_draw"}
        )
        dims = (
            ["batch", "target_position", "chain_draw"]
            if "target_position" in losses.dims
            else ["batch", "chain_draw"]
        )
        return losses.transpose(*dims)
    else:
        raise ValueError(f"Unknown method: {method}")


# ─── Correlation computation ────────────────────────────────────────────────


def _compute_correlations(
    losses_1: xr.DataArray,
    losses_2: xr.DataArray,
    *,
    correlation_method: CorrelationMethod,
    average_tokenwise_bif: bool,
    compute_covariance: bool,
    batch_size: int,
    device: str | torch.device,
) -> xr.Dataset:
    coords_1 = np.asarray(losses_1.coords["batch"].values)
    coords_2 = np.asarray(losses_2.coords["batch"].values)

    if correlation_method == "token":
        n_tokens = losses_1.sizes["target_position"]
        correlations = _tokenwise_bif(
            losses_1,
            losses_2,
            batch_size=batch_size,
            average=average_tokenwise_bif,
            covariance=compute_covariance,
            device=device,
        )

        dims = ["batch_1", "batch_2"]
        coords: dict[str, np.ndarray] = {"batch_1": coords_1, "batch_2": coords_2}
        if not average_tokenwise_bif:
            dims += ["target_position", "target_position_T"]
            coords["target_position"] = np.arange(1, n_tokens + 1)
            coords["target_position_T"] = np.arange(1, n_tokens + 1)

        return xr.DataArray(
            correlations, name="influences", dims=dims, coords=coords
        ).to_dataset()

    elif correlation_method == "sequence":
        n1 = losses_1.sizes["batch"]
        result = xr_corrcoef_with_torch_backend(
            losses_1, losses_2, device=device, compute_covariance=compute_covariance
        )
        result = result.isel(
            batch=slice(0, n1), batch_T=slice(n1, n1 + losses_2.sizes["batch"])
        )
        result = result.rename(
            {"correlation": "influences", "batch": "batch_1", "batch_T": "batch_2"}
        )
        return result.assign_coords(
            batch_1=("batch_1", coords_1), batch_2=("batch_2", coords_2)
        )

    else:
        raise ValueError(f"Unknown correlation method: {correlation_method}")


@torch.no_grad()
def _tokenwise_bif(
    losses_1: xr.DataArray,
    losses_2: xr.DataArray,
    *,
    batch_size: int,
    average: bool,
    covariance: bool,
    device: str | torch.device,
) -> np.ndarray:
    n1, n2 = losses_1.sizes["batch"], losses_2.sizes["batch"]
    n_tokens = losses_1.sizes["target_position"]

    shape = (n1, n2) if average else (n1, n2, n_tokens, n_tokens)
    result = np.empty(shape, dtype=np.float32)

    for s1, s2, block in tqdm(
        _iter_blocks(losses_1, losses_2, batch_size, covariance, device),
        desc="Computing token-wise BIF",
        unit="batch",
    ):
        if average:
            block = block.mean(dim=(-1, -2))
        result[s1, s2, ...] = block.cpu().numpy()

    return result


def _iter_blocks(
    losses_1: xr.DataArray,
    losses_2: xr.DataArray,
    batch_size: int,
    covariance: bool,
    device: str | torch.device,
) -> Iterator[tuple[slice, slice, torch.Tensor]]:
    n1, n2 = losses_1.sizes["batch"], losses_2.sizes["batch"]
    n_tokens = losses_1.sizes["target_position"]
    corr_fn = batch_cov if covariance else batch_corrcoef

    for i in range(0, n1, batch_size):
        s1 = slice(i, i + batch_size)
        t1 = torch.as_tensor(losses_1.isel(batch=s1).data, device=device)
        for j in range(0, n2, batch_size):
            s2 = slice(j, j + batch_size)
            t2 = torch.as_tensor(losses_2.isel(batch=s2).data, device=device)
            block = corr_fn(t1, t2)[:, :, :n_tokens, n_tokens:]
            yield s1, s2, block
            del t2, block
        del t1
