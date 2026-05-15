"""Susceptibility computation.

Computes per-token susceptibilities from SGLD sampling results,
as described in appendix C.4 of https://arxiv.org/pdf/2504.18274.

Two entry points:
- susceptibilities(): high-level, takes model + datasets, runs sampling + post-processing
- compute_susceptibilities(): low-level, takes pre-computed sampling DataTrees
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import xarray as xr
from datasets import Dataset

from devinterp.slt.lm_loss import LossFn
from devinterp.slt.observables import to_obs_id
from devinterp.slt.sampler import ParamMasks
from devinterp.slt.sampling import ObservableSpec, sample


def susceptibilities(
    model: torch.nn.Module,
    dataset: Dataset,
    observables: dict[str, ObservableSpec],
    weight_restrictions: dict[str, ParamMasks | None],
    *,
    sampling_task: str,
    lr: float,
    n_beta: float,
    loss_fn: LossFn | None = None,
    include_sampling_task: bool = False,
    **kwargs,
) -> xr.DataTree:
    """Sample multiple weight restrictions and compute susceptibilities.

    Args:
        model: PyTorch model.
        dataset: HuggingFace Dataset with "input_ids" column.
        observables: Dict mapping names to datasets (or (dataset, batches_per_draw) tuples).
        weight_restrictions: Dict mapping WR names to param masks.
            Must include "full" (use None for full model).
        sampling_task: Name of the sampling dataset (must be in observables).
        lr: SGLD learning rate.
        n_beta: SGLD inverse temperature.
        loss_fn: Optional custom per-token loss `(model, input_ids) -> (batch, seq-1)`.
            Defaults to cross-entropy on the model's logits.
        include_sampling_task: If True, compute susceptibilities for the
            sampling_task observable too (per-token variation within that task
            is still informative). Default False.
        **kwargs: Additional arguments passed to sample() (num_chains, num_draws,
            batch_size, output_path, etc.). If output_path is provided, each
            weight restriction's samples are saved to a separate zarr with the
            WR name appended (e.g. "samples_full.zarr", "samples_l0h1.zarr").

    Returns:
        DataTree with /susceptibilities and /context subtrees.
    """
    if "full" not in weight_restrictions:
        raise ValueError(
            "weight_restrictions must include a 'full' key (use None for the full model)."
        )
    if sampling_task not in observables:
        raise ValueError(
            f"sampling_task '{sampling_task}' not found in observables. "
            f"Available: {list(observables)}"
        )
    if not include_sampling_task:
        non_sampling = [name for name in observables if name != sampling_task]
        if not non_sampling:
            raise ValueError(
                "Need at least one observable besides sampling_task "
                "(or set include_sampling_task=True)."
            )

    # If output_path is provided, append WR name so each sample() writes to a
    # unique zarr (otherwise they'd all overwrite the same path).
    base_output_path = kwargs.pop("output_path", None)

    wr_map: dict[str, xr.DataTree] = {}
    for wr_name, masks in weight_restrictions.items():
        if masks is None:
            masks = {name: None for name, _ in model.named_parameters()}
        wr_kwargs = dict(kwargs)
        if base_output_path is not None:
            base = Path(str(base_output_path))
            wr_kwargs["output_path"] = (
                base.parent / f"{base.stem}_{wr_name}{base.suffix}"
            )
        wr_map[wr_name] = sample(
            model=model,
            dataset=dataset,
            observables=observables,
            param_masks=masks,
            lr=lr,
            n_beta=n_beta,
            loss_fn=loss_fn,
            **wr_kwargs,
        )
    return compute_susceptibilities(
        wr_map, sampling_task, include_sampling_task=include_sampling_task
    )


def compute_susceptibilities(
    wr_map: dict[str, xr.DataTree],
    sampling_task: str,
    observable_names: list[str] | None = None,
    include_sampling_task: bool = False,
) -> xr.DataTree:
    """Compute per-token susceptibilities from sampling results.

    Args:
        wr_map: Dict mapping weight restriction names to DataTrees.
            Must include a "full" key for the unrestricted model.
            Each DataTree is the output of sample().
        sampling_task: Name of the sampling/pretraining dataset task
            (e.g. "pile10k"). Must appear as an observable.
        observable_names: Which observables to compute susceptibilities for.
            If None, uses all observables (optionally including sampling_task,
            see include_sampling_task).
        include_sampling_task: If True and observable_names is None, include
            sampling_task in the discovered observables. Default False.

    Returns:
        DataTree with /susceptibilities and /context subtrees.
    """
    full = wr_map["full"].dataset
    sampling_id = to_obs_id(sampling_task)

    # Discover observable ids from the DataTree
    all_obs_ids = [
        str(v)[len("loss_") :]
        for v in full.data_vars
        if str(v).startswith("loss_") and str(v) != "sampling_loss"
    ]
    if observable_names is None:
        obs_ids = (
            all_obs_ids
            if include_sampling_task
            else [oid for oid in all_obs_ids if oid != sampling_id]
        )
    else:
        obs_ids = [to_obs_id(n) for n in observable_names]

    if not obs_ids:
        raise ValueError(
            f"No probe observables to compute susceptibilities for. "
            f"Available loss_* observables: {all_obs_ids!r}; "
            f"sampling_task={sampling_id!r}; "
            f"include_sampling_task={include_sampling_task}. "
            f"Add a probe observable distinct from the sampling task, "
            f"or pass include_sampling_task=True."
        )

    # Pre-compute full-WR quantities
    full_sampling_mean = float(full[f"loss_{sampling_id}"].values.mean())
    init_loss_mean = float(full["init_loss"].values.mean())

    full_delta = {}
    for obs_id in obs_ids:
        # Mean over chain+draw, keeping (batch, target_position)
        full_delta[obs_id] = full_sampling_mean - full[f"loss_{obs_id}"].values.mean(
            axis=(0, 1)
        )

    # Compute susceptibilities per WR x observable
    results: list[tuple[str, str, xr.DataArray, xr.DataArray]] = []

    for wr_name, wr in wr_map.items():
        ds = wr.dataset

        # Verify expected layout since we slice positionally below.
        for oid in [sampling_id, *obs_ids]:
            expected = ("chain", "draw", f"batch_{oid}", "token_pos")
            assert ds[f"loss_{oid}"].dims == expected, (
                f"loss_{oid} has dims {ds[f'loss_{oid}'].dims}, expected {expected}"
            )

        # phi = per-(chain, draw) mean pretraining loss minus init_loss_mean
        pt_loss = ds[f"loss_{sampling_id}"].values  # (chain, draw, batch, token_pos)
        pt_mean = pt_loss.mean(axis=(-2, -1))  # (chain, draw)
        phi = pt_mean - init_loss_mean
        E_phi = float(phi.mean())
        E_phi_wr = float((phi * pt_mean).mean())

        for obs_id in obs_ids:
            probe = ds[f"loss_{obs_id}"].values  # (chain, draw, batch, token_pos)
            E_phi_probe = (phi[:, :, None, None] * probe).mean(axis=(0, 1))

            sus = E_phi_wr - E_phi_probe - E_phi * full_delta[obs_id]

            ctx_len = sus.shape[1]
            sus_da = xr.DataArray(
                sus,
                dims=["batch", "target_position"],
                coords={"target_position": np.arange(1, ctx_len + 1)},
            )

            ids = ds[f"input_ids_{obs_id}"].values
            while ids.ndim > 2:
                ids = ids[0]
            ids_da = xr.DataArray(
                ids,
                dims=["batch", "position"],
                coords={"position": np.arange(ids.shape[1])},
            )

            results.append((wr_name, obs_id, sus_da, ids_da))

    return _build_output(results)


# ─── Output format ──────────────────────────────────────────────────────────
# The output uses the SusceptibilitiesV2 format: susceptibilities are flattened
# into a 2D array (sus_flat, wr) and input_ids into a 1D array (ctx_flat),
# both with coordinate arrays for traceability. This matches aether's format
# so downstream visualization tools work unchanged.


def _build_output(
    results: list[tuple[str, str, xr.DataArray, xr.DataArray]],
) -> xr.DataTree:
    """Build SusceptibilitiesV2 DataTree from (wr, obs, sus, ids) tuples."""
    sus_by_wr: dict[str, dict[str, xr.DataArray]] = defaultdict(dict)
    all_ids: dict[str, dict[str, xr.DataArray]] = defaultdict(dict)
    wr_order: list[str] = []

    for wr_name, obs, sus, ids in results:
        sus_by_wr[wr_name][obs] = sus
        all_ids[wr_name][obs] = ids
        if wr_name not in wr_order:
            wr_order.append(wr_name)

    obs_order = sorted(sus_by_wr[wr_order[0]].keys())

    # Validate input_ids consistency across WRs
    first_wr = wr_order[0]
    deduped_ids: dict[str, xr.DataArray] = {}
    for obs in obs_order:
        ref = all_ids[first_wr][obs]
        deduped_ids[obs] = ref
        for wr in wr_order[1:]:
            if not np.array_equal(ref.values, all_ids[wr][obs].values):
                raise AssertionError(
                    f"input_ids for {obs} differ between {first_wr} and {wr}"
                )

    # Flatten input_ids → 1D with (dataset_id, batch, position) coords
    ids_items = {i: (obs, deduped_ids[obs]) for i, obs in enumerate(obs_order)}
    ids_coords = _flat_coords(ids_items, "ctx_flat")
    ids_data = np.concatenate([v.values.ravel() for _, v in ids_items.values()])

    # Flatten susceptibilities → 2D (sus_flat, wr)
    sus_items = {
        i: (obs, sus_by_wr[wr_order[0]][obs]) for i, obs in enumerate(obs_order)
    }
    sus_coords = _flat_coords(sus_items, "sus_flat")
    sus_data = np.concatenate(
        [
            np.stack([sus_by_wr[wr][obs].values.ravel() for wr in wr_order], axis=-1)
            for obs in obs_order
        ],
        axis=0,
    )

    return xr.DataTree.from_dict(
        {
            "/susceptibilities": xr.Dataset(
                {"sus": xr.DataArray(sus_data, dims=["sus_flat", "wr"])},
                coords=sus_coords.merge(xr.Coordinates({"wr": wr_order})),
                attrs={"dataset_id_to_name": obs_order},
            ),
            "/context": xr.Dataset(
                {"input_ids": xr.DataArray(ids_data, dims=["ctx_flat"])},
                coords=ids_coords,
                attrs={"dataset_id_to_name": obs_order},
            ),
        }
    )


def _flat_coords(
    items: dict[int, tuple[str, xr.DataArray]], dim_name: str
) -> xr.Coordinates:
    """Build flattened coordinates from per-observable DataArrays."""
    coords: dict[str, list] = {"dataset_id": []}
    for i, (_name, var) in items.items():
        coords["dataset_id"].append(np.full(var.size, i))
        grids = np.meshgrid(*[var.coords[d].values for d in var.dims], indexing="ij")
        for dim, grid in zip(var.dims, grids):
            coords.setdefault(str(dim), []).append(grid.ravel())
    return xr.Coordinates(
        {k: (dim_name, np.concatenate(v)) for k, v in coords.items()},
        indexes={},
    )
