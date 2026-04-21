"""SGLD sampling with observables, writing results to zarr.

Provides sample() as the main entry point. Internally uses
sample_single_chain from sampler.py for the SGLD inner loop.
"""

from __future__ import annotations

import logging
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any, cast

import torch
import xarray as xr
import zarr
from datasets import Dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from zarr.storage import LocalStore

from devinterp.optim.metrics import Metrics
from devinterp.slt.config import (
    SAMPLING_METHODS,
    SamplerConfig,
    SamplingMethodLiteral,
)
from devinterp.slt.lm_loss import LossFn, compute_per_token_loss, make_evaluate_fn
from devinterp.slt.observables import Observable
from devinterp.slt.sampler import (
    ParamMasks,
    _make_feed,
    is_transformer_lens_model,
    set_seed,
    calculate_num_steps,
    sample_single_chain,
)
from devinterp.slt.writing import ZarrWriter
from devinterp.slt.zarr_schema import DataArraySpec, ZarrSchema

SAMPLES_LOSS_DTYPE_STR = "float32"
ZARR_MAX_WRITE_THREADS = 4

logger = logging.getLogger(__name__)

# Type for observable specification: dataset alone (uses default batches_per_draw)
# or (dataset, batches_per_draw) tuple for explicit control.
ObservableSpec = Dataset | tuple[Dataset, int]


def sample(
    model: torch.nn.Module,
    dataset: Dataset,
    observables: dict[str, ObservableSpec],
    *,
    lr: float,
    n_beta: float,
    param_masks: ParamMasks | None = None,
    num_chains: int = 4,
    num_draws: int = 200,
    batch_size: int = 32,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    num_init_loss_batches: int = 32,
    init_seed: int = 100,
    batches_per_draw: int = 3,
    obs_seed: int = 1337,
    gradient_accumulation_steps: int = 1,
    localization: float = 0.0,
    noise_level: float = 1.0,
    llc_weight_decay: float = 0.0,
    bounding_box_size: float | None = None,
    sampling_method: SamplingMethodLiteral = "sgmcmc_sgld",
    sampling_method_kwargs: dict[str, Any] | None = None,
    rmsprop_eps: float | None = None,
    rmsprop_alpha: float | None = None,
    shuffle: bool = True,
    match_sampling_input_ids_across_chains: bool = True,
    init_noise: float | None = None,
    device: str | None = None,
    save_metrics: bool = False,
    output_path: str | Path | None = None,
    loss_fn: LossFn | None = None,
) -> xr.DataTree:
    """Run SGLD sampling with observables.

    Args:
        model: PyTorch model.
        dataset: HuggingFace Dataset with "input_ids" column, used for SGLD sampling.
        observables: Dict mapping observable names to datasets (or (dataset, batches_per_draw)
            tuples). Each dataset must have an "input_ids" column.
        lr: SGLD learning rate.
        n_beta: SGLD inverse temperature.
        param_masks: Which parameters to optimize. None means all parameters (full model).
            Otherwise a dict mapping param names to mask tensors (or None for unrestricted).
        num_chains: Number of SGLD chains.
        num_draws: Number of draws per chain.
        batch_size: Batch size for sampling and observables.
        num_burnin_steps: SGLD burn-in steps before drawing.
        num_steps_bw_draws: Steps between draws.
        num_init_loss_batches: Batches for init_loss computation.
        init_seed: Random seed.
        batches_per_draw: Default batches_per_draw for observables (used when
            an observable is specified as just a dataset, not a tuple).
        obs_seed: Seed for deterministic observable sampling.
        gradient_accumulation_steps: Number of micro-batches per optimizer step.
            Effective batch size is batch_size * gradient_accumulation_steps.
        localization: Strength of the pull toward initial parameters (gamma in
            Lau et al. 2023). 0 disables localization.
        noise_level: Standard deviation of SGLD noise. Defaults to 1.0;
            changing this breaks the SGLD posterior-sampling guarantee.
        llc_weight_decay: L2 regularization strength (lambda). Applied as a
            Gaussian prior centered at zero.
        bounding_box_size: If set, restricts sampling to a box of this radius
            around the initial parameters. None disables.
        sampling_method: Which SGLD variant to use. "sgmcmc_sgld" is the
            default; "rmsprop_sgld" adds RMSprop-style preconditioning.
        sampling_method_kwargs: Extra kwargs forwarded to the sampling-method
            constructor (e.g. rmsprop's "alpha" / "eps", or
            "add_grad_correction"). Use `rmsprop_eps` / `rmsprop_alpha` as
            convenience aliases for the two most common rmsprop knobs.
        rmsprop_eps: RMSprop stability constant. Only valid when
            sampling_method='rmsprop_sgld'. Shorthand for
            sampling_method_kwargs={"eps": ...}.
        rmsprop_alpha: RMSprop moving-average coefficient. Only valid when
            sampling_method='rmsprop_sgld'. Shorthand for
            sampling_method_kwargs={"alpha": ...}.
        shuffle: Whether to shuffle the sampling dataset. Default True.
        match_sampling_input_ids_across_chains: If True, every chain sees the
            same input_ids in the same order (only the SGLD noise differs
            across chains). If False, each chain gets an independently-seeded
            shuffle.
        init_noise: If set, add Gaussian noise with this std to parameters before sampling.
        device: Compute device. None for auto-detect.
        save_metrics: If True, save per-step SGLD diagnostics (gradient norms,
            noise norms, distance from init, etc.) for tuning sampling parameters.
        output_path: Path for output zarr. None for a temp directory.
        loss_fn: Optional custom per-token loss `(model, input_ids) -> (batch, seq-1)`.
            Defaults to cross-entropy on the model's logits.

    Returns:
        Lazy-loaded DataTree of sampling results.
    """
    start = time.perf_counter()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if param_masks is None:
        param_masks = {name: None for name, _ in model.named_parameters()}

    context_length = len(dataset[0]["input_ids"]) - 1
    if context_length < 1:
        raise ValueError(
            f"Sequences must have length >= 2 for next-token loss "
            f"(got {context_length + 1})."
        )
    ds = cast(TorchDataset, dataset)

    sampling_method_kwargs = dict(sampling_method_kwargs or {})
    for name, value in (("eps", rmsprop_eps), ("alpha", rmsprop_alpha)):
        if value is None:
            continue
        if sampling_method != "rmsprop_sgld":
            raise ValueError(
                f"rmsprop_{name} can only be set when sampling_method='rmsprop_sgld', "
                f"got sampling_method={sampling_method!r}"
            )
        if name in sampling_method_kwargs:
            raise ValueError(
                f"rmsprop_{name} is set both as a top-level argument and in "
                f"sampling_method_kwargs[{name!r}]; specify only one."
            )
        sampling_method_kwargs[name] = value

    # Build observable objects
    obs_list: list[Observable] = []
    for name, spec in observables.items():
        if isinstance(spec, tuple):
            obs_ds, bpd = spec
        else:
            obs_ds, bpd = spec, batches_per_draw
        obs_list.append(
            Observable(
                dataset=obs_ds,
                task_name=name,
                batches_per_draw=bpd,
                batch_size=batch_size,
                context_length=context_length,
                device=torch.device(device),
                seed=obs_seed,
                loss_fn=loss_fn,
            )
        )

    # Build config
    config = SamplerConfig(
        lr=lr,
        n_beta=n_beta,
        num_chains=num_chains,
        num_draws=num_draws,
        batch_size=batch_size,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        num_init_loss_batches=num_init_loss_batches,
        init_seed=init_seed,
        gradient_accumulation_steps=gradient_accumulation_steps,
        localization=localization,
        noise_level=noise_level,
        llc_weight_decay=llc_weight_decay,
        bounding_box_size=bounding_box_size,
        sampling_method=sampling_method,
        sampling_method_kwargs=sampling_method_kwargs,
        shuffle=shuffle,
        match_sampling_input_ids_across_chains=match_sampling_input_ids_across_chains,
        init_noise=init_noise,
        save_metrics=save_metrics,
    )

    # Cache: if output_path exists, validate and return early on match.
    if output_path is not None and Path(output_path).exists():
        cached = _check_cache(output_path, config)
        logger.info("sample() using cached output at %s", output_path)
        return cached

    # Warn about non-persisted big runs
    total_work = (
        num_chains * (num_draws * num_steps_bw_draws + num_burnin_steps) * batch_size
    )
    if output_path is None and total_work > 1000:
        warnings.warn(
            f"Sampling without output_path set -- {total_work} effective "
            "samples will be written to a temp directory and lost when "
            "the process exits. Pass output_path='/path/to/samples.zarr' "
            "to save them.",
            stacklevel=2,
        )

    # Compute numel for metrics if needed
    numel: int | None = None
    if save_metrics:
        name_to_param = dict(model.named_parameters())
        numel = sum(
            int(mask.count_nonzero())
            if mask is not None
            else name_to_param[name].numel()
            for name, mask in param_masks.items()
        )

    # Build zarr schema and store
    chain_buffer_size = min(50, num_draws)
    schema = _build_sampling_schema(
        config=config,
        context_length=context_length,
        chain_buffer_size=chain_buffer_size,
        observables=obs_list,
        numel=numel,
    )

    if output_path is None:
        output_path = Path(tempfile.mkdtemp()) / "samples.zarr"
    store = LocalStore(output_path)
    _, arrays = schema.create_hierarchy(store)

    # Resolve sampling method
    sampling_method_cls = SAMPLING_METHODS.get(config.sampling_method)
    if sampling_method_cls is None:
        raise ValueError(f"Unknown sampling method {config.sampling_method}")

    sampling_method_kwargs = dict(
        nbeta=n_beta,
        lr=lr,
        localization=config.localization,
        noise_level=config.noise_level,
        weight_decay=config.llc_weight_decay,
        bounding_box_size=config.bounding_box_size,
        save_metrics=save_metrics,
        **config.sampling_method_kwargs,
    )

    # Callbacks for zarr writing
    def on_draw(*, loss, draw, chain, model, **_):
        writer.push("sampling_loss", loss, chain=chain, draw=draw)
        for obs in obs_list:
            assert not torch.is_grad_enabled()
            obs_loss = obs.compute_loss(model)
            writer.push(f"loss_{obs.obs_id}", obs_loss, chain=chain, draw=draw)
        writer.flush_full_buffers()

    # Buffer grad_accum micro-batches per (chain, step) and write once full,
    # since the zarr writer expects a full row per push.
    micro_loss_buf: torch.Tensor | None = None
    micro_ids_buf: torch.Tensor | None = None
    micro_total = gradient_accumulation_steps * batch_size

    def on_micro(
        loss: torch.Tensor,
        input_ids: torch.Tensor,
        chain: int,
        step: int,
        micro_step: int,
    ) -> None:
        nonlocal micro_loss_buf, micro_ids_buf
        if micro_step == 0:
            micro_loss_buf = torch.empty(micro_total, context_length, dtype=loss.dtype)
            micro_ids_buf = torch.empty(
                micro_total, context_length + 1, dtype=input_ids.dtype
            )
        s = slice(micro_step * batch_size, (micro_step + 1) * batch_size)
        assert micro_loss_buf is not None and micro_ids_buf is not None
        micro_loss_buf[s] = loss.cpu()
        micro_ids_buf[s] = input_ids
        if micro_step == gradient_accumulation_steps - 1:
            writer.push("sampling_loss_micro", micro_loss_buf, chain=chain, step=step)
            writer.push(
                "sampling_input_ids_micro", micro_ids_buf, chain=chain, step=step
            )
            writer.flush_full_buffers()

    def on_step(chain: int, step: int, optimizer: torch.optim.Optimizer) -> None:
        metrics = optimizer.get_metrics()
        for field_name in Metrics.NORM_FIELDS + Metrics.DOT_FIELDS:
            value = getattr(metrics, field_name).squeeze()
            writer.push(f"metrics_{field_name}", value, chain=chain, step=step)
        writer.flush_full_buffers()

    step_callback = on_step if save_metrics else None

    # Seed and chain setup
    dataloader_seed = (
        init_seed if config.match_sampling_input_ids_across_chains else None
    )

    with ZarrWriter.open(
        arrays, chain_buffer_size, torch.device(device), ZARR_MAX_WRITE_THREADS
    ) as writer:
        # Write fixed observable input_ids
        for obs in obs_list:
            writer.write(f"input_ids_{obs.obs_id}", obs.input_ids)

        # Compute and write init loss
        _write_init_loss(writer, config, model, param_masks, ds, device, loss_fn)

        # Run SGLD chains
        for chain_idx in range(num_chains):
            sample_single_chain(
                ref_model=model,
                dataset=ds,
                evaluate=make_evaluate_fn(loss_fn),
                param_masks=param_masks,
                num_draws=num_draws,
                num_burnin_steps=num_burnin_steps,
                num_steps_bw_draws=num_steps_bw_draws,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                sampling_method=sampling_method_cls,
                sampling_method_kwargs=sampling_method_kwargs,
                chain=chain_idx,
                seed=init_seed + chain_idx,
                dataloader_seed=dataloader_seed
                if dataloader_seed is not None
                else init_seed + chain_idx,
                device=device,
                callbacks=[on_draw],
                step_callback=step_callback,
                micro_callback=on_micro,
                batch_size=batch_size,
                init_noise=init_noise,
                shuffle=config.shuffle,
                epoch_mode=config.epoch_mode,
            )

    # Mark as completed so future cache checks can trust the output.
    zarr.open_group(str(output_path)).attrs["completed"] = 1

    logger.info("sample() total time: %.2f seconds", time.perf_counter() - start)
    return xr.open_datatree(str(output_path), engine="zarr", consolidated=False)


def _check_cache(output_path: str | Path, config: SamplerConfig) -> xr.DataTree:
    """Validate an existing sample output against the current config.

    Raises RuntimeError with a clear "delete and retry" message if the file
    is unreadable, incomplete, or was produced with different sampler args.
    Otherwise returns the loaded DataTree.
    """
    path_str = str(output_path)
    try:
        existing = xr.open_datatree(path_str, engine="zarr", consolidated=False)
    except Exception as e:
        raise RuntimeError(
            f"Output path '{output_path}' exists but couldn't be opened as zarr:\n"
            f"  {e!r}\n"
            f"Delete and retry: rm -rf '{output_path}'"
        ) from e

    if existing.attrs.get("completed") != 1:
        raise RuntimeError(
            f"Output path '{output_path}' exists but sampling was incomplete "
            f"(no 'completed' flag — likely interrupted).\n"
            f"Delete and retry: rm -rf '{output_path}'"
        )

    stored_sampler = existing.metadata["config"]["sampler"]
    expected_sampler = {
        **config.model_dump(),
        "scheduler_type": "constant",
        "scheduler_kwargs": None,
        "cores": 1,
        "online": False,
    }
    if stored_sampler != expected_sampler:
        diffs = []
        all_keys = set(stored_sampler) | set(expected_sampler)
        for key in sorted(all_keys):
            s = stored_sampler.get(key)
            e = expected_sampler.get(key)
            if s != e:
                diffs.append(f"  {key}: stored={s!r}, current={e!r}")
        raise RuntimeError(
            f"Output path '{output_path}' has a different sampler config:\n"
            + "\n".join(diffs)
            + f"\nDelete and retry: rm -rf '{output_path}'"
        )

    return existing


# ─── Internal helpers ────────────────────────────────────────────────────────


def _build_sampling_schema(
    *,
    config: SamplerConfig,
    context_length: int,
    chain_buffer_size: int,
    observables: list[Observable],
    numel: int | None = None,
) -> ZarrSchema:
    """Build a ZarrSchema for the sampling pipeline."""
    num_chains = config.num_chains
    num_draws = config.num_draws

    chain_chunk_size = 1
    draw_chunk_size = chain_buffer_size
    chain_draw_chunks = (chain_chunk_size, draw_chunk_size)

    num_steps = calculate_num_steps(
        num_draws=num_draws,
        num_steps_bw_draws=config.num_steps_bw_draws,
        num_burnin_steps=config.num_burnin_steps,
    )

    arrays_meta: dict[str, DataArraySpec] = {}

    arrays_meta["init_loss"] = DataArraySpec(
        dtype_str=SAMPLES_LOSS_DTYPE_STR,
        dims=("chain", "token_pos"),
        shape=(num_chains, context_length),
        chunks=(1, context_length),
    )

    arrays_meta["sampling_loss"] = DataArraySpec(
        dtype_str=SAMPLES_LOSS_DTYPE_STR,
        dims=("chain", "draw", "batch", "token_pos"),
        shape=(num_chains, num_draws, config.batch_size, context_length),
        chunks=chain_draw_chunks + (config.batch_size, context_length),
    )

    micro_batch = config.gradient_accumulation_steps * config.batch_size
    arrays_meta["sampling_loss_micro"] = DataArraySpec(
        dtype_str=SAMPLES_LOSS_DTYPE_STR,
        dims=("chain", "step", "batch_sampling", "token_pos"),
        shape=(num_chains, num_steps, micro_batch, context_length),
        chunks=chain_draw_chunks + (micro_batch, context_length),
    )
    arrays_meta["sampling_input_ids_micro"] = DataArraySpec(
        dtype_str="int64",
        dims=("chain", "step", "batch_sampling", "token"),
        shape=(num_chains, num_steps, micro_batch, context_length + 1),
        chunks=chain_draw_chunks + (micro_batch, context_length + 1),
    )

    for obs in observables:
        batch_dim = f"batch_{obs.obs_id}"

        arrays_meta[f"loss_{obs.obs_id}"] = DataArraySpec(
            dtype_str=SAMPLES_LOSS_DTYPE_STR,
            dims=("chain", "draw", batch_dim, "token_pos"),
            shape=(num_chains, num_draws, obs.n_samples, context_length),
            chunks=chain_draw_chunks + (obs.n_samples, context_length),
        )

        arrays_meta[f"input_ids_{obs.obs_id}"] = DataArraySpec(
            dtype_str="int64",
            dims=(batch_dim, "token"),
            shape=(obs.n_samples, context_length + 1),
        )

    group_attrs: dict[str, dict[str, Any]] = {
        "": {
            "metadata": {
                "config": {
                    "sampler": {
                        **config.model_dump(),
                        # Fields we removed but can provide defaults for
                        "scheduler_type": "constant",
                        "scheduler_kwargs": None,
                        "cores": 1,
                        "online": False,
                    },
                    "observables": [
                        {
                            "type": "ObserveDifferentDataset",
                            "obs_id": obs.obs_id,
                            "batches_per_draw": obs.batches_per_draw,
                        }
                        for obs in observables
                    ],
                    "chain_buffer_size": chain_buffer_size,
                    "shared_observable_dims": ["token", "token_pos"],
                    "tokenizer_context_length": context_length + 1,
                },
            },
        }
    }

    if config.save_metrics:
        step_chunk_size = chain_buffer_size
        chain_step_chunks = (chain_chunk_size, step_chunk_size)
        for field_name in Metrics.NORM_FIELDS + Metrics.DOT_FIELDS:
            arrays_meta[f"metrics_{field_name}"] = DataArraySpec(
                dtype_str="float32",
                dims=("chain", "step"),
                shape=(num_chains, num_steps),
                chunks=chain_step_chunks,
            )
        assert numel is not None, "numel must be provided when save_metrics=True"
        group_attrs[""]["metrics_numel"] = numel

    return ZarrSchema(
        arrays_meta=arrays_meta,
        group_attrs=group_attrs,
    )


def _write_init_loss(
    writer: ZarrWriter,
    config: SamplerConfig,
    model: torch.nn.Module,
    param_masks: ParamMasks,
    dataset: TorchDataset,
    device: str,
    loss_fn: LossFn | None,
) -> None:
    """Compute and write the initial (pre-sampling) loss for each chain."""
    num_batches = config.gradient_accumulation_steps * config.num_init_loss_batches
    fn = loss_fn if loss_fn is not None else compute_per_token_loss

    # Remember original device so we can restore after init_loss computation.
    # nn.Module.to() is in-place — without this, the caller's model would be
    # silently moved to the compute device.
    orig_device = next(model.parameters()).device
    was_training = model.training

    if is_transformer_lens_model(model):
        model_on_device = model.to(device, print_details=False)  # pyright: ignore[reportCallIssue]
    else:
        model_on_device = model.to(device)
    model_on_device.train()

    torch.manual_seed(config.init_seed)

    name_to_param = dict(model_on_device.named_parameters())
    orig_data: dict[str, torch.Tensor] = {}
    if config.init_noise:
        # init_loss is computed at the unrestricted-noise state (noise on all
        # params, ignoring weight_restrictions) to match aether's behavior.
        # The sampling chain itself still applies masked init_noise.
        orig_data = {name: p.data.clone() for name, p in name_to_param.items()}

    for chain_idx in range(config.num_chains):
        set_seed(config.init_seed + chain_idx, device=device)

        dataloader_rng = torch.Generator(device="cpu")
        dataloader_rng.manual_seed(
            config.init_seed
            if config.match_sampling_input_ids_across_chains
            else config.init_seed + chain_idx
        )
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            generator=dataloader_rng,
            drop_last=True,
        )

        if config.init_noise:
            for parameter in name_to_param.values():
                parameter.data.add_(
                    torch.randn_like(parameter.data) * config.init_noise
                )

        feed = _make_feed(loader, config.epoch_mode, num_batches)
        accumulated_loss = 0.0
        with torch.no_grad():
            for i in range(num_batches):
                batch = next(feed)
                input_ids = batch["input_ids"].to(device)
                loss = fn(model_on_device, input_ids)
                if chain_idx == 0 and i == 0:
                    expected = input_ids.shape[:-1] + (input_ids.shape[-1] - 1,)
                    if tuple(loss.shape) != tuple(expected):
                        raise ValueError(
                            f"loss_fn must return per-token loss of shape "
                            f"{tuple(expected)}, got {tuple(loss.shape)}"
                        )
                accumulated_loss += loss.detach().float() / num_batches

        writer.write("init_loss", accumulated_loss.mean(dim=0), chain=chain_idx)

        if config.init_noise:
            for name, parameter in name_to_param.items():
                parameter.data.copy_(orig_data[name])

    # Restore model to its original device and training mode
    if is_transformer_lens_model(model):
        model.to(orig_device, print_details=False)  # pyright: ignore[reportCallIssue]
    else:
        model.to(orig_device)
    model.train(was_training)
