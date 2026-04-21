"""Integration tests for caching in bif() and susceptibilities()."""

import pytest
import zarr

from devinterp.slt.bif import bif
from devinterp.slt.susceptibilities import susceptibilities
from devinterp.slt.weight_restrictions import create_param_masks

from .test_standalone import _load_dataset, _load_model_and_tokenizer


@pytest.fixture(scope="module")
def model_and_data():
    model, tokenizer = _load_model_and_tokenizer()
    train = _load_dataset("timaeus/dsir-pile-10k", tokenizer, column="contents")
    code = _load_dataset("timaeus/pile-github", tokenizer, column="text")
    return model, train, code


def _sample_kwargs(model, train, code, output_path, lr=0.001):
    return dict(
        model=model,
        dataset=train,
        observables={"train": (train, 2), "code": (code, 2)},
        lr=lr,
        n_beta=30,
        num_chains=1,
        num_draws=2,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
        output_path=output_path,
    )


@pytest.mark.gpu
def test_bif_caching(model_and_data, tmp_path):
    model, train, code = model_and_data
    sample_path = tmp_path / "samples.zarr"

    bif(**_sample_kwargs(model, train, code, sample_path, lr=0.001))
    assert zarr.open_group(str(sample_path)).attrs.get("completed") == 1

    bif(**_sample_kwargs(model, train, code, sample_path, lr=0.001))

    with pytest.raises(RuntimeError, match="different sampler config"):
        bif(**_sample_kwargs(model, train, code, sample_path, lr=0.005))


@pytest.mark.gpu
def test_susceptibilities_per_wr_cache(model_and_data, tmp_path):
    model, train, code = model_and_data
    base_path = tmp_path / "sus.zarr"

    susceptibilities(
        model=model,
        dataset=train,
        observables={"train": (train, 2), "code": (code, 2)},
        weight_restrictions={
            "full": None,
            "l0h0": create_param_masks(model, "l0h0"),
        },
        sampling_task="train",
        lr=0.001,
        n_beta=30,
        num_chains=1,
        num_draws=2,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
        output_path=base_path,
    )

    for wr in ["full", "l0h0"]:
        wr_path = base_path.parent / f"{base_path.stem}_{wr}{base_path.suffix}"
        assert zarr.open_group(str(wr_path)).attrs.get("completed") == 1
