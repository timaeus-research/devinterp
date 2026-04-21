"""Standalone test: runs the full pipeline with only public dependencies.

No aether imports. Loads model/tokenizer/datasets directly from HuggingFace.
This validates that devinterp can work independently.
"""

import pytest
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from devinterp.utils import tokenize_and_concatenate

from devinterp.slt.sampling import sample


def _load_model_and_tokenizer(name="EleutherAI/pythia-14m"):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(name)
    # Pythia's tokenizer.model_max_length is a sentinel (~1e30) so we fall back
    # to model_config.max_position_embeddings (= 2048).
    tokenizer.model_max_length = model.config.max_position_embeddings
    return model, tokenizer


def _load_dataset(hf_path, tokenizer, column="text", limit=200):
    ds = load_dataset(hf_path, split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    ds = tokenize_and_concatenate(
        ds,
        tokenizer,
        column_name=column,
        add_bos_token=True,
        max_length=tokenizer.model_max_length,
    )
    return ds


@pytest.mark.gpu
def test_standalone_sample():
    """Run sample() with no aether dependencies."""
    model, tokenizer = _load_model_and_tokenizer()

    train_data = _load_dataset("timaeus/dsir-pile-10k", tokenizer, column="contents")
    code_data = _load_dataset("timaeus/pile-github", tokenizer, column="text")

    result = sample(
        model=model,
        dataset=train_data,
        observables={
            "train": (train_data, 2),
            "code": (code_data, 2),
        },
        lr=0.001,
        n_beta=30,
        num_chains=1,
        num_draws=2,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
    )

    ds = result.dataset
    assert "init_loss" in ds
    assert "sampling_loss" in ds
    assert "loss_train" in ds
    assert "loss_code" in ds
    assert "input_ids_train" in ds
    assert "input_ids_code" in ds


@pytest.mark.gpu
def test_standalone_susceptibilities():
    """Run susceptibilities() with no aether dependencies."""
    from devinterp.slt.susceptibilities import susceptibilities
    from devinterp.slt.weight_restrictions import create_param_masks

    model, tokenizer = _load_model_and_tokenizer()

    train_data = _load_dataset("timaeus/dsir-pile-10k", tokenizer, column="contents")
    code_data = _load_dataset("timaeus/pile-github", tokenizer, column="text")

    observables = {
        "train": (train_data, 2),
        "code": (code_data, 2),
    }

    result = susceptibilities(
        model=model,
        dataset=train_data,
        observables=observables,
        weight_restrictions={
            "full": None,
            "l0h1": create_param_masks(model, "l0h1"),
        },
        sampling_task="train",
        lr=0.001,
        n_beta=30,
        num_chains=1,
        num_draws=2,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
    )

    assert "susceptibilities" in result.children
    assert "context" in result.children
    sus = result["susceptibilities"].dataset
    assert "sus" in sus.data_vars
    assert "wr" in sus.coords
    assert list(sus.coords["wr"].values) == ["l0h1"]


@pytest.mark.gpu
def test_standalone_bif():
    """Run compute_bif() with no aether dependencies."""
    from devinterp.slt.bif import compute_bif

    model, tokenizer = _load_model_and_tokenizer()

    train_data = _load_dataset("timaeus/dsir-pile-10k", tokenizer, column="contents")
    code_data = _load_dataset("timaeus/pile-github", tokenizer, column="text")

    samples = sample(
        model=model,
        dataset=train_data,
        observables={
            "train": (train_data, 2),
            "code": (code_data, 2),
        },
        lr=0.001,
        n_beta=30,
        num_chains=2,
        num_draws=4,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
    )

    result = compute_bif(samples, correlation_method="token")
    assert "influences" in result.data_vars


@pytest.mark.gpu
def test_standalone_llc():
    """Run llc() with no aether dependencies."""
    from devinterp.slt.llc import llc

    model, tokenizer = _load_model_and_tokenizer()

    train_data = _load_dataset("timaeus/dsir-pile-10k", tokenizer, column="contents")

    result = llc(
        model=model,
        dataset=train_data,
        observables={"train": (train_data, 2)},
        lr=0.001,
        n_beta=30,
        num_chains=2,
        num_draws=4,
        batch_size=2,
        num_init_loss_batches=1,
        init_seed=42,
    )

    assert "llc_mean" in result
    assert "llc_std" in result
    assert "llc_per_chain" in result
    assert "loss_trace" in result
    assert result["llc_per_chain"].shape == (2,)  # num_chains
    assert result["loss_trace"].shape == (2, 4)  # (num_chains, num_draws)
    # LLC should be positive (loss increases away from optimum)
    assert float(result["llc_mean"]) > 0
