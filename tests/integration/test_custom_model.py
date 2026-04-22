"""Verify sample/bif/llc/susceptibilities work with a custom torch module
(no HuggingFace wrapper, no TransformerLens). Exercises the fallback path
in lm_forward_logits (model returns logits directly)."""

import numpy as np
import pytest
import torch
import torch.nn as nn
from datasets import Dataset

from devinterp.slt.bif import bif
from devinterp.slt.llc import llc
from devinterp.slt.sampling import sample
from devinterp.slt.susceptibilities import susceptibilities


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=16, n_heads=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x, _ = self.attn(x, x, x)
        return self.ff(x)


@pytest.fixture
def model_and_data():
    vocab, seq, n = 50, 16, 16
    model = TinyTransformer(vocab)
    ids = np.random.RandomState(0).randint(0, vocab, (n, seq))
    ds = Dataset.from_dict({"input_ids": ids.tolist()})
    ds.set_format(type="torch", columns=["input_ids"])
    return model, ds


_SAMPLER_KW = dict(
    lr=1e-4,
    n_beta=1.0,
    num_chains=1,
    num_draws=2,
    batch_size=2,
    num_init_loss_batches=1,
    init_seed=42,
)


def _head_0_mask(model, d_model=16, n_heads=2):
    """Mirrors 'l0h0' from the README: restrict updates to attention head 0.

    Params not in the returned dict (embed, ff, out_proj.bias) get
    requires_grad_(False) and don't train at all.
    """
    head_dim = d_model // n_heads
    mask: dict[str, torch.Tensor] = {}
    # in_proj_weight: concatenated [Q; K; V] of shape (3*d_model, d_model).
    # Head 0 lives in rows [0:head_dim] of each Q/K/V block.
    w = torch.zeros_like(model.attn.in_proj_weight)
    for offset in (0, d_model, 2 * d_model):
        w[offset : offset + head_dim] = 1
    mask["attn.in_proj_weight"] = w
    b = torch.zeros_like(model.attn.in_proj_bias)
    for offset in (0, d_model, 2 * d_model):
        b[offset : offset + head_dim] = 1
    mask["attn.in_proj_bias"] = b
    # out_proj.weight: shape (d_model, d_model). Head 0 = first head_dim columns.
    w = torch.zeros_like(model.attn.out_proj.weight)
    w[:, 0:head_dim] = 1
    mask["attn.out_proj.weight"] = w
    return mask


def test_sample(model_and_data):
    model, ds = model_and_data
    result = sample(
        model=model, dataset=ds, observables={"self": (ds, 2)}, **_SAMPLER_KW
    )
    assert "sampling_loss" in result.dataset


def test_bif(model_and_data):
    model, ds = model_and_data
    result = bif(
        model=model,
        dataset=ds,
        observables={"a": (ds, 2), "b": (ds, 2)},
        **_SAMPLER_KW,
    )
    assert "influences" in result


def test_llc(model_and_data):
    model, ds = model_and_data
    result = llc(model=model, dataset=ds, observables={"self": (ds, 2)}, **_SAMPLER_KW)
    assert "llc_mean" in result


def test_llc_scaling_invariance(model_and_data):
    """LLC(loss_fn=2*L, n_beta=N/2) must equal LLC(loss_fn=L, n_beta=N).

    Scaling the loss by 2 doubles the gradient; halving n_beta cancels
    (SGLD update uses `grad.mul(nbeta)`), so sampling trajectories are
    bit-identical. The LLC arithmetic n_beta*(mean_loss - init_loss)
    also cancels. Strong end-to-end test that loss_fn is actually
    threaded through sampling, init_loss, and observables.
    """
    model, ds = model_and_data
    base = llc(model=model, dataset=ds, observables={"self": (ds, 2)}, **_SAMPLER_KW)

    def doubled_ce(m: nn.Module, ids: torch.Tensor) -> torch.Tensor:
        logits = m(ids)
        shift_logits = logits[..., :-1, :]
        shift_ids = ids[..., 1:].unsqueeze(-1)
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        return -2.0 * log_probs.gather(dim=-1, index=shift_ids).squeeze(-1)

    scaled_kw = {**_SAMPLER_KW, "n_beta": _SAMPLER_KW["n_beta"] * 0.5}
    scaled = llc(
        model=model,
        dataset=ds,
        observables={"self": (ds, 2)},
        loss_fn=doubled_ce,
        **scaled_kw,
    )
    assert float(scaled["llc_mean"]) == float(base["llc_mean"])


def test_susceptibilities(model_and_data):
    model, ds = model_and_data
    result = susceptibilities(
        model=model,
        dataset=ds,
        observables={"train": (ds, 2), "probe": (ds, 2)},
        weight_restrictions={"full": None, "head0": _head_0_mask(model)},
        sampling_task="train",
        **_SAMPLER_KW,
    )
    assert "susceptibilities" in result.children


@pytest.mark.parametrize("field", ["rmsprop_eps", "rmsprop_alpha"])
def test_rmsprop_arg_rejected_for_non_rmsprop_sampler(model_and_data, field):
    model, ds = model_and_data
    with pytest.raises(ValueError, match=r"rmsprop_(eps|alpha) can only be set"):
        sample(
            model=model,
            dataset=ds,
            observables={"self": (ds, 2)},
            sampling_method="sgmcmc_sgld",
            **{field: 0.5},
            **_SAMPLER_KW,
        )


@pytest.mark.parametrize(
    "field,kwarg_name", [("rmsprop_eps", "eps"), ("rmsprop_alpha", "alpha")]
)
def test_rmsprop_arg_rejects_duplicate(model_and_data, field, kwarg_name):
    model, ds = model_and_data
    with pytest.raises(ValueError, match=r"specify only one"):
        sample(
            model=model,
            dataset=ds,
            observables={"self": (ds, 2)},
            sampling_method="rmsprop_sgld",
            sampling_method_kwargs={kwarg_name: 0.3},
            **{field: 0.3},
            **_SAMPLER_KW,
        )


def test_rmsprop_eps_alpha_equivalent_to_kwargs(model_and_data):
    """Setting rmsprop_eps/rmsprop_alpha must produce identical samples to
    passing the same values via sampling_method_kwargs."""
    model, ds = model_and_data
    via_top = sample(
        model=model,
        dataset=ds,
        observables={"self": (ds, 2)},
        sampling_method="rmsprop_sgld",
        rmsprop_eps=0.2,
        rmsprop_alpha=0.95,
        **_SAMPLER_KW,
    )
    via_kwargs = sample(
        model=model,
        dataset=ds,
        observables={"self": (ds, 2)},
        sampling_method="rmsprop_sgld",
        sampling_method_kwargs={"eps": 0.2, "alpha": 0.95},
        **_SAMPLER_KW,
    )
    assert torch.equal(
        torch.as_tensor(via_top.dataset["sampling_loss_micro"].values),
        torch.as_tensor(via_kwargs.dataset["sampling_loss_micro"].values),
    )
