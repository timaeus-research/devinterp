"""Quickstart: LLC, susceptibilities, and BIF on Qwen2.5-0.5B.

Requires: pip install devinterp transformers datasets
GPU recommended.
"""

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from devinterp.slt.llc import llc
from devinterp.slt.susceptibilities import susceptibilities
from devinterp.slt.weight_restrictions import (
    create_param_masks,
    preview_weight_restriction,
)
from devinterp.utils import default_nbeta, tokenize_and_concatenate

MODEL = "Qwen/Qwen2.5-0.5B"
BATCH_SIZE = 4

# --- Load model and data ---

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

raw = load_dataset("NeelNanda/pile-10k", split="train")
ds = tokenize_and_concatenate(
    raw.select(range(200)),
    tokenizer,
    column_name="text",
    add_bos_token=False,
    max_length=256,
)
# A second dataset for probing susceptibilities
probe = tokenize_and_concatenate(
    raw.select(range(200, 400)),
    tokenizer,
    column_name="text",
    add_bos_token=False,
    max_length=256,
)

n_beta = default_nbeta(BATCH_SIZE)

# --- LLC ---

result = llc(
    model=model,
    dataset=ds,
    observables={"train": ds},
    lr=1e-4,
    n_beta=n_beta,
    num_chains=2,
    num_draws=50,
    batch_size=BATCH_SIZE,
    num_init_loss_batches=4,
)
print(f"LLC: {result['llc_mean']:.2f} +/- {result['llc_std']:.2f}")

# --- Weight restrictions ---
# Preview which params a restriction selects before running susceptibilities.
# "l0h0" = layer 0, head 0. For Qwen2.5 (14 Q heads, 2 KV heads via GQA),
# Q and O are per-head, so selecting a single head hits ~7% (1/14). K and V
# are per-KV-head, and each KV head is shared by 7 Q heads, so selecting any
# of those 7 picks up the full shared KV head -> 50% (1/2).
l0h0_mask = create_param_masks(model, "l0h0")
print("\n--- l0h0 (single attention head) ---")
preview_weight_restriction(model, l0h0_mask)

# --- Susceptibilities with weight restrictions ---

result = susceptibilities(
    model=model,
    dataset=ds,
    observables={"train": (ds, 2), "probe": (probe, 2)},
    weight_restrictions={
        "full": None,
        "l0h0": l0h0_mask,
        "l0h1": create_param_masks(model, "l0h1"),
    },
    sampling_task="train",
    lr=1e-4,
    n_beta=n_beta,
    num_chains=2,
    num_draws=50,
    batch_size=BATCH_SIZE,
    num_init_loss_batches=4,
)
sus = result["susceptibilities"].dataset
print(f"Susceptibilities shape: {dict(sus.dims)}")
print(sus["sus"])

# --- Manual weight restrictions ---
# For unsupported architectures, build masks directly.
# A mask is just {param_name: bool_tensor | None} where None means unrestricted.
# Example: restrict to the first MLP layer's gate projection

manual_masks = {}
for name, param in model.named_parameters():
    if "model.layers.0.mlp.gate_proj" in name:
        manual_masks[name] = None  # optimize this entire param
    elif "model.layers.0.mlp.up_proj" in name:
        # partially mask: only first half of neurons
        mask = torch.zeros_like(param, dtype=torch.bool)
        mask[: param.shape[0] // 2] = True
        manual_masks[name] = mask

print("\nManual mask:")
preview_weight_restriction(model, manual_masks)
