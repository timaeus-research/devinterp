"""Model-agnostic loss computation for language models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from transformers import PreTrainedModel


class NonFiniteLogitsError(ValueError):
    """Raised when logits contain NaNs or Infs."""


def lm_forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and return logits. Handles HF and TransformerLens models."""
    # TransformerLens HookedTransformer
    if hasattr(model, "cfg") and hasattr(model.cfg, "n_heads"):
        return model(input_ids, return_type="logits")
    # Any HuggingFace causal LM: skip KV-cache allocation and output wrapping.
    if isinstance(model, PreTrainedModel):
        return model(input_ids, return_dict=False, use_cache=False)[0]
    # Fallback for plain nn.Modules: return either .logits or a raw tensor.
    output = model(input_ids)
    if hasattr(output, "logits"):
        return output.logits
    return output


def lm_cross_entropy_loss(
    logits: torch.Tensor, input_ids: torch.Tensor
) -> torch.Tensor:
    """Per-token cross entropy loss. Returns shape (batch, seq-1)."""
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    shift_log_probs = log_probs[..., :-1, :]
    shift_input_ids = input_ids[..., 1:, None]
    return -shift_log_probs.gather(dim=-1, index=shift_input_ids)[..., 0]


def compute_per_token_loss(
    model: torch.nn.Module, input_ids: torch.Tensor
) -> torch.Tensor:
    """Compute per-token cross-entropy loss. Returns shape (batch, seq-1)."""
    logits = lm_forward_logits(model, input_ids)
    if not torch.isfinite(logits).all():
        raise NonFiniteLogitsError("Non-finite values detected in model logits.")
    return lm_cross_entropy_loss(logits, input_ids)


EvaluateFn = Callable[
    [torch.nn.Module, dict[str, Any]], tuple[torch.Tensor, dict[str, Any]]
]

LossFn = Callable[[torch.nn.Module, torch.Tensor], torch.Tensor]
"""(model, input_ids) -> per-token loss tensor of shape (batch, seq-1)."""


def make_evaluate_fn(loss_fn: LossFn | None = None) -> EvaluateFn:
    """Create an evaluation function returning unreduced per-token loss.

    Returns (loss, {}) matching the (loss, results) protocol expected by sample_single_chain.
    If loss_fn is None, uses compute_per_token_loss (cross-entropy on the model's logits).
    """
    fn = loss_fn if loss_fn is not None else compute_per_token_loss

    def evaluate(
        model: torch.nn.Module, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        return fn(model, batch["input_ids"]), {}

    return evaluate
