"""Observable: evaluates probe datasets during SGLD sampling.

Each observable wraps a dataset and computes per-token losses at each draw.
Input IDs are fixed (same sequences every draw via DeterministicShuffledSampler).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
import torch
from datasets import Dataset
from torch.utils.data import DataLoader, Sampler

from devinterp.slt.lm_loss import LossFn, compute_per_token_loss


def to_obs_id(task_name: str) -> str:
    """Convert a task name to a valid identifier for use in variable names."""
    result = "".join(c if c.isalnum() or c == "_" else "_" for c in task_name)
    if result and result[0].isdigit():
        result = "_" + result
    return result or "_"


class DeterministicShuffledSampler(Sampler):
    """A sampler that returns a fixed shuffled order of indices, deterministic from seed."""

    def __init__(self, data_source: Dataset, num_samples: int, seed: int = 42):
        self.data_source = data_source
        self.num_samples = num_samples
        assert len(data_source) >= num_samples, (
            f"Dataset size ({len(data_source)}) must be >= num_samples ({num_samples})"
        )
        self.indices = random.Random(seed).sample(range(len(data_source)), num_samples)

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return self.num_samples


class Observable:
    """Evaluates a probe dataset at each SGLD draw.

    On construction, loads fixed input_ids (same sequences every draw).
    At each draw, compute_loss(model) returns per-token losses.

    Attributes:
        obs_id: Identifier derived from task_name (e.g. "pile_github").
        input_ids: Fixed input_ids tensor, shape (n_samples, ctx_len+1).
        n_samples: batch_size * batches_per_draw.
        context_length: Number of predicted positions (ctx_len).
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        task_name: str,
        batches_per_draw: int,
        batch_size: int,
        context_length: int,
        device: torch.device,
        seed: int = 1337,
        loss_fn: LossFn | None = None,
    ):
        self.obs_id = to_obs_id(task_name)
        self.batch_size = batch_size
        self.batches_per_draw = batches_per_draw
        self.n_samples = batch_size * batches_per_draw
        self.context_length = context_length
        self.device = device
        self.loss_fn = loss_fn if loss_fn is not None else compute_per_token_loss

        sampler = DeterministicShuffledSampler(dataset, self.n_samples, seed=seed)
        loader = DataLoader(
            dataset, batch_size=batch_size, sampler=sampler, drop_last=True
        )

        # Load fixed input_ids
        self.input_ids = torch.zeros(
            self.n_samples, context_length + 1, dtype=torch.int64, device=device
        )
        for i, batch in enumerate(loader):
            s, e = i * batch_size, (i + 1) * batch_size
            self.input_ids[s:e] = batch["input_ids"]

    def compute_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Compute per-token loss for the model on this observable's data.

        Returns shape (n_samples, context_length).
        """
        assert not torch.is_grad_enabled()
        loss = torch.zeros(
            self.n_samples, self.context_length, dtype=torch.float32, device=self.device
        )
        for i in range(self.batches_per_draw):
            s, e = i * self.batch_size, (i + 1) * self.batch_size
            loss[s:e] = self.loss_fn(model, self.input_ids[s:e])
        return loss
