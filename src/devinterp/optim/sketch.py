from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch

SKETCH_QUANTITIES: tuple[str, ...] = (
    "scaled_grad",
    "unscaled_grad",
    "localization",
    "weight_decay",
    "noise",
)


@dataclass
class CountSketch:
    """Count sketch projection for dimensionality reduction.

    Projects d-dimensional vectors to k-dimensional sketches while preserving
    inner products in expectation: E[<Sv, Sw>] = <v, w>.

    The sketch is defined by a hash function h: [d] -> [k] (mapping each input
    coordinate to an output bucket) and a sign function s: [d] -> {-1, +1}
    (random per coordinate). The sketch of vector v is:

        S(v)[j] = sum_{i : h(i) = j} s(i) * v[i]

    Both h and s are generated deterministically from a seed. Two sketch vectors
    are only comparable when produced by the same CountSketch instance (same
    seed, same input_dim). When used with an optimizer, input_dim is the total
    trainable parameter count, so different weight restrictions yield
    incomparable sketches even with the same seed.

    This single-row construction is equivalent to what Weinberger et al. call
    "feature hashing". The inner product preservation property is proved there.

    References:
      - Weinberger, Dasgupta, Langford, Smola & Attenberg, "Feature Hashing
        for Large Scale Multitask Learning" (ICML 2009),
        https://doi.org/10.1145/1553374.1553516
      - Charikar, Chen & Farach-Colton, "Finding Frequent Items in Data
        Streams" (ICALP 2002), https://doi.org/10.1007/3-540-45465-9_59
      - Larsen, Pagh & Tetek, "CountSketches, Feature Hashing and the Median
        of Three" (ICML 2021), https://doi.org/10.48550/arXiv.2102.02193
    """

    hash_indices: torch.Tensor
    hash_signs: torch.Tensor
    _output_dim: int

    @property
    def device(self) -> torch.device:
        if self.hash_indices.device != self.hash_signs.device:
            raise RuntimeError(
                f"Inconsistent devices: hash_indices on {self.hash_indices.device}, "
                f"hash_signs on {self.hash_signs.device}"
            )
        return self.hash_indices.device

    @property
    def input_dim(self) -> int:
        return self.hash_indices.shape[0]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @classmethod
    def create(cls, input_dim: int, output_dim: int, seed: int = 0) -> CountSketch:
        gen = torch.Generator().manual_seed(seed)
        hash_indices = torch.randint(0, output_dim, (input_dim,), generator=gen)
        hash_signs = (2 * torch.randint(0, 2, (input_dim,), generator=gen) - 1).float()
        return cls(
            hash_indices=hash_indices,
            hash_signs=hash_signs,
            _output_dim=output_dim,
        )

    def to(self, device: torch.device | str | int) -> CountSketch:
        if self.device == torch.device(device):
            return self
        return CountSketch(
            hash_indices=self.hash_indices.to(device),
            hash_signs=self.hash_signs.to(device),
            _output_dim=self._output_dim,
        )

    def sketch(self, v: torch.Tensor) -> torch.Tensor:
        """Apply the full sketch to a flat vector."""
        result = torch.zeros(self._output_dim, dtype=torch.float32, device=v.device)
        self.scatter_into_(result, v, 0)
        return result

    def scatter_into_(self, result: torch.Tensor, v: torch.Tensor, offset: int) -> None:
        """Accumulate one parameter's contribution into a running sketch buffer.

        Exploits linearity: sketching cat(p1, p2, ...) is equivalent to
        accumulating each pi at its offset into the same buffer.
        """
        n = v.numel()
        v_flat = v.detach().reshape(-1).float()
        idx = self.hash_indices[offset : offset + n]
        signs = self.hash_signs[offset : offset + n]
        result.scatter_add_(0, idx, signs * v_flat)


@dataclass
class SketchBuffer:
    """Per-step accumulation buffers for count sketch metrics.

    One buffer per tracked quantity. Lifecycle:
    zero_() at step start -> accumulate per-param via scatter_into_ -> read.
    """

    QUANTITIES: ClassVar[tuple[str, ...]] = SKETCH_QUANTITIES

    scaled_grad: torch.Tensor
    unscaled_grad: torch.Tensor
    localization: torch.Tensor
    weight_decay: torch.Tensor
    noise: torch.Tensor

    @classmethod
    def create(
        cls, output_dim: int, device: torch.device | str = "cpu"
    ) -> SketchBuffer:
        def _z() -> torch.Tensor:
            return torch.zeros(output_dim, dtype=torch.float32, device=device)

        return cls(
            scaled_grad=_z(),
            unscaled_grad=_z(),
            localization=_z(),
            weight_decay=_z(),
            noise=_z(),
        )

    @property
    def device(self) -> torch.device:
        devices = {q: getattr(self, q).device for q in self.QUANTITIES}
        unique = set(devices.values())
        if len(unique) != 1:
            raise RuntimeError(f"Inconsistent devices across buffers: {devices}")
        return next(iter(unique))

    def zero_(self) -> None:
        for q in self.QUANTITIES:
            getattr(self, q).zero_()

    def to(self, device: torch.device | str | int) -> SketchBuffer:
        if self.device == torch.device(device):
            return self
        return SketchBuffer(**{q: getattr(self, q).to(device) for q in self.QUANTITIES})
