from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

import torch


def _zeros() -> torch.Tensor:
    return torch.zeros(1, dtype=torch.float32)


@dataclass
class Metrics:
    """Norms and dot products of SGMCMC parameter update components.

    Each step, w += dw where dw = -(scaled_grad + prior) + noise.

    Norm fields store L2 norms of the post-preconditioned update components
    (i.e. actual magnitudes applied to parameters, not raw gradients).

    scaled_grad:    (ε/2) · nβ · G · ∇L     — preconditioned gradient
    unscaled_grad:  (ε/2) · nβ · ∇L         — raw gradient (no preconditioner)
    localization:   (ε/2) · G · γ(w - w₀)   — pull toward initial params
    weight_decay:   (ε/2) · G · λw           — L2 regularization
    noise:          √ε · √G · η             — stochastic exploration
    distance:       w - w₀                   — raw displacement from init

    Dot product fields store inner products between the three main component
    vectors (scaled_grad, combined prior, noise).

    dot_grad_prior:  ⟨scaled_grad, localization + weight_decay⟩
    dot_grad_noise:  ⟨scaled_grad, noise⟩
    dot_prior_noise: ⟨localization + weight_decay, noise⟩

    Cosine similarities can be derived: cos = dot / (norm_a * norm_b).

    Lifecycle (one Metrics per optimizer param group, on the group's device):

    1. __init__:       group["metrics"] = Metrics().to(device)
    2. step(), start:  group["metrics"].zero_()
    3. step(), per-p:  group["metrics"].add_sum_squared_(...)
                       group["metrics"].add_dot_products_(...)
    4. step(), end:    group["metrics"].sqrt_norms_()
    5. get_metrics():  combine per-group metrics on CPU
    """

    # TODO: Could these be annotations in the `field` declarations?
    NORM_FIELDS: ClassVar[tuple[str, ...]] = (
        "scaled_grad",
        "unscaled_grad",
        "localization",
        "weight_decay",
        "noise",
        "distance",
    )

    DOT_FIELDS: ClassVar[tuple[str, ...]] = (
        "dot_grad_prior",
        "dot_grad_noise",
        "dot_prior_noise",
    )

    scaled_grad: torch.Tensor = field(default_factory=_zeros)
    unscaled_grad: torch.Tensor = field(default_factory=_zeros)
    localization: torch.Tensor = field(default_factory=_zeros)
    weight_decay: torch.Tensor = field(default_factory=_zeros)
    noise: torch.Tensor = field(default_factory=_zeros)
    distance: torch.Tensor = field(default_factory=_zeros)

    dot_grad_prior: torch.Tensor = field(default_factory=_zeros)
    dot_grad_noise: torch.Tensor = field(default_factory=_zeros)
    dot_prior_noise: torch.Tensor = field(default_factory=_zeros)

    numel: int = 0

    @property
    def prior(self) -> torch.Tensor:
        """Combined prior norm: ||[localization; weight_decay]||₂."""
        return (self.localization.square() + self.weight_decay.square()).sqrt()

    def to(self, device: str | torch.device | int) -> "Metrics":
        """Return a copy of these metrics on the specified device."""
        return Metrics(
            scaled_grad=self.scaled_grad.to(device),
            unscaled_grad=self.unscaled_grad.to(device),
            localization=self.localization.to(device),
            weight_decay=self.weight_decay.to(device),
            noise=self.noise.to(device),
            distance=self.distance.to(device),
            dot_grad_prior=self.dot_grad_prior.to(device),
            dot_grad_noise=self.dot_grad_noise.to(device),
            dot_prior_noise=self.dot_prior_noise.to(device),
            numel=self.numel,
        )

    def zero_(self) -> None:
        """Reset all metrics to zero in-place."""
        self.scaled_grad.zero_()
        self.unscaled_grad.zero_()
        self.localization.zero_()
        self.weight_decay.zero_()
        self.noise.zero_()
        self.distance.zero_()
        self.dot_grad_prior.zero_()
        self.dot_grad_noise.zero_()
        self.dot_prior_noise.zero_()
        self.numel = 0

    def add_sum_squared_(
        self,
        scaled_grad: torch.Tensor,
        unscaled_grad: torch.Tensor,
        localization: torch.Tensor,
        weight_decay: torch.Tensor,
        noise: torch.Tensor,
        distance: torch.Tensor,
    ) -> None:
        """Accumulate sum-of-squares for each norm component in-place.

        Casts to float32 before squaring to avoid precision loss with bf16/fp16
        inputs (where squaring can overflow or underflow in the input dtype).
        """
        self.scaled_grad += scaled_grad.float().square().sum()
        self.unscaled_grad += unscaled_grad.float().square().sum()
        self.localization += localization.float().square().sum()
        self.weight_decay += weight_decay.float().square().sum()
        self.noise += noise.float().square().sum()
        self.distance += distance.float().square().sum()

    def add_dot_products_(
        self,
        scaled_grad: torch.Tensor,
        prior: torch.Tensor,
        noise: torch.Tensor,
    ) -> None:
        """Accumulate dot products between the three main component vectors.

        Args:
            scaled_grad: The preconditioned gradient vector.
            prior: Combined prior vector (localization + weight_decay).
            noise: The noise vector.
        """
        sg = scaled_grad.float()
        p = prior.float()
        n = noise.float()
        self.dot_grad_prior += (sg * p).sum()
        self.dot_grad_noise += (sg * n).sum()
        self.dot_prior_noise += (p * n).sum()

    def sqrt_norms_(self) -> None:
        """Convert norm fields from sum-of-squares to L2 norms in-place."""
        self.scaled_grad = self.scaled_grad.sqrt()
        self.unscaled_grad = self.unscaled_grad.sqrt()
        self.localization = self.localization.sqrt()
        self.weight_decay = self.weight_decay.sqrt()
        self.noise = self.noise.sqrt()
        self.distance = self.distance.sqrt()

    @staticmethod
    def aggregate(group_metrics: Iterable["Metrics"]) -> "Metrics":
        """Combine per-group metrics into a single Metrics on CPU.

        Norms: re-square to get sum-of-squares, accumulate, then sqrt:
        ||[a; b]|| = sqrt(||a||^2 + ||b||^2).

        Dot products: additive across disjoint parameter sets.
        """
        result = Metrics()
        for m in group_metrics:
            m_cpu = m.to("cpu")
            result.scaled_grad += m_cpu.scaled_grad.square()
            result.unscaled_grad += m_cpu.unscaled_grad.square()
            result.localization += m_cpu.localization.square()
            result.weight_decay += m_cpu.weight_decay.square()
            result.noise += m_cpu.noise.square()
            result.distance += m_cpu.distance.square()
            result.dot_grad_prior += m_cpu.dot_grad_prior
            result.dot_grad_noise += m_cpu.dot_grad_noise
            result.dot_prior_noise += m_cpu.dot_prior_noise
            result.numel += m_cpu.numel
        result.sqrt_norms_()
        return result
