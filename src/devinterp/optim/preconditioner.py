from abc import ABC, abstractmethod
from functools import reduce
from typing import List, NamedTuple, Optional, Union

import torch


class PreconditionerCoefs(NamedTuple):
    """Coefficients returned by preconditioners"""

    grad_coef: Union[float, torch.Tensor]  # Coefficient for gradient terms
    prior_coef: Union[float, torch.Tensor]  # Coefficient for prior terms
    noise_coef: Union[float, torch.Tensor]  # Coefficient for noise terms
    overall_coef: Union[float, torch.Tensor]  # Overall scaling coefficient
    grad_correction: Optional[Union[float, torch.Tensor]] = (
        None  # Optional momentum-based correction, note that this is additive when multiplying PreconditionerCoefs
    )

    def combine_with(self, other: "PreconditionerCoefs") -> "PreconditionerCoefs":
        grad_correction = (
            self.grad_correction
            if self.grad_correction is not None
            else other.grad_correction
        )

        if self.grad_correction is not None and other.grad_correction is not None:
            grad_correction = self.grad_correction + other.grad_correction

        return PreconditionerCoefs(
            grad_coef=self.grad_coef * other.grad_coef,
            prior_coef=self.prior_coef * other.prior_coef,
            noise_coef=self.noise_coef * other.noise_coef,
            overall_coef=self.overall_coef * other.overall_coef,
            grad_correction=grad_correction,
        )

    @classmethod
    def combine(cls, *coefs: "PreconditionerCoefs") -> "PreconditionerCoefs":
        return reduce(lambda a, b: a.combine_with(b), coefs)


class Preconditioner(ABC):
    """Base class for preconditioners that generate coefficients for MCMC terms"""

    @abstractmethod
    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        """
        Compute coefficients for gradient, prior, and noise terms
        Returns PreconditionerCoefs containing all coefficients
        Each coefficient can be a scalar or tensor of shape matching param
        """
        pass


class IdentityPreconditioner(Preconditioner):
    """Identity preconditioning (i.e., no preconditioning)"""

    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        return PreconditionerCoefs(1.0, 1.0, 1.0, 1.0, None)

    def __repr__(self) -> str:
        return "IdentityPreconditioner()"


class MaskPreconditioner(Preconditioner):
    """Applies masks to the overall coefficient while keeping other coefficients at 1.0

    Stores one mask per parameter in the parameter group.
    """

    def __init__(self, masks: List[Union[torch.Tensor, float]]):
        """
        Args:
            masks: List of masks, one per parameter in the parameter group.
                  Each mask can be either a tensor matching the parameter shape
                  or a float that will be broadcast.
        """
        self.masks = masks

    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        # Get the parameter index from the state dict
        if "param_idx" not in state:
            raise ValueError("param_idx not found in state dict")

        param_idx = state["param_idx"]
        mask = self.masks[param_idx]

        return PreconditionerCoefs(
            grad_coef=1.0,
            prior_coef=1.0,
            noise_coef=1.0,
            overall_coef=mask,
            grad_correction=None,
        )

    def __repr__(self) -> str:
        return f"MaskPreconditioner(masks={self.masks})"


class CompositePreconditioner(Preconditioner):
    """Combines multiple preconditioners by multiplying their coefficients"""

    def __new__(cls, preconditioners: List[Preconditioner]):
        # Filter out identity preconditioners
        non_identity_preconditioners = [
            p for p in preconditioners if not isinstance(p, IdentityPreconditioner)
        ]

        # If no preconditioners left, return identity
        if not non_identity_preconditioners:
            return IdentityPreconditioner()

        # If only one preconditioner left, return it directly
        if len(non_identity_preconditioners) == 1:
            instance = non_identity_preconditioners[0]
        else:
            instance = super().__new__(cls)
            instance.preconditioners = non_identity_preconditioners

        return instance

    def __init__(self, preconditioners: List[Preconditioner]):
        self.preconditioners = preconditioners

    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        # Initialize with first preconditioner's coefficients
        coefs = self.preconditioners[0].get_coefficients(param, grad, state)

        # Multiply coefficients from remaining preconditioners
        for precond in self.preconditioners[1:]:
            coefs = PreconditionerCoefs.combine(
                coefs, precond.get_coefficients(param, grad, state)
            )

        return coefs

    def __repr__(self) -> str:
        return f"CompositePreconditioner({', '.join(repr(precond) for precond in self.preconditioners)})"


class RMSpropPreconditioner(Preconditioner):
    """RMSprop-style diagonal preconditioning"""

    def __init__(
        self, alpha: float = 0.99, eps: float = 1e-1, add_grad_correction=False
    ):
        self.alpha = alpha
        self.eps = eps
        self.add_grad_correction = add_grad_correction

        if self.add_grad_correction:
            raise NotImplementedError(
                "Gradient correction not yet implemented for RMSprop"
            )

    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        # Update running average of squared gradient
        if "square_avg" not in state:
            state["square_avg"] = torch.zeros_like(grad)

        state["square_avg"].mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)

        # Compute preconditioner (1/sqrt(v))
        preconditioner = 1.0 / (torch.sqrt(state["square_avg"]) + self.eps)

        return PreconditionerCoefs(
            grad_coef=preconditioner,
            prior_coef=preconditioner,
            noise_coef=torch.sqrt(preconditioner),
            overall_coef=1.0,
            grad_correction=None,
        )

    def __repr__(self) -> str:
        return f"RMSpropPreconditioner(alpha={self.alpha}, eps={self.eps}, add_grad_correction={self.add_grad_correction})"


class NHTPreconditioning(Preconditioner):
    """Nose-Hoover Thermostat preconditioning"""

    def __init__(self, diffusion_factor: float = 0.01, eps: float = 1e-8):
        self.diffusion_factor = diffusion_factor
        self.eps = eps

    def get_coefficients(
        self, param: torch.Tensor, grad: torch.Tensor, state: dict
    ) -> PreconditionerCoefs:
        # Initialize or get thermostat variable
        state["thermostat"] = thermostat = state.get(
            "thermostat", torch.tensor(self.diffusion_factor, device=param.device)
        )

        # Initialize or get momentum
        state["momentum"] = momentum = state.get(
            "momentum", torch.randn_like(param.data)
        )

        grad_correction = -thermostat * momentum

        # Update thermostat based on kinetic energy
        kinetic_energy = torch.sum(momentum * momentum) / momentum.numel()
        state["thermostat"] += kinetic_energy - 1.0

        return PreconditionerCoefs(
            grad_coef=1.0,
            prior_coef=1.0,
            noise_coef=torch.sqrt(torch.tensor(2.0 * self.diffusion_factor)),
            overall_coef=1.0,
            grad_correction=grad_correction,
        )

    def __repr__(self) -> str:
        return f"NHTPreconditioning(diffusion_factor={self.diffusion_factor}, eps={self.eps})"
