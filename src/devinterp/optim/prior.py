from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from numbers import Real
from typing import Any, Iterable, Literal, Optional, Union

import numpy as np
import torch


class Prior(ABC):
    """Abstract base class for parameter priors"""

    key: str

    @abstractmethod
    def initialize(
        self, params: Sequence[torch.Tensor]
    ) -> dict[torch.Tensor, dict[str, Any]]:
        """Initialize prior for parameters

        Args:
            params: Iterator of model parameters

        Returns:
            Updated state dictionary
        """
        pass

    @abstractmethod
    def grad(
        self,
        param: torch.Tensor,
        state: dict[str, Any],
    ) -> torch.Tensor:
        """Compute gradient of the prior

        Args:
            param: Parameter tensor
            state: State dictionary

        Returns:
            Gradient tensor
        """
        pass


class GaussianPrior(Prior):
    """Gaussian prior with configurable center and precision"""

    def __init__(
        self,
        localization: float,
        center: Optional[
            Union[Literal["initial"], Iterable[torch.Tensor], Real]
        ] = "initial",
    ):
        """
        Args:
            localization: Precision (inverse variance) of the Gaussian
            center: Where to center the Gaussian:
                - None: centered at 0 (standard L2 regularization)
                - 'initial': centered at initial parameter values (localization)
                - iterable of tensors: centered at provided parameter values
                  (must match model parameter shapes)
        """
        self.localization = localization
        self.key = (
            "prior_center"  # Mutable - will be modified if passed to a CompositePrior
        )

        if isinstance(center, (str, type(None))):
            self.center = center
        elif isinstance(center, Real):
            self.center = center
        else:
            # Convert iterable to list to ensure we can reuse it
            self.center = list(center)

    def initialize(
        self, params: Sequence[torch.Tensor]
    ) -> dict[torch.Tensor, dict[str, Any]]:
        """Initialize centers for all parameters

        Args:
            params: Iterator of model parameters

        Returns:
            State dictionary containing prior centers
        """
        state = defaultdict(dict)

        if isinstance(self.center, list):
            # Validate and use provided centers
            if len(self.center) != len(params):
                raise ValueError(
                    f"Number of centers ({len(self.center)}) does not match "
                    f"number of parameters ({len(params)})"
                )
            for c, p in zip(self.center, params):
                if c.shape != p.shape:
                    raise ValueError(
                        f"Center shape {c.shape} does not match "
                        f"parameter shape {p.shape}"
                    )
                state[p][self.key] = c.detach().clone()

        elif self.center == "initial":
            # Use initial parameter values as centers
            for p in params:
                state[p][self.key] = p.detach().clone()

        else:  # None case - zero-centered
            for p in params:
                state[p][self.key] = None

        return state

    def grad(
        self,
        param: torch.Tensor,
        state: dict[str, Any],
    ) -> torch.Tensor:
        """Compute gradient of the prior. If state is provided, the prior center is
        looked up in the state dictionary using the instance key.

        Args:
            param: Parameter tensor
            state: State dictionary

        Returns:
            Gradient tensor
        """
        center = state.get(self.key)

        if center is None:
            return self.localization * param
        else:
            return self.localization * (param - center)

    def __repr__(self) -> str:
        return f"GaussianPrior(localization={self.localization}, center={self.center})"


class UniformPrior(Prior):
    """Uniform prior."""

    def __init__(self, box_size: float = np.inf):
        """
        Args:
            box_size: Size of the box constraint
        """
        # Required by the Prior ABC but unused: initialize() returns {} and
        # grad() returns zeros, so the key is never looked up in any state dict.
        self.key = "uniform_prior_center"
        self.box_size = box_size

        if box_size != np.inf:
            raise NotImplementedError(
                "Uniform prior with finite box size not implemented"
            )

    def initialize(
        self, params: Sequence[torch.Tensor]
    ) -> dict[torch.Tensor, dict[str, Any]]:
        return {}

    def grad(self, param: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        return torch.zeros_like(param)


class CompositePrior(Prior):
    """Combines multiple priors, summing their contributions.

    Always wraps even a single non-uniform prior (no short-circuit).
    Callers that need to decompose sub-priors (e.g. SGMCMC._update_metrics)
    can iterate ``self.priors`` uniformly.
    """

    def __init__(self, priors: list[Prior]):
        self.key = "composite_prior"
        self.priors = [p for p in priors if not isinstance(p, UniformPrior)]
        for i, prior in enumerate(self.priors):
            prior.key = f"{prior.key}_{i}"

    def initialize(
        self, params: Sequence[torch.Tensor]
    ) -> dict[torch.Tensor, dict[str, Any]]:
        combined_state = defaultdict(dict)

        for prior in self.priors:
            prior_state = prior.initialize(params)
            for param, state in prior_state.items():
                for key, value in state.items():
                    combined_state[param][key] = value

        return combined_state

    def grad(self, param: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
        result = torch.zeros_like(param)
        for prior in self.priors:
            result += prior.grad(param, state)
        return result

    def __repr__(self) -> str:
        return f"CompositePrior({self.priors})"
