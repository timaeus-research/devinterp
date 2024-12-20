from abc import ABC, abstractmethod
from collections import defaultdict
from numbers import Real
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from torch.optim import Optimizer


class Prior(ABC):
    """Abstract base class for parameter priors"""

    @abstractmethod
    def initialize(
        self, params: Iterator[torch.Tensor]
    ) -> Dict[torch.Tensor, Dict[str, Any]]:
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
        state: Dict[str, Any],
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
        key: str = "prior_center",
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
        self.key = key
        if isinstance(center, (str, type(None))):
            self.center = center
        elif isinstance(center, Real):
            self.center = center
        else:
            # Convert iterable to list to ensure we can reuse it
            self.center = list(center)

    def initialize(
        self, params: Iterator[torch.Tensor]
    ) -> Dict[torch.Tensor, Dict[str, Any]]:
        """Initialize centers for all parameters

        Args:
            params: Iterator of model parameters

        Returns:
            State dictionary containing prior centers
        """
        params_list = list(params)
        state = defaultdict(dict)

        if isinstance(self.center, list):
            # Validate and use provided centers
            if len(self.center) != len(params_list):
                raise ValueError(
                    f"Number of centers ({len(self.center)}) does not match "
                    f"number of parameters ({len(params_list)})"
                )
            for c, p in zip(self.center, params_list):
                if c.shape != p.shape:
                    raise ValueError(
                        f"Center shape {c.shape} does not match "
                        f"parameter shape {p.shape}"
                    )
                state[p][self.key] = c.detach().clone()

        elif self.center == "initial":
            # Use initial parameter values as centers
            for p in params_list:
                state[p][self.key] = p.detach().clone()

        else:  # None case - zero-centered
            for p in params_list:
                state[p][self.key] = None

        return state

    def grad(
        self,
        param: torch.Tensor,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute gradient of the prior. If state is provided, the prior center is
        looked up in the state dictionary using the instance key.

        Args:
            param: Parameter tensor
            state: State dictionary

        Returns:
            Gradient tensor
        """
        center = state[self.key]

        if center is None:
            return self.localization * param
        else:
            return self.localization * (param - center)

    def distance_sq(
        self,
        param: torch.Tensor,
        state: Dict[str, Any],
        scale: Optional[Union[float, torch.Tensor]] = 1.0,
    ) -> torch.Tensor:
        """Compute squared distance from prior center. If state is provided, the
        prior center is looked up in the state dictionary using the instance key.


        Args:
            param: Parameter tensor
            state: State dictionary
            scale: Scale factor
        """
        center = state[self.key]

        if center is not None:
            return ((scale * (param - center)) ** 2).sum()
        else:
            return ((scale * param) ** 2).sum()

    def __repr__(self) -> str:
        return f"GaussianPrior(localization={self.localization}, center={self.center})"


class UniformPrior(Prior):
    """Uniform prior."""

    def __init__(self, box_size: float = np.inf):
        self.box_size = box_size

        if box_size != np.inf:
            raise NotImplementedError(
                "Uniform prior with finite box size not implemented"
            )

    def initialize(
        self, params: Iterator[torch.Tensor]
    ) -> Dict[torch.Tensor, Dict[str, Any]]:
        return {}

    def grad(self, param: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return torch.zeros_like(param)


class CompositePrior(Prior):
    """Combines multiple priors, summing their contributions.
    The last prior in the list takes precedence for distance_sq and as a default for getattr.
    """

    def __new__(cls, priors: List[Prior]):
        non_uniform_priors = [p for p in priors if not isinstance(p, UniformPrior)]

        if not non_uniform_priors:
            return UniformPrior()

        if len(non_uniform_priors) == 1:
            instance = non_uniform_priors[0]
        else:
            instance = super().__new__(cls)
            instance.priors = non_uniform_priors

        return instance

    def __init__(self, priors: List[Prior]):
        self.priors = priors

        for i, prior in enumerate(priors):
            prior.key = f"{prior.key}_{i}"

    def initialize(
        self, params: Iterator[torch.Tensor]
    ) -> Dict[torch.Tensor, Dict[str, Any]]:
        params = list(params)  # Convert iterator to list for reuse
        combined_state = defaultdict(dict)

        for prior in self.priors:
            prior_state = prior.initialize(params)
            for param, state in prior_state.items():
                for key, value in state.items():
                    combined_state[param][key] = value

        return combined_state

    def grad(self, param: torch.Tensor, state: Dict[str, Any]) -> torch.Tensor:
        return sum(prior.grad(param, state) for prior in self.priors)

    def distance_sq(
        self,
        param: torch.Tensor,
        state: Dict[str, Any],
        scale: Optional[Union[float, torch.Tensor]] = 1.0,
    ) -> torch.Tensor:
        """Compute squared distance from prior center. The last prior in the list
        takes precedence (i.e. is used for distance_sq).
        """
        return self.priors[-1].distance_sq(param, state, scale)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.priors[-1], name)

    def __repr__(self) -> str:
        return f"CompositePrior({self.priors})"
