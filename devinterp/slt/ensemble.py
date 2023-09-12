from copy import deepcopy
from typing import Type

from torch import nn


class Ensemble(nn.Module):
    """
    A container class for managing multiple independent instances (chains)
    of a neural network model. Primarily used for sampling observables
    over a distribution using stochastic optimizers like SGLD.

    Attributes:
        model (nn.Module): The original model.
        models (nn.ModuleList): List containing independent copies of the original model.
        num_chains (int): Number of independent model instances.
    """

    def __init__(self, model: nn.Module, num_chains=1):
        super().__init__()
        self.model = model
        self.models = nn.ModuleList(
            [deepcopy(model) for _ in range(num_chains )]
        )
        self.num_chains = num_chains

    @classmethod
    def from_cls(cls, model_cls: Type[nn.Module], *args, num_chains=1, **kwargs):
        """Alternative constructor using a model class and initialization arguments."""
        model = model_cls(*args, **kwargs)
        return cls(model, num_chains=num_chains)

    def forward(self, inputs):
        """Runs forward pass on all models and returns their outputs as a tuple."""
        return tuple(model(inputs) for model in self.models)

    def reset(self):
        """Resets all model instances to the initial parameters of the original model."""
        original_params = self.model.state_dict()

        for model in self.models:
            model.load_state_dict(original_params)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)
