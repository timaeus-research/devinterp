import itertools
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.notebook import tqdm

from devinterp.evals import dataloader_map, dataloader_reduce
from devinterp.slt.sgld import SGLD


class Ensemble(nn.Module):
    """
    A class which contains copies of the same model for managing several 
    independent runs when sampling observables over some distribution.

    To be used in conjunction with an optimizer like SGLD.
    """
    def __init__(self, model: nn.Module, num_chains=1):
        super().__init__()
        self.model = model
        self.models = nn.ModuleList([model] + [deepcopy(model) for _ in range(num_chains-1)])
        self.num_chains = num_chains

    @classmethod
    def from_cls(cls, model_cls: Type[nn.Module], *args, num_chains=1, **kwargs):
        model = model_cls(*args, **kwargs)
        return cls(model, num_chains=num_chains)
        
    def forward(self, inputs):
        return tuple(model(inputs) for model in self.models)
        
    def reset(self):
        """Sets all models to the original model's parameters"""
        original_params = self.model.state_dict()

        for model in self.models:
            model.load_state_dict(original_params)

    # TODO: Think about smarter recombining strategies (i.e., put the compute to use)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)
    
