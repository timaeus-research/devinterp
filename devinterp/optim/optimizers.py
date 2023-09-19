import logging
from typing import Iterable, Literal, Optional, Tuple, Union

import torch
from pydantic import BaseModel, model_validator

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT


class OptimizerConfig(BaseModel):
    optimizer_type: Literal["SGD", "Adam", "AdamW", "SGLD", 'SGNHT'] = "SGD"
    lr: float = 0.01
    weight_decay: Optional[float] = 0.0001
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None
    noise_level: Optional[float] = None
    elasticity: Optional[float] = None
    diffusion_factor: Optional[float] = None
    bounding_box_size: Optional[float] = None
    batch_size: Optional[int] = None
    temperature: Optional[Union[Literal["adaptive"], float]] = None
    num_samples: Optional[int] = None  # If -1, then this needs to be filled in later.

    class Config:
        frozen = True

    def model_dump(self, *args, **kwargs):
        # Only export relevant fields based on optimizer_type
        fields = {"optimizer_type", "lr"}
        if self.optimizer_type == "SGD":
            fields.update("momentum",  "weight_decay")
        elif self.optimizer_type in {"Adam", "AdamW"}:
            fields.update("betas",  "weight_decay")
        elif self.optimizer_type == "SGLD":
            fields.update({"noise_level", "elasticity", "temperature", "bounding_box_size", "num_samples",  "weight_decay"})
        elif self.optimizer_type == "SGNHT":
            fields.update({"diffusion_factor", "bounding_box_size", "num_samples", "batch_size"})
        return super().model_dump(include=fields, *args, **kwargs)

    @model_validator(mode="after")
    def validate_optimizer_type(self) -> "OptimizerConfig":
        if self.optimizer_type == "SGLD":
            assert (
                self.noise_level is not None
            ), "noise_level must be specified for SGLD"
            assert self.elasticity is not None, "elasticity must be specified for SGLD"
            assert (
                self.temperature is not None
            ), "temperature must be specified for SGLD"
            assert (
                self.num_samples is not None
            ), "num_samples must be specified for SGLD"
        elif self.optimizer_type == "SGD":
            assert self.momentum is not None, "momentum must be specified for SGD"
        elif self.optimizer_type in {"Adam", "AdamW"}:
            assert self.betas is not None, "betas must be specified for Adam/AdamW"
        elif self.optimizer_type == "SGNHT":
            assert self.diffusion_factor is not None, "diffusion_factor must be specified for SGLD"
            # assert self.bounding_box_size is not None, "bounding_box_size must be specified for SGLD"
            assert self.batch_size is not None, "batch_size must be specified for SGLD"
            assert self.num_samples is not None, "num_samples must be specified for SGLD"
        return self

    def factory(self, parameters: Iterable[torch.nn.Parameter]):
        optimizer_type = self.optimizer_type
        optimizer_params = self.model_dump(exclude={"optimizer_type"})

        if optimizer_type == "SGD":
            return torch.optim.SGD(parameters, **optimizer_params)
        elif optimizer_type == "Adam":
            return torch.optim.Adam(parameters, **optimizer_params)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(parameters, **optimizer_params)
        elif optimizer_type == "SGLD":
            return SGLD(parameters, **optimizer_params)
        elif self.optimizer_type == "SGNHT":
            return SGNHT(parameters, **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")




