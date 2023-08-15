import logging
import warnings
from typing import List, Literal, Optional, Set, Tuple, Union, Iterable

import torch
import yaml
from pydantic import BaseModel, Field, validator

from devinterp.utils import int_linspace, int_logspace

logger = logging.getLogger(__name__)


class OptimizerConfig(BaseModel):
    optimizer_type: Literal["SGD", "Adam", "AdamW"] = "SGD"
    lr: float = 0.01
    weight_decay: float = 0.0001
    momentum: Optional[float] = None
    betas: Optional[Tuple[float, float]] = None

    class Config:
        validate_assignment = True
        frozen = True

    def model_dump(self, *args, **kwargs):
        # Only export relevant fields based on optimizer_type
        fields = {"optimizer_type", "lr", "weight_decay"}
        if self.optimizer_type == "SGD":
            fields.add("momentum")
        elif self.optimizer_type in {"Adam", "AdamW"}:
            fields.add("betas")

        return super().model_dump(include=fields, *args, **kwargs)

    def factory(self, parameters: Iterable[torch.nn.Parameter]):
        optimizer_type = self.optimizer_type
        optimizer_params = self.model_dump(exclude={"optimizer_type"})

        if optimizer_type == "SGD":
            return torch.optim.SGD(parameters, **optimizer_params)
        elif optimizer_type == "Adam":
            return torch.optim.Adam(parameters, **optimizer_params)
        elif optimizer_type == "AdamW":
            return torch.optim.AdamW(parameters, **optimizer_params)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")

class SchedulerConfig(BaseModel):
    scheduler_type: Literal["StepLR", "CosineAnnealingLR", "MultiStepLR"]
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    T_max: Optional[int] = None
    eta_min: Optional[float] = None
    last_epoch: Optional[int] = -1
    milestones: Optional[List[int]] = None

    class Config:
        validate_assignment = True
        frozen = True

    def model_dump(self, *args, **kwargs):
        # Only export relevant fields based on scheduler_type
        fields = {"scheduler_type", "last_epoch"}
        if self.scheduler_type == "StepLR":
            fields.update({"step_size", "gamma"})
        elif self.scheduler_type == "CosineAnnealingLR":
            fields.update({"T_max", "eta_min"})
        elif self.scheduler_type == "MultiStepLR":
            fields.update({"milestones", "gamma"})

        # Add other scheduler types as needed

        return super().model_dump(include=fields, *args, **kwargs)

    def factory(self, optimizer: torch.optim.Optimizer):
        scheduler_type = self.scheduler_type
        scheduler_params = self.model_dump(exclude={"scheduler_type"})

        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
        elif scheduler_type == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
class Config(BaseModel):
    # Dataset & loader
    num_training_samples: int
    batch_size: int = 128

    # Training loop
    num_epochs: Optional[int] = None
    num_steps: Optional[int] = None
    logging_steps: Optional[Set[int]] = None
    checkpoint_steps: Optional[Set[int]] = None

    # Optimizer
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler_config: Optional[SchedulerConfig] = None

    # Wandb: if omitted, will not log to wandb
    project: Optional[str] = None
    entity: Optional[str] = None

    # Misc
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # TODO: Make this frozen
    # class Config:
    #     frozen = True

    def __init__(self, logging_steps=None, checkpoint_steps=None, **data):
        super().__init__(**data)
        assert (
            self.num_epochs is not None or self.num_steps is not None
        ), "Must specify either num_epochs or num_steps"

        if self.num_steps is None:
            self.num_steps = self.num_epochs * self.num_steps_per_epoch

        if self.num_epochs is None:
            self.num_epochs = (self.num_steps // self.num_steps_per_epoch) + 1

        if isinstance(logging_steps, tuple) and len(logging_steps) == 2:
            self.logging_steps = self._process_steps(logging_steps)
        else:
            self.logging_steps = set(logging_steps)

        if isinstance(checkpoint_steps, tuple) and len(checkpoint_steps) == 2:
            self.checkpoint_steps = self._process_steps(checkpoint_steps)
        else:
            self.checkpoint_steps = set(checkpoint_steps)

        if self.is_wandb_enabled:
            wandb_msg = f"Logging to wandb enabled (project: {self.project}, entity: {self.entity})"
            logger.info(wandb_msg)
        else:
            logger.info("Logging to wandb disabled")

        logger.info(yaml.dump(self.dict()))

    def _process_steps(
        self, num_steps: Tuple[Optional[int], Optional[int]]
    ) -> Set[int]:
        """
        Processes steps for logging & taking checkpoints, allowing customization of intervals.

        Args:
            num_steps: A tuple (x, y) with optional integer values:
            - x: Specifies the number of steps to take on a linear interval.
            - y: Specifies the number of steps to take on a log interval.

        Returns:
            A set of step numbers at which logging or checkpointing should occur.

        Raises:
            ValueError: If the num_steps input is not valid.

        """

        if num_steps is None:
            return set()

        if isinstance(num_steps, tuple):
            result = set()

            if num_steps[0] is not None:
                result |= int_linspace(0, self.num_steps, num_steps[0], return_type="set")  # type: ignore

            if num_steps[1] is not None:
                result |= int_logspace(1, self.num_steps, num_steps[1], return_type="set")  # type: ignore

            return result

        raise ValueError(f"Invalid num_steps: {num_steps}")

    @property
    def is_wandb_enabled(self):
        if self.entity is not None and self.project is None:
            warnings.warn(
                "Wandb entity is specified but project is not. Disabling wandb."
            )

        return self.project is not None

    @property
    def num_steps_per_epoch(self):
        return self.num_training_samples // self.batch_size

    @validator("device", pre=True)
    def validate_device(cls, value):
        return torch.device(value)

    def model_dump(self, *args, **kwargs):
        config_dict = super().model_dump(*args, **kwargs)
        config_dict["optimizer_config"] = self.optimizer_config.model_dump()

        if self.scheduler_config is not None:
            config_dict["scheduler_config"] = self.scheduler_config.model_dump()
        else:
            config_dict["scheduler_config"] = None

        return config_dict
