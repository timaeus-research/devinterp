import logging
import warnings
from typing import (Callable, Iterable, List, Literal, Optional, Set, Tuple,
                    Union)

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
    scheduler_type: Literal["StepLR", "CosineAnnealingLR", "MultiStepLR", "LambdaLR"] 
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    T_max: Optional[int] = None
    eta_min: Optional[float] = None
    last_epoch: Optional[int] = -1
    milestones: Optional[List[int]] = None
    lr_lambda: Optional[Callable[[int], float]] = None

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
        elif scheduler_type == "LambdaLR" and self.lr_lambda is not None:
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_lambda)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
class Config(BaseModel):
    # Dataset & loader
    num_training_samples: int
    batch_size: int = 128

    # Training loop
    # num_epochs: int = None
    num_steps: int = None
    logging_steps: Set[int] = None
    checkpoint_steps: Set[int] = None

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

    @property
    def num_steps_per_epoch(self):
        return self.num_training_samples // self.batch_size

    # @validator("num_steps", pre=True, always=True)
    # def validate_num_steps(cls, value, values):
    #     print("steps", value, values)
    #     if value is None:
    #         return values['num_epochs'] * values['num_training_samples'] // values['batch_size']
    
    # @validator("num_epochs", pre=True, always=True)
    # def validate_num_epochs(cls, value, values):
    #     print("epochs", value, values)
    #     if value is None:
    #         return (values['num_steps'] * values['batch_size'] // values['num_training_samples']) + 1
    
    @validator("optimizer_config", pre=True)
    def validate_optimizer_config(cls, value):
        if isinstance(value, dict):
            return OptimizerConfig(**value)
        return value
    
    @validator("scheduler_config", pre=True)
    def validate_scheduler_config(cls, value):
        if isinstance(value, dict):
            return SchedulerConfig(**value)
        return value

    @validator("logging_steps", "checkpoint_steps", pre=True, always=True)
    def validate_steps(cls, value, values):
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

        if isinstance(value, tuple) and len(value) == 2:
            result = set()

            if value[0] is not None:
                result |= int_linspace(0, values["num_steps"], value[0], return_type="set")  # type: ignore

            if value[1] is not None:
                result |= int_logspace(1, values["num_steps"], value[1], return_type="set")  # type: ignore

            return result
        elif value is None:
            return set()
        return set(value)

    @validator("device", pre=True)
    def validate_device(cls, value):
        return str(torch.device(value))

    def __init__(self, **data):
        super().__init__(**data)
        if self.is_wandb_enabled:
            wandb_msg = f"Logging to wandb enabled (project: {self.project}, entity: {self.entity})"
            logger.info(wandb_msg)
        else:
            logger.info("Logging to wandb disabled")

        logger.info(yaml.dump(self.model_dump(exclude=("logging_steps", "checkpoint_steps"))))

    @property
    def is_wandb_enabled(self):
        if self.entity is not None and self.project is None:
            warnings.warn(
                "Wandb entity is specified but project is not. Disabling wandb."
            )

        return self.project is not None
    
    def model_dump(self, *args, **kwargs):
        config_dict = super().model_dump(*args, **kwargs)
        config_dict["optimizer_config"] = self.optimizer_config.model_dump()

        if self.scheduler_config is not None:
            config_dict["scheduler_config"] = self.scheduler_config.model_dump()
        else:
            config_dict["scheduler_config"] = None

        return config_dict
