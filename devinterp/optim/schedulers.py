from typing import Any, Callable, Dict, List, Literal, Optional, Protocol, Union

import torch
from pydantic import BaseModel


class LRScheduler(Protocol):
    def step(self, epoch: Optional[int] = None):
        ...

    def get_last_lr(self) -> List[float]:
        ...

    def load_state_dict(self, state_dict) -> None:
        ...

    def state_dict(self) -> dict:
        ...

    def print_lr(
        self,
        is_verbose: bool,
        group: Dict[str, Any],
        lr: float,
        epoch: Union[int, None] = ...,
    ) -> None:
        ...


class SchedulerConfig(BaseModel):
    scheduler_type: Literal[
        "StepLR", "CosineAnnealingLR", "MultiStepLR", "LambdaLR", "OneCycleLR"
    ]
    step_size: Optional[int] = None
    gamma: Optional[float] = None
    T_max: Optional[int] = None
    eta_min: Optional[float] = None
    last_epoch: Optional[int] = -1
    milestones: Optional[List[int]] = None
    lr_lambda: Optional[Callable[[int], float]] = None
    # for OneCycleLR
    max_lr: Optional[float] = None
    total_steps: Optional[int] = None
    anneal_strategy: Optional[str] = None
    div_factor: Optional[float] = None
    final_div_factor: Optional[float] = None
    pct_start: Optional[float] = None
    cycle_momentum: Optional[bool] = None

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
        elif self.scheduler_type == "OneCycleLR":
            fields.update(
                {
                    "max_lr",
                    "total_steps",
                    "anneal_strategy",
                    "div_factor",
                    "final_div_factor",
                    "pct_start",
                    "cycle_momentum",
                }
            )

        # Add other scheduler types as needed

        return super().model_dump(include=fields, *args, **kwargs)

    def factory(self, optimizer: torch.optim.Optimizer) -> LRScheduler:
        scheduler_type = self.scheduler_type
        scheduler_params = self.model_dump(exclude={"scheduler_type"})

        if scheduler_type == "StepLR":
            return torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
        elif scheduler_type == "CosineAnnealingLR":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, **scheduler_params
            )
        elif scheduler_type == "MultiStepLR":
            return torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler_params)
        elif scheduler_type == "LambdaLR" and self.lr_lambda is not None:
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=self.lr_lambda
            )
        elif scheduler_type == "OneCycleLR":
            return torch.optim.lr_scheduler.OneCycleLR(optimizer, **scheduler_params)
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
