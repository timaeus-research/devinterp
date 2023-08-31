import functools
import logging
import math
import random
import warnings
from typing import Any, Callable, Container, Dict, List, Optional, Set, Tuple, TypedDict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field, validator
from torch.optim.lr_scheduler import LambdaLR
from tqdm.notebook import tqdm

import wandb
from devinterp.data import CustomDataloader
from devinterp.ops.logging import MetricLogger, MetricLoggingConfig
from devinterp.ops.storage import BaseStorageProvider, CheckpointerConfig
from devinterp.optim.optimizers import OptimizerConfig
from devinterp.optim.schedulers import LRScheduler, SchedulerConfig
from devinterp.utils import int_linspace, int_logspace

logger = logging.getLogger(__name__)


class LearnerStateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Optional[Dict]


class Learner:
    def __init__(
        self,
        model: torch.nn.Module,
        train_set: torch.utils.data.Dataset,
        test_set: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        config: LearnerConfig,
        scheduler: Optional[LRScheduler] = None,
        logger: Optional[MetricLogger] = None,
        checkpointer: Optional[BaseStorageProvider] = None,
        metrics: Optional[List[Callable[["Learner"], Dict]]] = None,
    ):
        """
        Initializes the Learner object.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            train_set (torch.utils.data.Dataset): The training dataset.
            test_set (torch.utils.data.DataLoader): The test DataLoader.
            config (Config): Configuration object containing hyperparameters.
            metrics (Optional[List[Callable]]): List of metric functions to evaluate the model.

        """

        self.config = config
        self.model = model
        self.train_set = train_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = CustomDataloader(train_set, batch_size=config.batch_size)
        self.test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=config.batch_size, shuffle=False
        )
        self.metrics = metrics or []
        self.logger = logger
        self.checkpointer = checkpointer

    def measure(self) -> Dict[str, Any]:
        """
        Applies metrics to the current state of the model and returns results.

        Returns:
            Dict: Metrics calculated from the current model state.
        """
        return functools.reduce(
            lambda x, y: x | y, [metric(self) for metric in self.metrics], {}
        )

    def resume(self, batch_idx: Optional[int] = None):
        """
        Resumes training from a saved checkpoint.

        Args:
            batch_idx (Optional[int]): Batch index to resume training from.

        """
        if self.checkpointer is None:
            raise ValueError("Cannot resume training without a checkpointer.")

        if batch_idx is None:
            epoch, batch_idx = self.checkpointer[-1]
        else:
            epoch, batch = min(self.checkpointer, key=lambda x: abs(x[1] - batch_idx))

            if batch != batch_idx:
                warnings.warn(
                    f"Could not find checkpoint with batch_idx {batch_idx}. Resuming from closest batch ({batch}) instead."
                )

        self.load_checkpoint(epoch, batch_idx)

    def train(self, resume=0, run_id: Optional[str] = None):
        """
        Trains the model.

        Args:
            resume (int): Batch index to resume training from. If 0, will not resume.
            run_id (Optional[str]): Wandb run ID to resume from. If None, will create a new run.

        """

        if resume:
            self.resume(resume)

        self.model.to(self.config.device)
        self.model.train()

        if self.config.is_wandb_enabled:
            if resume and not run_id:
                warnings.warn(
                    "Resuming from checkpoint but no run_id provided. Will not log to existing wandb run."
                )

            if not run_id:
                wandb.init(project=self.config.project, entity=self.config.entity)
            else:
                wandb.init(
                    project=self.config.project,
                    entity=self.config.entity,
                    run_id=run_id,
                )

        pbar = tqdm(
            total=self.config.num_steps,
            desc=f"Epoch 0 Batch 0/{self.config.num_steps} Loss: ?.??????",
        )

        for epoch in range(0, self.config.num_epochs):
            self.set_seed(epoch)

            for _batch_idx, (data, target) in enumerate(self.train_loader):
                batch_idx = self.config.num_steps_per_epoch * epoch + _batch_idx
                data, target = data.to(self.config.device), target.to(
                    self.config.device
                )
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                # Update progress bar description
                pbar.set_description(
                    f"Epoch {epoch} Batch {batch_idx}/{self.config.num_steps} Loss: {loss.item():.6f}"
                )
                pbar.update(1)

                if self.config.is_wandb_enabled:
                    # TODO: Figure out how to make this work with Logger
                    wandb.log({"Batch/Loss": loss.item()}, step=batch_idx)

                # Log to wandb & save checkpoints according to log_steps
                if batch_idx in self.config.checkpoint_steps:
                    self.save_checkpoint(epoch, batch_idx)

                if batch_idx in self.config.logging_steps:
                    self.logger.log(self.measure(), step=batch_idx)
                    self.model.train()

            pbar.close()

        if self.config.is_wandb_enabled:
            wandb.finish()

    def state_dict(self) -> LearnerStateDict:
        """
        Returns the current state of the Learner in a LearnerStateDict format.

        Returns:
            LearnerStateDict: Dictionary containing model, optimizer, and scheduler states.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }

    def load_state_dict(self, checkpoint: LearnerStateDict):
        """
        Loads a LearnerStateDict into the Learner.

        Args:
            checkpoint (LearnerStateDict): Dictionary containing model, optimizer, and scheduler states.
        """
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.scheduler is not None and checkpoint["scheduler"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

    def save_checkpoint(self, epoch: int, batch_idx: int):
        """
        Saves a checkpoint of the current Learner state.

        Args:
            epoch (int): Epoch to save checkpoint at.
            batch_idx (int): Batch index to save checkpoint at.
        """
        checkpoint = self.state_dict()
        self.checkpointer.save_file((epoch, batch_idx), checkpoint)

    def load_checkpoint(self, epoch: int, batch_idx: int):
        """
        Loads a checkpoint of the Learner state.

        Args:
            epoch (int): Epoch to load checkpoint from.
            batch_idx (int): Batch index to load checkpoint from.
        """
        checkpoint = self.checkpointer.load_file((epoch, batch_idx))
        self.load_state_dict(checkpoint)

    def set_seed(self, seed: int):
        """
        Sets the seed for the Learner.

        Args:
            seed (int): Seed to set.
        """
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        self.generator.manual_seed(seed)

        if "cuda" in str(self.config.device):
            torch.cuda.manual_seed_all(seed)


class LearnerConfig(BaseModel):
    # Dataset & loader
    num_training_samples: int
    batch_size: int = 128

    # Training loop
    # num_epochs: int = None
    num_steps: int = 100_000
    logging_config: Optional[MetricLoggingConfig] = None
    checkpointer_config: Optional[CheckpointerConfig] = None

    # Optimizer
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler_config: Optional[SchedulerConfig] = None

    # Misc
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    # TODO: Make this frozen
    class Config:
        frozen = True

    def __init__(self, **data):
        super().__init__(**data)

        if self.is_wandb_enabled:
            wandb_msg = f"Logging to wandb enabled (project: {self.project}, entity: {self.entity})"
            logger.info(wandb_msg)
        else:
            logger.info("Logging to wandb disabled")

        logger.info(
            yaml.dump(self.model_dump(exclude=("logging_steps", "checkpoint_steps")))
        )

    # Properties

    @property
    def num_steps_per_epoch(self):
        """Number of steps per epoch."""
        return self.num_training_samples // self.batch_size

    @property
    def num_epochs(self):
        """Number of epochs."""
        return math.ceil(self.num_steps / self.num_steps_per_epoch)

    @property
    def is_wandb_enabled(self):
        """Whether wandb is enabled."""
        return self.logging_config and self.logging_config.is_wandb_enabled

    # Validators

    @validator("logging_steps", "checkpoint_steps", pre=True, always=True)
    @classmethod
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
    @classmethod
    def validate_device(cls, value):
        """Validates `device` field."""
        return str(torch.device(value))

    def model_dump(self, *args, **kwargs):
        """Dumps the model configuration to a dictionary."""
        config_dict = super().model_dump(*args, **kwargs)
        config_dict["optimizer_config"] = self.optimizer_config.model_dump()

        if self.scheduler_config is not None:
            config_dict["scheduler_config"] = self.scheduler_config.model_dump()
        else:
            config_dict["scheduler_config"] = None

        return config_dict

    def factory(
        self,
        model: torch.nn.Module,
        train_set: torch.utils.data.Dataset,
        test_set: torch.utils.data.DataLoader,
        metrics: Optional[List[Callable[["Learner"], Dict]]] = None,
    ):
        """Produces a Learner object from the configuration."""
        optimizer = self.optimizer_config.factory(model.parameters())

        if self.scheduler_config is not None:
            scheduler = self.scheduler_config.factory(optimizer)
        else:
            scheduler = None

        logger = self.logging_config.factory() if self.logging_config else None
        checkpointer = (
            self.checkpointer_config.factory() if self.checkpointer_config else None
        )

        return Learner(
            model=model,
            train_set=train_set,
            test_set=test_set,
            optimizer=optimizer,
            scheduler=scheduler,
            config=self,
            logger=logger,
            checkpointer=checkpointer,
            metrics=metrics,
        )
