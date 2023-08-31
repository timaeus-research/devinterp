import csv
import dataclasses
import logging
import warnings
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
import torch
import yaml
from pydantic import BaseModel

import wandb


class MetricLogger:
    """
    Base class for logging. Subclasses should implement the `log` method.
    """

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        raise NotImplementedError


class WandbLogger(MetricLogger):
    """Logs data to Weights & Biases (wandb) platform.

    Parameters:
        project (str, optional): Name of the wandb project.
        entity (str, optional): Name of the wandb entity.
    """

    def __init__(self, project, entity):
        self.project = project
        self.entity = entity

    def log(
        self,
        data: Dict[str, Any],
        *args,
        **kwargs,
    ):
        """Log the data at a specific step."""
        wandb.log(data, *args, **kwargs)


class CsvLogger(MetricLogger):
    """Logs data to a CSV file."""

    def __init__(self, out_file: str, metrics):
        self.out_file = out_file
        self.metrics = metrics
        self.current_step = -1
        self.buffer = {}

    def log(self, data, step=None, commit=None, **_):
        """Log the data at a specific step."""
        if step is not None:
            if step < self.current_step:
                warning = f"Step must be greater than {self.current_step}, got {step}. Ignoring."
                warnings.warn(warning)
                return
            elif step > self.current_step:
                if self.current_step >= 0:
                    self._commit()

                self.current_step = step

        self.buffer.update(data)

        if commit:
            self._commit()
            self.current_step += 1  # Auto-increment

    def _commit(self):
        """Write the buffered data to the CSV file."""
        if self.current_step < 0:
            return

        with open(self.out_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [self.current_step]
                + [self.buffer.get(metric, None) for metric in self.metrics]
            )
        self.buffer.clear()


class DataFrameLogger(MetricLogger):
    """Logs data to a Pandas DataFrame.

    Parameters
        logging_steps (list[int], required): List of steps at which logging will occur. Must be provided in ascending order.
        metrics (list[str], required): List of metric names that will be logged.

    """

    def __init__(self, logging_steps: List[int], metrics: List[str], dtype=float):
        self.df = pd.DataFrame(
            index=range(len(logging_steps)), columns=["step"] + metrics, dtype=dtype
        )
        self.df["step"] = logging_steps
        self.current_step = -1
        self.buffer = {}

    def log(self, data, step=None, commit=None, **_):
        """Log the data at a specific step."""
        if step is not None:
            if step not in self.df.step.values:
                warning = f"Step {step} not in logging_steps. Ignoring."
                warnings.warn(warning)
                return

            if step != self.current_step:
                self._commit()

            self.current_step = step

        self.buffer.update(data)

        if commit:
            self._commit()

    def _commit(self):
        """Write the buffered data to the DataFrame."""
        if self.current_step < 0:
            return

        for key, value in self.buffer.items():
            self.df.loc[self.df.step == self.current_step, key] = value

        self.buffer.clear()


class StdLogger(MetricLogger):
    """A wrapper for the standard Logger that uses the MetricLogger interface."""

    def __init__(self, project):
        self._logger = logging.getLogger(project)

    def format(self, obj):
        """Format the object for logging."""
        if isinstance(obj, BaseModel):
            return yaml.dump(obj.dict())
        elif dataclasses.is_dataclass(obj):
            return yaml.dump(dataclasses.asdict(obj))
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()  # or any other method to format the tensor
        elif isinstance(obj, dict):
            return yaml.dump({k: self.format(v) for k, v in obj.items()})
        else:
            return str(obj)

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        # Rendering logic here
        self._logger.info(self.format(data))

    def debug(self, data):
        self._logger.debug(self.format(data))

    def info(self, data):
        self._logger.info(self.format(data))

    def warning(self, data):
        self._logger.warning(self.format(data))

    def error(self, data):
        self._logger.error(self.format(data))


class CompositeLogger(MetricLogger):
    def __init__(self, loggers):
        self.loggers = loggers

    def log(self, data: Dict[str, Any], *args, **kwargs):
        for logger in self.loggers:
            logger.log(data, *args, **kwargs)

    def info(self, msg):
        for logger in self.loggers:
            if hasattr(logger, "info"):
                logger.info(msg)

    def debug(self, msg):
        for logger in self.loggers:
            if hasattr(logger, "debug"):
                logger.debug(msg)

    def warning(self, msg):
        for logger in self.loggers:
            if hasattr(logger, "warning"):
                logger.warning(msg)


class MetricLoggingConfig(BaseModel):
    project: Optional[str] = None
    entity: Optional[str] = None
    logging_steps: Optional[List[int]] = None
    metrics: Optional[List[str]] = None
    out_file: Optional[str] = None
    use_df: Optional[bool] = False
    stdout: Optional[bool] = False

    class Config:
        frozen = True

    @property
    def is_wandb_enabled(self):
        if self.entity is not None and self.project is None:
            warnings.warn(
                "Wandb entity is specified but project is not. Disabling wandb."
            )

        return self.project is not None

    def factory(self):
        """Creates a MetricLogger based on the configuration."""
        loggers = []

        if self.project and self.entity:
            loggers.append(WandbLogger(self.project, self.entity))

        if self.logging_steps:
            if self.use_df and self.metrics:
                loggers.append(DataFrameLogger(self.logging_steps, self.metrics))
            elif self.out_file:
                loggers.append(CsvLogger(self.out_file, self.metrics))

        if self.stdout:
            loggers.append(StdLogger(self.project or "MetricLogger"))

        return CompositeLogger(loggers)


def Logger(
    project=None,
    entity=None,
    logging_steps=[],
    metrics=[],
    out_file=None,
    use_df=False,
):
    """For backwards-compatibility. Use CompositeLogger or MetricLoggingConfig.factory instead."""
    warnings.warn("Logger is deprecated. Use CompositeLogger instead.")
    return MetricLoggingConfig(
        project=project,
        entity=entity,
        logging_steps=logging_steps,
        metrics=metrics,
        out_file=out_file,
        use_df=use_df,
    ).factory()
