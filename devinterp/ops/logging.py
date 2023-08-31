import csv
import dataclasses
import logging
import warnings
from  import FileDescriptorOrPath
from typing import Any, Dict, List, Optional, Protocol

import pandas as pd
from pydantic import BaseModel
import torch
import yaml

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
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        """Log the data at a specific step."""
        wandb.log(data, step=step, commit=commit, sync=sync)


class CsvLogger(MetricLogger):
    """Logs data to a CSV file."""

    def __init__(self, out_file: FileDescriptorOrPath, metrics):
        self.out_file = out_file
        self.metrics = metrics
        self.current_step = -1
        self.buffer = {}

    def log(self, data, step=None, commit=None, sync=None):
        """Log the data at a specific step."""
        if step is not None:
            if step > self.current_step:
                self._commit()
                self.current_step = step
                self.buffer.update(data)
            else:
                warning = f"Step must be greater than {self.current_step}, got {step}. Ignoring."
                warnings.warn(warning)

        if commit:
            self._commit()
            self.current_step += 1  # Auto-increment

    def _commit(self):
        """Write the buffered data to the CSV file."""
        with open(self.out_file, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [self.buffer.get(metric, None) for metric in ["step"] + self.metrics]
            )
        self.buffer.clear()


class DataFrameLogger(MetricLogger):
    """Logs data to a Pandas DataFrame.

    Parameters
        logging_steps (list[int], required): List of steps at which logging will occur. Must be provided in ascending order.
        metrics (list[str], required): List of metric names that will be logged.

    """

    def __init__(self, logging_steps: List[int], metrics: List[str]):
        self.df = pd.DataFrame(index=logging_steps, columns=["step"] + metrics)
        self.current_step = -1
        self.buffer = {}

    def log(self, data, step=None, commit=None, sync=None):
        """Log the data at a specific step."""
        if step is not None:
            if step > self.current_step:
                self._commit()
                self.current_step = step
                self.buffer.update(data)
            else:
                warning = f"Step must be greater than {self.current_step}, got {step}. Ignoring."
                warnings.warn(warning)

        if commit:
            self._commit()
            self.current_step += 1  # Auto-increment

    def _commit(self):
        """Write the buffered data to the DataFrame."""
        self.df.loc[self.current_step] = self.buffer
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

    def log(
        self,
        data: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
        sync: Optional[bool] = None,
    ):
        for logger in self.loggers:
            logger.log(data, step, commit, sync)

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


def Logger(
        project=None,
        entity=None,
        logging_steps=[],
        metrics=[],
        out_file=None,
        use_df=False,
    ):
    """For backwards-compatibility. Use CompositeLogger instead."""
    warnings.warn("Logger is deprecated. Use CompositeLogger instead.")

    loggers = []

    if project and entity:
        loggers.append(WandbLogger(project, entity))
    
    if logging_steps:
        if use_df:
            loggers.append(DataFrameLogger(logging_steps, metrics))
        else:
            loggers.append(CsvLogger(out_file, metrics))

    return CompositeLogger(loggers)