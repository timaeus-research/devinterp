
import csv
import logging
import warnings

import pandas as pd
import yaml

import wandb


class Logger:
    """
    Logger class for handling logging tasks. Has three modes (which can be used in combination):
      - Logging to wandb. If you provide a `project` and `entity`, this will write logs to wandb.
      - Logging to a CSV file. If you provide an `out_file`, this will write logs to a CSV file.
      - Logging to a Pandas DataFrame. If you set `use_df` to True, this will write logs to a Pandas DataFrame. 

    Parameters:
        project (str, optional): Name of the wandb project. If both project and entity are provided, will log to wandb.
        entity (str, optional): Name of the wandb entity. If both project and entity are provided, will log to wandb.
        logging_steps (list[int], required): List of steps at which logging will occur. Must be provided in ascending order.
        metrics (list[str], optional): List of metric names that will be logged.
        out_file (str, optional): Path to the CSV file where logs will be saved. If provided, logs will be written to this file.
        use_df (bool, optional): If True, logs will be saved to a Pandas DataFrame within the Logger object.

    Methods:
        log(data: dict, step: int): Logs the provided data at the specified step.
        get_dataframe(): Returns the DataFrame containing the logs if use_df is True, otherwise raises an error.
    """

    def __init__(self, project=None, entity=None, logging_steps=[], metrics=[], out_file=None, use_df=False):
        self.project = project
        self.entity = entity
        self.logging_steps = sorted(list(logging_steps))
        self.metrics = metrics
        self.out_file = out_file
        self.use_df = use_df
        self.current_step_idx = 0
        self.buffer = {}
        self._logger = logging.getLogger(project)

        if self.use_df:
            self.df = pd.DataFrame(index=logging_steps, columns=["step"] + self.metrics)
        else:
            self.df = None

        if self.out_file:
            with open(self.out_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["step"] + self.metrics)

        if not self.is_wandb_enabled and not self.use_df and not self.out_file:
            warnings.warn("Must provide either project and entity, out_file, or use_df")

    @property
    def is_wandb_enabled(self):
        return bool(self.project and self.entity)

    def log(self, data, step):
        if self.current_step_idx >= len(self.logging_steps) or step != self.logging_steps[self.current_step_idx]:
            raise ValueError(f"Step must be one of the logging_steps ({self.logging_steps}) in correct order")

        if self.is_wandb_enabled:
            wandb.log(data, step=step)

        self.buffer.update(data)
        self.buffer["step"] = step

        if self.use_df:
            self.df.loc[step] = self.buffer

        if self.out_file:
            with open(self.out_file, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([self.buffer[metric] if metric in self.buffer else None for metric in ["step"] + self.metrics])

        self.debug(yaml.dump(self.buffer))
            
        self.current_step_idx += 1

    def debug(self, msg):
        self._logger.debug(msg)

    def info(self, msg):
        self._logger.info(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def error(self, msg):
        self._logger.error(msg)

    