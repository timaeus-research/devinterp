import itertools
from dataclasses import dataclass
from logging import Logger
from typing import Callable, Dict, Literal, Optional, Union

import torch
import yaml
from pydantic import BaseModel, Field, validator
from torch import nn
from torch.nn import functional as F

from devinterp.optim.optimizers import OptimizerConfig
from devinterp.optim.sgld import SGLD
from devinterp.slt.ensemble import Ensemble
from devinterp.slt.observables import (
    MicroscopicObservable,
    estimate_free_energy,
    estimate_rlct,
)
from devinterp.utils import get_criterion

logger = Logger(__name__)


class SamplerConfig(BaseModel):
    """
    Configuration class for Sampler. Specifies optimizer, number of chains, sampling settings, etc.

    Attributes (example):
        optimizer_config (OptimizerConfig): Configuration for the optimizer.
        num_chains (int): Number of independent chains for sampling.
        num_draws_per_chain (int): Number of draws per chain.
        num_burnin_steps (int): Number of burnin steps.
        verbose (bool): Whether to print verbose output.
        batch_size (int): Batch size for dataloader.
    """

    optimizer_config: OptimizerConfig  # This isn't working for some unknown reason
    num_chains: int = 5
    num_draws_per_chain: int = 10
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    verbose: bool = False
    batch_size: int = 256
    criterion: Literal["mse_loss", "cross_entropy"]

    class Config:
        validate_assignment = True
        # frozen = True

    def factory(self, model: nn.Module, data: torch.utils.data.Dataset):
        return Sampler(model, data, self)


class Sampler:
    """
    Class for sampling from a given model using a specified optimizer and dataset.

    Attributes:
        config (SamplerConfig): Configuration for sampling.
        model (nn.Module): The original model.
        ensemble (Ensemble): Container for independent model instances.
        optimizer (Optimizer): The optimizer to use for sampling.
        data (Dataset): Dataset for sampling.
        loader (DataLoader): DataLoader for the dataset.
    """

    def __init__(
        self, model: nn.Module, data: torch.utils.data.Dataset, config: SamplerConfig
    ):
        self.config = config
        self.model = model
        self.ensemble = Ensemble(model, num_chains=config.num_chains)
        self.optimizer = config.optimizer_config.factory(self.ensemble.parameters())
        self.data = data
        self.loader = torch.utils.data.DataLoader(
            data, batch_size=config.batch_size, shuffle=True
        )
        self.criterion = get_criterion(config.criterion)

        logger.info(yaml.dump(config.model_dump()))

    def sample(
        self,
        observables: Optional[Dict[str, MicroscopicObservable]] = None,
        summary_fn: Optional[Callable] = None,
    ):
        """
        Performs the sampling process, returning metric summaries as specified.

        Parameters:
            metrics (Optional[Dict[str, Metric]]): Metrics to compute during sampling.
            summary_fn (Callable): Function to summarize metrics after sampling.
        """
        self.ensemble.train()
        self.ensemble.zero_grad()

        observables = observables or {}
        if "losses" not in observables:
            observables["losses"] = lambda xs, ys, yhats, losses, loss, model: loss

        loss_init = 0
        draws = {m: [[] for _ in range(self.ensemble.num_chains)] for m in observables}
        num_steps = (
            self.config.num_draws_per_chain * self.config.num_steps_bw_draws
            + self.config.num_burnin_steps
        )

        for i, (xs, ys) in zip(range(num_steps), itertools.cycle(self.loader)):
            for j, model in enumerate(self.ensemble):
                yhats = model(xs)
                losses = self.criterion(yhats, ys, reduction="none")
                loss = losses.mean()
                loss.backward(retain_graph=True)

                if (
                    i >= self.config.num_burnin_steps
                    and i % self.config.num_steps_bw_draws == 0
                ):
                    for m, fn in observables.items():
                        draws[m][j].append(fn(xs, ys, yhats, losses, loss, model))

                if i == 0 and j == 0:
                    loss_init = loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        draws = {m: torch.Tensor(v) for m, v in draws.items()}

        if summary_fn:
            return summary_fn(loss_init=loss_init, num_samples=len(self.data), **draws)

        return draws
