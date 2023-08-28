import itertools
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import numpy as np
import torch
import yaml
from pydantic import BaseModel, Field, validator
from torch import nn
from torch.nn import functional as F

from devinterp.config import OptimizerConfig
from devinterp.slt.ensemble import Ensemble
from devinterp.slt.metrics import Metric
from devinterp.slt.sgld import SGLD


class SamplerConfig(BaseModel):
    optimizer_config: OptimizerConfig  # This isn't working for some unknown reason
    num_chains: int = 5
    num_draws_per_chain: int = 10
    num_burnin_steps: int = 0
    num_steps_bw_draws: int = 1
    verbose: bool = False
    batch_size: int = 256

    class Config:
        validate_assignment = True
        # frozen = True


class Sampler:
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

        print(yaml.dump(config.model_dump()))

    @property
    def num_samples(self):
        return len(self.data)

    def sample(
        self,
        metrics: Optional[Dict[str, Metric]] = None,
        summary_fn: Callable = lambda **kwargs: kwargs,
    ):
        self.ensemble.train()
        self.ensemble.zero_grad()

        metrics = metrics or {}
        if "losses" not in metrics:
            metrics["losses"] = lambda xs, ys, yhats, losses, loss, model: loss

        loss_init = 0
        draws = {m: [[] for _ in range(self.ensemble.num_chains)] for m in metrics}
        num_steps = (
            self.config.num_draws_per_chain * self.config.num_steps_bw_draws
            + self.config.num_burnin_steps
        )

        for i, (xs, ys) in zip(range(num_steps), itertools.cycle(self.loader)):
            for j, model in enumerate(self.ensemble):
                yhats = model(xs)
                losses = F.mse_loss(yhats, ys, reduction="none")
                loss = losses.mean()
                loss.backward(retain_graph=True)

                if (
                    i >= self.config.num_burnin_steps
                    and i % self.config.num_steps_bw_draws == 0
                ):
                    for m, fn in metrics.items():
                        draws[m][j].append(fn(xs, ys, yhats, losses, loss, model))

                if i == 0 and j == 0:
                    loss_init = loss.item()

            self.optimizer.step()
            self.optimizer.zero_grad()

        draws = {m: torch.Tensor(v) for m, v in draws.items()}

        return summary_fn(loss_init=loss_init, num_samples=self.num_samples, **draws)


def estimate_free_energy(losses, num_samples: int, **_):
    """
    Estimate the free energy, $E[nL_n]$.
    """
    loss_avg = losses.mean()
    free_energy_estimate = loss_avg * num_samples

    return free_energy_estimate.item()


def estimate_rlct(loss_init, losses, num_samples: int, **_):
    r"""
    Estimate $\hat\lambda$, using the WBIC.
    """
    loss_avg = losses.mean()
    rlct_estimate = (loss_avg - loss_init) * num_samples / np.log(num_samples)

    return rlct_estimate.item()


# def estimate_sing_fluc(loss_sum, loss_sq_sum, num_epochs: int):
#     r"""
#     Estimate the singular fluctuation, $\hat\nu$.

#     TODO: I don't understand what the right value/name for num_epochs is. In Edmund's original code,
#     he iterates over the entire dataloader (and increments num_epochs) each time he draws the loss_sum and loss_sq_sum.
#     I don't think this is necessary. I think we can get around it with a batch based approach.
#     """
#    return ((loss_sq_sum - loss_sum * loss_sum / num_epochs) / (num_epochs - 1)).sum()


def compose_summary_fns(**summary_fns):
    def fn(**metrics):
        output = {}
        for name, fn in summary_fns.items():
            output[name] = fn(**metrics)

        return output

    return fn


slt_summary_fn = compose_summary_fns(
    free_energy=estimate_free_energy,
    rlct=estimate_rlct,
    # sing_fluc=estimate_sing_fluc
)
