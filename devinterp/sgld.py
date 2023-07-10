"""

Based on code from [Javier Antoran](https://github.com/JavierAntoran/Bayesian-Neural-Networks/blob/master/src/Stochastic_Gradient_Langevin_Dynamics/optimizers.py)

TODO: Also borrow pSGLD?
"""

import copy
from typing import Any, Callable, Generator, Iterable, Literal, Tuple, Union

import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
from torch.utils.data import DataLoader

from devinterp.utils import Reduction, ReturnTensor, convert_tensor, reduce_tensor


class SGLD(Optimizer):
    r"""
    SGLD is from M. Welling, Y. W. Teh "Bayesian Learning via Stochastic Gradient Langevin Dynamics"
    We use equation (4) there, which says given $n$ samples $x_1, ... , x_N$,

    $$
        w' - w = \epsilon_t / 2 ( \grad \log p(w) - N \grad L'_N(w) ) + eta_t
    $$

    where $\eta_t$ is Gaussian noise, sampled from $\mathcal N(0, \epsilon_t)$, and $p(w)$ is a prior. We take
    this to be Gaussian centered at some fixed $w_0$ parameter, with covariance matrix $\lambda I_d$.

    $$
        p(w) \propto \exp(-1/(2\lambda)|| w - w_0 ||^2)
        \grad \log p(w) = \grad( -1/(2\lambda)|| w - w_0 ||^2 ) = -\lambda^{-1}( w - w_0 ).
    $$

    We use a tempered posterior, which means replacing $L'_N$ by $\beta L'_N$, at inverse temperature
    $\beta = 1/\log{N}$.

    """

    def __init__(self, params, lr=required, weight_decay=0.1, noise=0.0):
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr, weight_decay=weight_decay, noise=noise)

        super(SGLD, self).__init__(params, defaults)

    def step(self):
        """
        Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)

                if group["noise"]:
                    langevin_noise = p.data.new(p.data.size()).normal_(
                        mean=0, std=group["noise"]
                    ) / np.sqrt(group["lr"])

                    p.data.add_(-group["lr"], 0.5 * d_p + langevin_noise)
                else:
                    p.data.add_(-group["lr"], 0.5 * d_p)

        return loss


def to_tuple(x: Union[Tuple[int, ...], int]) -> Tuple[int, ...]:
    if isinstance(x, int):
        return (x,)
    else:
        return x


class Sampler:
    """
    Sample several learning machines using an optimizer.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: DataLoader,
        loss: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
        num_steps: int,
        num_inits: int,
        sample_from: Union[int, Tuple[int, ...]] = -1,
    ):
        """

        Args:
            model: The model (class) to sample from.
            optimizer: The optimizer to use for sampling.
            dataloader: The dataloader to use for sampling.
            num_steps: The number of steps to run the optimizer per initialization.
            num_inits: The number of initializations to sample from.
            sample_from: The index or indices of the parameters to sample from. If -1, sample from the last step of the trajectory.

        """
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.loss = loss
        self.num_steps = num_steps
        self.num_inits = num_inits

        self.sample_from = tuple(
            (x if x > 0 else num_steps + x) for x in to_tuple(sample_from)
        )

        # TODO: With very large models, trying to keep >1 copy of the model in memory may be too much
        self.init_model_state = copy.deepcopy(
            model.state_dict()
        )  # the deepcopy may not be necessary

    def reset_(self):
        """
        Reset the model to its initial state.
        """
        self.model.load_state_dict(self.init_model_state)

    def samples(self, seed=0) -> Generator[torch.nn.Module, None, None]:
        """
        Sample from the optimizer.

        Args:
            seed: The seed to use for sampling.

        """

        # TODO: Make this outer loop parallelizable
        for init in range(self.num_inits):
            self.reset_()

            # TODO: What if the dataloader is already shuffled?
            torch.random.manual_seed(seed + 0.1 * init)

            for step, batch in zip(range(self.num_steps), self.dataloader):
                self.optimizer.zero_grad()
                loss = self.loss(self.model, batch)
                loss.backward()
                self.optimizer.step()

                if step in self.sample_from:
                    yield self.model

    def estimates(self, observable: Callable[[torch.nn.Module], Any], seed=0):
        """
        Sample snapshots of an observable from the optimizer.

        Args:
            observable: The observable to sample from.
        """

        for sample in self.samples(seed=seed):
            yield observable(sample)

    def estimate(
        self,
        observable: Callable[[torch.nn.Module], Any],
        reduction: Reduction = "mean",
        seed=0,
        return_tensor: ReturnTensor = "pt",
    ):
        """
        Estimate some observable over samples drawn from the optimizer.

        Args:
            observable: The observable to sample from.
            reduction: How to reduce the estimates.
            seed: The seed to use for sampling.

        TODO: Allow `return_tensors` as in transformers
        """

        estimates = convert_tensor(self.estimates(observable, seed=seed), return_tensor=return_tensor)
        return reduce_tensor(estimates, reduction=reduction)
