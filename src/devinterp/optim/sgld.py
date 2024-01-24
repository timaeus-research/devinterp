from typing import Literal, Union
import warnings

import numpy as np
import torch


class SGLD(torch.optim.Optimizer):
    r"""
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer blends Stochastic Gradient Descent (SGD) with Langevin Dynamics,
    introducing Gaussian noise to the gradient updates. This makes it sample weights from the posterior distribution, instead of optimizing weights.

    This implementation follows Lau et al.'s (2023) implementation, which is a modification of
    Welling and Teh (2011) that omits the learning rate schedule and introduces
    an elasticity term that pulls the weights towards their initial values.

    The equation for the update is as follows:

    $$\Delta w_t = \frac{\epsilon}{2}\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right)+\gamma\left(w_0-w_t\right) - \lambda w_t\right) + N(0, \epsilon\sigma^2)$$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of training samples, $m$ is the batch size, $\gamma$ is
    the elasticity strength, $\lambda$ is the weight decay strength,
    and $\sigma$ is the noise term.

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=0.1, temperature=torch.log(n)/n)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    Note:
        - :code:`elasticity` is unique to this class and serves to guide the weights towards their original values. This is useful for estimating quantities over the local posterior.
        - :code:`noise_level` is not intended to be changed, except when testing! Doing so will raise a warning.
        - Although this class is a subclass of :code:`torch.optim.Optimizer`, this is a bit of a misnomer in this case. It's not used for optimizing in LLC estimation, but rather for sampling from the posterior distribution around a point. 
        - Hyperparameter optimization is more of an art than a science. Check out the example notebooks LINK TODO
    :param params: Iterable of parameters to optimize or dicts defining parameter groups. Either :code:`model.parameters()` or something more fancy, just like other :code:`torch.optim.Optimizer` classes.
    :type params: Iterable
    :param lr: Learning rate. Defaults to 0.01
    :type lr: float, optional
    :param noise_level: Amount of Gaussian noise introduced into gradient updates. Don't change this unless you know very well what you're doing! Defaults to 1
    :type noise_level: float, optional
    :param weight_decay: L2 regularization term, applied as weight decay. Defaults to 0
    :type weight_decay: float, optional
    :param elasticity: Strength of the force pulling weights back to their initial values. Defaults to 0
    :type elasticity: float, optional
    :param temperature: Temperature, either as a float or 'adaptive'(:code:`=np.log(num_samples)`). Defaults to 'adaptive'
    :type temperature: float | 'adaptive', optional
    :param bounding_box_size: the size of the bounding box enclosing our trajectory. Defaults to None
    :type bounding_box_size: float, optional
    :param num_samples: Number of samples to average over, $$n$$ from the above formula.. Should be equal to the size of your dataset, unless you know what you're doing. Defaults to 1
    :type num_samples: int, optional
    :param save_noise: whether to store the per-parameter noise during optimization . Defaults to False
    :type save_noise: bool, optional
    
    :raises Warning: if :code:`noise_level` is set to anything other than 1
    :raises Warning: if :code:`num_samples` is set to 1
    """

    def __init__(
        self,
        params,
        lr=0.01,
        noise_level=1.0,
        weight_decay=0.0,
        elasticity=0.0,
        temperature: Union[Literal["adaptive"], float] = "adaptive",
        bounding_box_size=None,
        num_samples=1,
        save_noise=False,
    ):
        if noise_level != 1.0:
            warnings.warn(
                "Warning: noise_level in SGLD is unequal to zero, are you intending to use SGD?"
            )
        if num_samples == 1:
            warnings.warn(
                "Warning: num_samples is set to 1, make sure you know what you're doing! If not, just use num_samples = len(dataset)"
            )
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            elasticity=elasticity,
            temperature=temperature,
            bounding_box_size=bounding_box_size,
            num_samples=num_samples,
        )
        super(SGLD, self).__init__(params, defaults)
        self.save_noise = save_noise
        self.noise = None

        # Save the initial parameters if the elasticity term is set
        for group in self.param_groups:
            if group["elasticity"] != 0 or group["bounding_box_size"] != 0:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["initial_param"] = p.data.clone().detach()
            if group["temperature"] == "adaptive":  # TODO: Better name
                group["temperature"] = np.log(group["num_samples"])

    def step(self, closure=None):
        self.noise = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                dw = p.grad.data * group["num_samples"] / group["temperature"]

                if group["weight_decay"] != 0:
                    dw.add_(p.data, alpha=group["weight_decay"])

                if group["elasticity"] != 0:
                    initial_param = self.state[p]["initial_param"]
                    dw.add_((p.data - initial_param), alpha=group["elasticity"])

                p.data.add_(dw, alpha=-0.5 * group["lr"])

                # Add Gaussian noise
                noise = torch.normal(
                    mean=0.0, std=group["noise_level"], size=dw.size(), device=dw.device
                )
                if self.save_noise:
                    self.noise.append(noise)
                p.data.add_(noise, alpha=group["lr"] ** 0.5)

                # Rebound if exceeded bounding box size
                if group["bounding_box_size"]:
                    torch.clamp_(
                        p.data,
                        min=param_state["initial_param"] - group["bounding_box_size"],
                        max=param_state["initial_param"] + group["bounding_box_size"],
                    )
