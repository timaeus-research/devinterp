import warnings
from typing import Callable, Union, Optional

import torch


class SGLD(torch.optim.Optimizer):
    r"""
    Implements Stochastic Gradient Langevin Dynamics (SGLD) optimizer.

    This optimizer blends Stochastic Gradient Descent (SGD) with Langevin Dynamics,
    introducing Gaussian noise to the gradient updates. This makes it sample weights from the posterior distribution, instead of optimizing weights.

    This implementation follows Lau et al.'s (2023) implementation, which is a modification of
    Welling and Teh (2011) that omits the learning rate schedule and introduces
    an localization term that pulls the weights towards their initial values.

    The equation for the update is as follows:

    $$\Delta w_t = \frac{\epsilon}{2}\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right)+\gamma\left(w_0-w_t\right) - \lambda w_t\right) + N(0, \epsilon\sigma^2)$$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of training samples, $m$ is the batch size, $\gamma$ is
    the localization strength, $\lambda$ is the weight decay strength,
    and $\sigma$ is the noise term.

    Example:
        >>> optimizer = SGLD(model.parameters(), lr=0.1, nbeta=utils.optimal_nbeta(dataloader))

        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. |colab6| image:: https://colab.research.google.com/assets/colab-badge.svg
        :target: https://colab.research.google.com/github/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb

    Note:
        - :python:`localization` is unique to this class and serves to guide the weights towards their original values. This is useful for estimating quantities over the local posterior.
        - :python:`noise_level` is not intended to be changed, except when testing! Doing so will raise a warning.
        - Although this class is a subclass of :python:`torch.optim.Optimizer`, this is a bit of a misnomer in this case. It's not used for optimizing in LLC estimation, but rather for sampling from the posterior distribution around a point.
        - Hyperparameter optimization is more of an art than a science. Check out `the calibration notebook <https://www.github.com/timaeus-research/devinterp/blob/main/examples/sgld_calibration.ipynb>`_ |colab6| for how to go about it in a simple case.
    :param params: Iterable of parameters to optimize or dicts defining parameter groups. Either :python:`model.parameters()` or something more fancy, just like other :python:`torch.optim.Optimizer` classes.
    :type params: Iterable
    :param lr: Learning rate $\epsilon$. Default is 0.01
    :type lr: float, optional
    :param noise_level: Amount of Gaussian noise $\sigma$ introduced into gradient updates. Don't change this unless you know very well what you're doing! Default is 1
    :type noise_level: float, optional
    :param weight_decay: L2 regularization term $\lambda$, applied as weight decay. Default is 0
    :type weight_decay: float, optional
    :param localization: Strength of the force $\gamma$ pulling weights back to their initial values. Default is 0
    :type localization: float, optional
    :param nbeta: Inverse reparameterized temperature (otherwise known as n*beta or ~beta), float (default: 1., set to utils.optimal_nbeta(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type nbeta: int, optional
    :param bounding_box_size: the size of the bounding box enclosing our trajectory in parameter space. Default is None, in which case no bounding box is used.
    :type bounding_box_size: float, optional
    :param save_noise: Whether to store the per-parameter noise during optimization. Default is False
    :type save_noise: bool, optional
    :param save_mala_vars: Whether to store variables for calculating Metropolis-Adjusted Langevin Algorithm (MALA) metrics.
    :type save_mala_vars: bool, optional
    :param optimize_over: A boolean tensor of the same shape as the parameters. Used to implement weight restrictions.
    Think of it as a boolean mask that restricts the set of parameters that can be updated. Default is None (no restrictions).
    :type optimize_over: torch.Tensor, optional
    :param noise_norm: Boolean flag to track the norm of the noise. Default is False
    :type noise_norm: bool, optional
    :param grad_norm: Boolean flag to track the norm of the gradient. Default is False
    :type grad_norm: bool, optional
    :param weight_norm: Boolean flag to track the norm of the weights. Default is False
    :type weight_norm: bool, optional
    :param distance: Boolean flag to track the distance between the current weights and the initial weights. Default is False
    :type distance: bool, optional


    :raises Warning: if :python:`noise_level` is set to anything other than 1
    :raises Warning: if :python:`nbeta` is set to 1
    """

    def __init__(
        self,
        params,
        lr=0.01,
        noise_level=1.0,
        weight_decay=0.0,
        localization=0.0,
        nbeta: Union[Callable, float] = 1.0,
        bounding_box_size=None,
        save_noise=False,
        save_mala_vars=False,
        optimize_over=None,
        noise_norm=False,
        grad_norm=False,
        weight_norm=False,
        distance=False,
        temperature: Optional[float] = None,
    ):
        if temperature is not None:
            nbeta = temperature
            warnings.warn("Temperature is deprecated. Please use nbeta in your yaml file instead.")

        if noise_level != 1.0:
            warnings.warn(
                "Warning: noise_level in SGLD is unequal to one, this removes SGLD posterior sampling guarantees."
            )
        if nbeta == 1.0:
            warnings.warn(
                "Warning: nbeta set to 1, LLC estimates will be off unless you know what you're doing. Use utils.optimal_nbeta(dataloader) instead"
            )
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            localization=localization,
            nbeta=nbeta,
            bounding_box_size=bounding_box_size,
            optimize_over=optimize_over,
            noise_norm=noise_norm,
            grad_norm=grad_norm,
            weight_norm=weight_norm,
            distance=distance,
        )

        # In torch.optim.Optimizer, the parameters are stored in a list of dictionaries.
        # defaults holds the default values for the optimizer parameters.
        super(SGLD, self).__init__(params, defaults)
        self.save_noise = save_noise
        self.save_mala_vars = save_mala_vars
        self.noise = None

        # Save the initial parameters if the localization term is set
        for group in self.param_groups:

            group["num_el"] = 0

            if group["localization"] != 0 or group["bounding_box_size"] != 0:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["initial_param"] = p.data.clone().detach()
                    group["num_el"] += p.numel()

            for hp in ["noise_norm", "grad_norm", "distance", "weight_norm"]:
                if group[hp] is not False:
                    group[hp] = torch.tensor(0.0).to(p.device)

    def step(self, noise_generator: Optional[torch.Generator] = None):
        """
        Perform a single SGLD optimization step.
        """
        if self.save_noise:
            self.noise = []

        if self.save_mala_vars:
            self.dws = []
            self.localization_loss = 0.0

        with torch.no_grad():
            for group in self.param_groups:
                for hp in ["noise_norm", "grad_norm", "distance", "weight_norm"]:
                    # Zero iteration-level metrics that haven't been disabled.
                    if group[hp] is not False:
                        group[hp] *= 0.0

                for p in group["params"]:
                    param_state = self.state[p]

                    # Gradients are None if the parameter is not trainable
                    # We'll denote the gradient of the loss with respect to this param group (p) as dw
                    if p.grad is None:
                        dw = torch.zeros_like(p.data)
                    else:
                        dw = p.grad.data * group["nbeta"]

                    # Weight decay
                    if group["weight_decay"] != 0:
                        dw.add_(
                            p.data, alpha=group["weight_decay"]
                        )  # inplace addition. Effectively, dw = dw + p.data * group["weight_decay"]

                    # Here, group["localization"] is the localization strength $\gamma$ (a single float). If it's 0, we don't do anything.
                    initial_param = self.state[p]["initial_param"]
                    initial_param_distance = p.data - initial_param
                    if group["localization"] != 0:
                        dw.add_(initial_param_distance, alpha=group["localization"])

                    if self.save_mala_vars:
                        # TODO: Initial param distance is used as a m
                        if group["optimize_over"] is not None:
                            initial_param_distance = (
                                initial_param_distance * group["optimize_over"]
                            )
                        # localization_loss = (p.data - initial_param)^2 * group["optimize_over"]^2 * group["localization"] / 2
                        #                                                           ^ boolean
                        distance = (initial_param_distance.detach() ** 2).sum() * group[
                            "localization"
                        ]
                        self.localization_loss += distance / 2
                        self.dws.append(dw.clone())

                        if group["distance"] is not False:
                            group["distance"] += (distance**0.5) * group["lr"]

                    # Add Gaussian noise
                    noise = torch.normal(
                        mean=0.0,
                        std=group["noise_level"],
                        size=dw.size(),
                        device=dw.device,
                        generator=noise_generator,
                    )
                    if self.save_noise:
                        # Noise saved here is the unscaled noise.
                        self.noise.append(noise)

                    if group["optimize_over"] is not None:
                        # Restrict the noise and gradient to the subset of parameters we're optimizing over.
                        dw = dw * group["optimize_over"]
                        noise = noise * group["optimize_over"]

                    # Update parameters
                    p.data.add_(dw, alpha=-0.5 * group["lr"])
                    p.data.add_(
                        noise, alpha=group["lr"] ** 0.5
                    )  # Scale noise by sqrt(lr)

                    # Track the size of the changes & relative contributions
                    if group["grad_norm"] is not False and p.grad is not None:
                        group["grad_norm"] += (
                            ((p.grad.data * group["nbeta"] * 0.5 * group["lr"]) ** 2)
                            .sum()
                            .detach()
                        )

                    if group["weight_norm"] is not False:
                        group["weight_norm"] += (p.data**2).sum().detach()

                    if group["noise_norm"] is not False:
                        group["noise_norm"] += (noise**2).sum().detach()

                    if group["distance"] is not False and not self.save_mala_vars:
                        group["distance"] += (
                            ((initial_param_distance * group["localization"]) ** 2)
                            .sum()
                            .detach()
                        )

                    # Rebound if exceeded bounding box size
                    if group["bounding_box_size"]:
                        torch.clamp_(
                            p.data,
                            min=param_state["initial_param"]
                            - group["bounding_box_size"],
                            max=param_state["initial_param"]
                            + group["bounding_box_size"],
                        )

                for hp in ["noise_norm", "grad_norm", "distance", "weight_norm"]:
                    if group[hp] is not False:
                        group[hp] = (group[hp] ** 0.5).detach()

    def __getattr__(self, name):
        """
        Return iteration-level metrics if requested.
        """
        if name in ["noise_norm", "grad_norm", "distance", "weight_norm", "lr"]:
            return next(group[name] for group in self.param_groups)

        raise AttributeError(f"'SGLD' object has no attribute '{name}'")
