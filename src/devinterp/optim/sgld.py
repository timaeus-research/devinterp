from typing import Union, Callable
import warnings

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
        >>> optimizer = SGLD(model.parameters(), lr=0.1, temperature=utils.optimal_temperature(dataloader))
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
    :param bounding_box_size: the size of the bounding box enclosing our trajectory. Default is None
    :type bounding_box_size: float, optional
    :param temperature: Temperature, float (default: 1., set by sample() to utils.optimal_temperature(dataloader)=len(batch_size)/np.log(len(batch_size)))
    :type temperature: int, optional
    :param save_noise: Whether to store the per-parameter noise during optimization. Default is False
    :type save_noise: bool, optional
    
    :raises Warning: if :python:`noise_level` is set to anything other than 1
    :raises Warning: if :python:`temperature` is set to 1
    """

    def __init__(
        self,
        params,
        lr=0.01,
        noise_level=1.0,
        weight_decay=0.0,
        localization=0.0,
        temperature: Union[Callable, float] = 1.0,
        bounding_box_size=None,
        save_noise=False,
        save_mala_vars=False,
    ):
        if noise_level != 1.0:
            warnings.warn(
                "Warning: noise_level in SGLD is unequal to one, this removes SGLD posterior sampling guarantees."
            )
        if temperature == 1.0:
            warnings.warn(
                "Warning: temperature set to 1, LLC estimates will be off unless you know what you're doing. Use utils.optimal_temperature(dataloader) instead"

            )
        defaults = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            localization=localization,
            temperature=temperature,
            bounding_box_size=bounding_box_size,
        )
        super(SGLD, self).__init__(params, defaults)
        self.save_noise = save_noise
        self.save_mala_vars = save_mala_vars
        self.noise = None

        # Save the initial parameters if the localization term is set
        for group in self.param_groups:
            if group["localization"] != 0 or group["bounding_box_size"] != 0:
                for p in group["params"]:
                    param_state = self.state[p]
                    param_state["initial_param"] = p.data.clone().detach()

    def step(self, closure=None):
        if self.save_noise:
            self.noise = []
        if self.save_mala_vars:
            self.dws = []
            self.localization_loss = 0.0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                param_state = self.state[p]
                dw = p.grad.data * group["temperature"]

                if group["weight_decay"] != 0:
                    dw.add_(p.data, alpha=group["weight_decay"])
                if group["localization"] != 0:
                    initial_param = self.state[p]["initial_param"]
                    initial_param_distance = p.data - initial_param
                    dw.add_(initial_param_distance, alpha=group["localization"])
                    if self.save_mala_vars:
                        self.localization_loss += (
                            torch.sum(
                                torch.pow(initial_param_distance.clone().detach(), 2)
                            )
                            * group["localization"]
                            / 2
                        ).item()
                        self.dws.append(dw.clone().detach())

                # Add Gaussian noise
                noise = torch.normal(
                    mean=0.0, std=group["noise_level"], size=dw.size(), device=dw.device
                )
                if self.save_noise:
                    self.noise.append(noise)

                if group.get("optimize_over") is not None:
                    dw = dw * group["optimize_over"]
                    noise = noise * group["optimize_over"]
                p.data.add_(dw, alpha=-0.5 * group["lr"])
                p.data.add_(noise, alpha=group["lr"] ** 0.5)
                # Rebound if exceeded bounding box size
                if group["bounding_box_size"]:
                    torch.clamp_(
                        p.data,
                        min=param_state["initial_param"] - group["bounding_box_size"],
                        max=param_state["initial_param"] + group["bounding_box_size"],
                    )
