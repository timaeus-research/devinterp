import numpy as np
import torch


class SGNHT(torch.optim.Optimizer):
    r"""
    Implement the Stochastic Gradient Nose Hoover Thermostat (SGNHT) Optimizer.
    This optimizer blends SGD with an adaptive thermostat variable to control the magnitude of the injected noise,
    maintaining the kinetic energy of the system.

    It follows Ding et al.'s (2014) implementation.

    The equations for the update are as follows:

    $$\Delta w_t = \epsilon\left(\frac{\beta n}{m} \sum_{i=1}^m \nabla \log p\left(y_{l_i} \mid x_{l_i}, w_t\right) - \xi_t w_t \right) + \sqrt{2A} N(0, \epsilon)$$
    $$\Delta\xi_{t} = \epsilon \left( \frac{1}{n} \|w_t\|^2 - 1 \right)$$

    where $w_t$ is the weight at time $t$, $\epsilon$ is the learning rate,
    $(\beta n)$ is the inverse temperature (we're in the tempered Bayes paradigm),
    $n$ is the number of samples, $m$ is the batch size,
    $\xi_t$ is the thermostat variable at time $t$, $A$ is the diffusion factor,
    and $N(0, A)$ represents Gaussian noise with mean 0 and variance $A$.
    
    Note:
        - :python:`diffusion_factor` is unique to this class, and functions as a way to allow for random parameter changes while keeping them from blowing up by guiding parameters back to a slowly-changing thermostat value using a friction term.
        - This class does not have an explicit localization term like :func:`~devinterp.optim.sgld.SGLD` does. If you want to constrain your sampling, use :python:`bounding_box_size`
        - Although this class is a subclass of :python:`torch.optim.Optimizer`, this is a bit of a misnomer in this case. It's not used for optimizing in LLC estimation, but rather for sampling from the posterior distribution around a point. 
    

    :param params: Iterable of parameters to optimize or dicts defining parameter groups. Either :python:`model.parameters()` or something more fancy, just like other :python:`torch.optim.Optimizer` classes.
    :type params: Iterable
    :param lr: Learning rate $\epsilon$. Default is 0.01
    :type lr: float, optional
    :param diffusion_factor: The diffusion factor $A$ of the thermostat. Default is 0.01
    :type diffusion_factor: float, optional
    :param bounding_box_size: the size of the bounding box enclosing our trajectory. Default is None
    :type bounding_box_size: float, optional
    :param num_samples: Number of samples $n$ to average over. Should be equal to the size of your dataset, unless you know what you're doing. Default is 1
    :type num_samples: int, optional
    
    :raises Warning: if :python:`num_samples` is set to 1
    """
    def __init__(
        self,
        params,
        lr=0.01,
        diffusion_factor=0.01,
        bounding_box_size=None,
        num_samples=1,
    ):
        defaults = dict(
            lr=lr,
            diffusion_factor=diffusion_factor,
            bounding_box_size=bounding_box_size,
            num_samples=num_samples,
        )
        super(SGNHT, self).__init__(params, defaults)

        # Initialize momentum/thermostat for each parameter
        for group in self.param_groups:
            # Default value of thermostat is the diffusion factor
            group["thermostat"] = torch.tensor(diffusion_factor)
            group["temperature"] = np.log(group["num_samples"])
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum"] = np.sqrt(lr) * torch.randn_like(p.data)

                if group["bounding_box_size"] != 0:
                    param_state["initial_param"] = p.data.clone().detach()

    def step(self, closure=None):
        with torch.no_grad():
            for group in self.param_groups:
                group_energy_sum = 0.0
                group_energy_size = 0

                for p in group["params"]:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    momentum = param_state["momentum"]

                    # Gradient term
                    dw = p.grad.data * (group["num_samples"] / group["temperature"])

                    momentum.sub_(group["lr"] * dw)

                    # Friction term
                    momentum.sub_(group["thermostat"] * momentum)

                    # Add Gaussian noise to momentum
                    noise = torch.normal(
                        mean=0.0, std=1.0, size=momentum.size(), device=momentum.device
                    )
                    momentum.add_(
                        noise * ((group["lr"] * 2 * group["diffusion_factor"]) ** 0.5)
                    )

                    # Update position
                    p.data.add_(momentum)

                    # Accumulate the energy sums to compute the average later
                    # This gets the sum of the squares of momentum (across all chains)
                    group_energy_sum += torch.einsum("...,...->", momentum, momentum)
                    group_energy_size += momentum.numel()

                    # Rebound if exceeded bounding box size
                    if group["bounding_box_size"]:
                        reflection_coefs = (
                            (
                                abs(p.data - param_state["initial_param"])
                                < group["bounding_box_size"]
                            )
                            * 2
                        ) - 1
                        torch.clamp_(
                            p.data,
                            min=param_state["initial_param"]
                            - group["bounding_box_size"],
                            max=param_state["initial_param"]
                            + group["bounding_box_size"],
                        )
                        momentum.mul_(reflection_coefs)

                # Update thermostat based on average kinetic energy
                d_thermostat = (group_energy_sum / group_energy_size) - group["lr"]
                group["thermostat"].add_(d_thermostat.to(group["thermostat"].device))
