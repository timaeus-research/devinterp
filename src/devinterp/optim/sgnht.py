import numpy as np
import torch


class SGNHT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        diffusion_factor=0.01,
        bounding_box_size=None,
        num_samples=1,
    ):
        """
        Initialize the SGNHT Optimizer.

        :param params: Iterable of parameters to optimize or dicts defining parameter groups
        :param lr: Learning rate (required)
        :param diffusion_factor: The diffusion factor of the thermostat (default: 0.01)
        :param bounding_box_size: the size of the bounding box enclosing our trajectory The diffusion factor (default: None)
        :param num_samples: Number of samples to average over (default: 1)
        """
        defaults = dict(
            lr=lr,
            diffusion_factor=0.01,
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
        """
        Perform one step of SGNHT optimization.
        """
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
                    momentum.add_(noise * ((group["lr"] * 2 * group["diffusion_factor"]) ** 0.5))

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
                            min=param_state["initial_param"] - group["bounding_box_size"],
                            max=param_state["initial_param"] + group["bounding_box_size"],
                        )
                        momentum.mul_(reflection_coefs)

                # Update thermostat based on average kinetic energy
                d_thermostat = (group_energy_sum / group_energy_size) - group["lr"]
                group["thermostat"].add_(d_thermostat.to(group["thermostat"].device))
