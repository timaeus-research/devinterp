import numpy as np
import torch


class BADODAB(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, diffusion_factor=0.01, bounding_box_size=None, num_samples=1, batch_size=None):
        """
        Initialize the BADODAB sampler (Leimkuhler & Shang 2015). This is just a better numerical integration scheme for SGNHT, in theory.
        But as of 09/29/2023 it hasn't proven better at avoiding numerical instability / oscillations, which is the primary problem this
        was implemented to address.

        :param params: Iterable of parameters to optimize or dicts defining parameter groups
        :param lr: Learning rate (required)
        :param diffusion factor: The diffusion factor (default: 0.01)
        :param num_samples: Number of samples to average over (default: 1)
        """
        defaults = dict(lr=lr, diffusion_factor=diffusion_factor, bounding_box_size=bounding_box_size, num_samples=num_samples, batch_size=batch_size)
        super(BADODAB, self).__init__(params, defaults)

        # Initialize momentum/thermostat for each parameter
        for group in self.param_groups:
            # Default value of thermostat is the diffusion factor
            group['thermostat'] = torch.tensor(diffusion_factor)
            for p in group['params']:
                param_state = self.state[p]
                param_state['momentum'] = np.sqrt(lr) * torch.randn_like(p.data)

    def step(self, closure):
        """
        Perform one step of BADODAB optimization.
        """
        # Compute gradients the first time
        closure()
      
        for group in self.param_groups:
            with torch.no_grad():
                # First loop: update momentum (gradient term only) and position
                group_energy_sum = 0.0
                group_energy_size = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    momentum = param_state['momentum']

                    # First momentum update (gradient only)
                    dw = (group["num_samples"] / group["batch_size"]) * p.grad.data
                    momentum.sub_(group['lr'] * 0.5 * dw)

                    # First position update
                    p.data.add_(group['lr'] * 0.5 * momentum)

                    # Accumulate the energy sums to compute the average later
                    # This gets the sum of the squares of momentum (across all chains)
                    group_energy_sum += torch.einsum('...,...->', momentum, momentum)
                    group_energy_size += momentum.numel()

                # Update thermostat based on average kinetic energy
                d_thermostat = (group_energy_sum / group_energy_size) - 1
                group['thermostat'].add_(group['lr'] * 0.5 * d_thermostat)

                # Constants used to update the momentum
                damping_coef = torch.exp(-group['thermostat'] * group['lr'])
                sqrt_term = (1 - torch.exp(-2 * group['thermostat'] * group['lr'])) / (2 * group['thermostat'])
                noise_coef = group['diffusion_factor'] * torch.sqrt(sqrt_term)

                # Second loop: update momentum (noise/friction terms) and position
                group_energy_sum = 0.0
                group_energy_size = 0
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    momentum = param_state['momentum']

                    # Apply momentum damping
                    momentum.mul_(damping_coef)

                    # Add momentum noise
                    noise = torch.normal(mean=0., std=1.0, size=momentum.size(), device=momentum.device)
                    momentum.add_(noise_coef * noise)

                    # Last position update
                    p.data.add_(group['lr'] * 0.5 * momentum)

                    # Accumulate the energy sums to compute the average later
                    # This gets the sum of the squares of momentum (across all chains)
                    group_energy_sum += torch.einsum('...,...->', momentum, momentum)
                    group_energy_size += momentum.numel()

                # Update thermostat again
                d_thermostat = (group_energy_sum / group_energy_size) - 1
                group['thermostat'].add_(group['lr'] * 0.5 * d_thermostat)

            # Recompute gradients for final momentum update
            closure()

            with torch.no_grad():
                # Last loop: update momentum (gradient term only)
                for p in group['params']:
                    if p.grad is None:
                        continue

                    param_state = self.state[p]
                    momentum = param_state['momentum']

                    # Last momentum update (gradient only)
                    dw = (group["num_samples"] / group["batch_size"]) * p.grad.data
                    momentum.sub_(group['lr'] * 0.5 * dw)


