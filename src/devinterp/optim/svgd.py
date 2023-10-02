import warnings

import numpy as np
import torch


class RBF(torch.nn.Module):
    def __init__(self, sigma=None):
        super(RBF, self).__init__()

        self.sigma = sigma

    def forward(self, X, Y):
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma**2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


# WARNING: this class does not currently estimate the RLCT correctly
class SVGD(torch.optim.Optimizer):
    def __init__(self, params, K, lr=1e-3):
        defaults = dict(lr=lr)
        super(SVGD, self).__init__(params, defaults)
        self.K = K  # Kernel function
        warnings.warn("This class is currently experimenntal and does not estimate RLCT correctly.")

    def step(self):
        for group in self.param_groups:
            for X in group["params"]:  # Here, X are the particles
                with torch.no_grad():
                    if X.grad is None:
                        continue

                    score_func = X.grad.clone()

                # Calculate the kernel matrix and its gradient
                K_XX = self.K(X, X)
                grad_K = -torch.autograd.grad(K_XX.sum(), X)[0]

                with torch.no_grad():
                    # Compute phi
                    phi = (K_XX.matmul(score_func) + grad_K) / X.size(0)

                    # Perform the update
                    X.sub_(phi, alpha=-group["lr"])
