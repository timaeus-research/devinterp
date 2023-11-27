import numpy as np
import torch
import torch.nn as nn

from devinterp.optim.sgld import SGLD
from devinterp.slt.callback import SamplerCallback

class WeightNorm(SamplerCallback):
    def __init__(self, num_chains: int, num_draws: int, p_norm: int = 2, device='cpu'):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.weight_norms = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.p_norm = p_norm
        self.device = device
    
    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)
    
    def update(self, chain: int, draw: int, model: nn.Module):
        total_norm = torch.tensor(0.)
        for param in model.parameters():
            total_norm += torch.square(torch.linalg.vector_norm(param, ord=2))
        total_norm = torch.pow(total_norm, 1/self.p_norm)
        self.weight_norms[chain, draw] = total_norm

    def sample(self):
        return {
            "weight_norm/trace": self.weight_norms.cpu().numpy(),
        }


class GradientNorm(SamplerCallback):
    def __init__(self, num_chains: int, num_draws: int, p_norm: int = 2, device='cpu'):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.gradient_norms = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.p_norm = p_norm
        self.device = device
    
    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)
    
    def update(self, chain: int, draw: int, model: nn.Module):
        total_norm = torch.tensor(0.)
        for param in model.parameters():
            total_norm += torch.square(torch.linalg.vector_norm(param.grad, ord=2))
        total_norm = torch.pow(total_norm, 1/self.p_norm)
        self.gradient_norms[chain, draw] = total_norm

    def sample(self):
        return {
            "gradient_norm/trace": self.gradient_norms.cpu().numpy(),
        }
