from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from devinterp.optim.sgld import \
    SGLD  # Make sure to import your SGLD optimizer


def test_SGLD_vs_SGD():
    for lr in [1e-1, 1e-2, 1e-3, 1e-4]:
        model1 = nn.Linear(1, 1)
        optimizer_sgd = optim.SGD(model1.parameters(), lr=lr)
        data = torch.tensor([[1.0]]).reshape(-1, 1)
        target = torch.tensor([[2.0]]).reshape(-1, 1)
        
        model2 = deepcopy(model1)
        optimizer_sgld = SGLD(model2.parameters(), lr=2*lr, noise_level=0., elasticity=0., temperature=1.)

        criterion = nn.MSELoss()

        # Using SGD optimizer
        optimizer_sgd.zero_grad()
        output = model1(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_sgd.step()

        # Using SGLD optimizer with noise=0, elasticity=0
        optimizer_sgld.zero_grad()
        output = model2(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer_sgld.step()

        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert torch.allclose(p1, p2, atol=1e-7), f"Parameters differ: {p1} vs {p2}"
