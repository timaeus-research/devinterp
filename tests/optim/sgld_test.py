from copy import deepcopy
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from devinterp.optim.sgld import SGLD
from devinterp.utils import optimal_temperature


@pytest.mark.parametrize("lr", [1e-1, 1e-2, 1e-3, 1e-4])
def test_SGLD_vs_SGD(lr):
    model1 = nn.Linear(1, 1)
    optimizer_sgd = optim.SGD(model1.parameters(), lr=lr)
    data = torch.tensor([[1.0]]).reshape(-1, 1)
    target = torch.tensor([[2.0]]).reshape(-1, 1)

    model2 = deepcopy(model1)
    optimizer_sgld = SGLD(
        model2.parameters(),
        lr=2 * lr,
        noise_level=0.0,
        localization=0.0,
        temperature=1.0,
    )

    criterion = nn.MSELoss()

    # Using SGD optimizer
    optimizer_sgd.zero_grad()
    output = model1(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer_sgd.step()

    # Using SGLD optimizer with noise=0, localization=0
    optimizer_sgld.zero_grad()
    output = model2(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer_sgld.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7), f"Parameters differ: {p1} vs {p2}"


def test_seeding():
    lr = 0.1
    data = torch.tensor([[1.0]]).reshape(-1, 1)
    target = torch.tensor([[2.0]]).reshape(-1, 1)
    criterion = nn.MSELoss()

    model1 = nn.Linear(1, 1)
    model2 = deepcopy(model1)
    optimizer_sgld_1 = SGLD(
        model1.parameters(), lr=lr, noise_level=1.0, temperature=1.0
    )
    optimizer_sgld_2 = SGLD(
        model2.parameters(), lr=lr, noise_level=1.0, temperature=1.0
    )
    
    torch.manual_seed(42)
    optimizer_sgld_1.zero_grad()
    output = model1(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer_sgld_1.step()

    torch.manual_seed(42)
    optimizer_sgld_2.zero_grad()
    output = model2(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer_sgld_2.step()

    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7), f"Parameters differ: {p1} vs {p2}"

