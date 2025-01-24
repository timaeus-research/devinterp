from collections import defaultdict
from copy import deepcopy

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from devinterp.optim import SGLD, SGMCMC


def serialize_model_state(model):
    """Helper to convert model parameters to serializable format"""
    return {
        name: param.detach().cpu().numpy() for name, param in model.named_parameters()
    }


@pytest.mark.parametrize("lr", [1e-1, 1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("sampler_cls", [SGLD, SGMCMC.sgld])
def test_SGLD_vs_SGD(lr, sampler_cls, snapshot):
    torch.manual_seed(42)

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
        nbeta=1.0,
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

    # Add snapshot assertion
    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
    }
    assert state == snapshot(name=f"test_SGLD_vs_SGD_{lr}")


@pytest.mark.parametrize("sampler_cls", [SGLD, SGMCMC.sgld])
def test_SGLD_metrics_tracking(snapshot, sampler_cls):
    torch.manual_seed(42)

    # Setup a simple model and data
    model = nn.Linear(1, 1)
    data = torch.tensor([[1.0]]).reshape(-1, 1)
    target = torch.tensor([[2.0]]).reshape(-1, 1)
    criterion = nn.MSELoss()

    # Test all available metrics
    metrics = [
        "noise_norm",
        "grad_norm",
        "weight_norm",
        "distance",
        "noise",
        "dws",
        "localization_loss",
    ]
    optimizer = sampler_cls(
        model.parameters(),
        lr=0.1,
        noise_level=1.0,
        localization=0.1,
        nbeta=1.0,
        metrics=metrics,
    )

    # Perform optimization step
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Check scalar metrics
    assert isinstance(
        optimizer.noise_norm, torch.Tensor
    ), "noise_norm should be a tensor"
    assert isinstance(optimizer.grad_norm, torch.Tensor), "grad_norm should be a tensor"
    assert isinstance(
        optimizer.weight_norm, torch.Tensor
    ), "weight_norm should be a tensor"
    assert isinstance(optimizer.distance, torch.Tensor), "distance should be a tensor"

    # Check list-based metrics
    # assert isinstance(optimizer.noise, defaultdict), "noise should be a defaultdict"
    assert isinstance(optimizer.dws, list), "dws should be a list"
    assert len(optimizer.dws) > 0, "dws should not be empty after step"

    # Add snapshot assertion
    state = {
        "noise_norm": optimizer.noise_norm.detach().cpu().numpy(),
        "grad_norm": optimizer.grad_norm.detach().cpu().numpy(),
        "weight_norm": optimizer.weight_norm.detach().cpu().numpy(),
        "distance": optimizer.distance.detach().cpu().numpy(),
    }
    assert state == snapshot(name="test_SGLD_metrics_tracking")


def test_SGLD_invalid_metrics():
    torch.manual_seed(42)

    model = nn.Linear(1, 1)

    # Test invalid metric name
    with pytest.raises(ValueError, match="Invalid metrics"):
        SGLD(model.parameters(), metrics=["invalid_metric"])
    with pytest.raises(ValueError, match="Invalid metrics"):
        SGMCMC.sgld(model.parameters(), metrics=["invalid_metric"])


@pytest.mark.parametrize("sampler_cls", [SGLD, SGMCMC.sgld])
def test_SGLD_selective_metrics(snapshot, sampler_cls):
    torch.manual_seed(42)

    model = nn.Linear(1, 1)
    data = torch.tensor([[1.0]]).reshape(-1, 1)
    target = torch.tensor([[2.0]]).reshape(-1, 1)
    criterion = nn.MSELoss()

    # Only track grad_norm and weight_norm
    optimizer = sampler_cls(
        model.parameters(), lr=0.1, metrics=["grad_norm", "weight_norm"]
    )

    # Perform optimization step
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    # Check tracked metrics exist
    assert optimizer.grad_norm is not None
    assert optimizer.weight_norm is not None

    # Check untracked metrics raise AttributeError
    with pytest.raises(AttributeError):
        _ = optimizer.noise_norm

    with pytest.raises(AttributeError):
        _ = optimizer.distance

    # Add snapshot assertion
    state = {
        "grad_norm": optimizer.grad_norm.detach().cpu().numpy(),
        "weight_norm": optimizer.weight_norm.detach().cpu().numpy(),
    }
    assert state == snapshot(name="test_SGLD_selective_metrics")
