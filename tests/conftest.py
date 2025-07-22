import json

import numpy as np
import pytest
import torch
import torch.nn as nn
from devinterp.slt.llc import LLCEstimator
from devinterp.slt.sampler import sample as _sample

# --- LLCEstimator runner fixture ---
from devinterp.utils import default_nbeta, evaluate_mse, get_init_loss_multi_batch
from torch.utils.data import DataLoader, TensorDataset


# --- Toy model fixtures ---
@pytest.fixture
def Polynomial():
    class _Poly(nn.Module):
        def __init__(self, powers=(1, 1)):
            super().__init__()
            self.powers = torch.tensor(powers, dtype=torch.float32)
            self.weights = nn.Parameter(torch.zeros_like(self.powers))

        def forward(self, x):
            return x * torch.prod(self.weights**self.powers)

    return _Poly


@pytest.fixture
def LinePlusDot():
    class _LPD(nn.Module):
        def __init__(self, dim=2):
            super().__init__()
            self.weights = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

        def forward(self, x):
            return (
                x
                * (self.weights[0] - 1)
                * (self.weights.pow(2).sum(dtype=torch.float32).pow(2))
            )

    return _LPD


def update_stored_models(model, m: int, h: int, n: int):
    # Do this in a context manager to prevent write race conditions.
    with open("shared/devinterp/tests/models.json", "r+") as f:
        models = json.load(f)

        models[f"{m}_{h}_{n}"] = {
            "fc1": model.fc1.weight.detach().numpy().tolist(),
            "fc2": model.fc2.weight.detach().numpy().tolist(),
        }

        f.seek(0)
        json.dump(models, f)

        f.truncate()

    return models


@pytest.fixture
def ReducedRankRegressor(is_snapshot_update):
    models = json.load(open("shared/devinterp/tests/models.json"))

    class _RRR(nn.Module):
        def __init__(self, m, h, n):
            super().__init__()
            self.fc1 = nn.Linear(m, h, bias=False)
            self.fc2 = nn.Linear(h, n, bias=False)
            self.is_cached = False

            key = f"{m}_{h}_{n}"

            if key in models:
                self.fc1.weight.data = torch.Tensor(models[key]["fc1"])
                self.fc2.weight.data = torch.Tensor(models[key]["fc2"])
                self.is_cached = True

        def forward(self, x):
            return self.fc2(self.fc1(x))

        def perturb(self):
            # Perturb the model by a large-enough amount
            # that our tests should fail.
            self.fc1.weight.data += torch.randn_like(self.fc1.weight.data)
            self.fc2.weight.data += torch.randn_like(self.fc2.weight.data)

    def maybe_train_model(m, h, n, x, y, criterion):
        nonlocal models
        # We'll retrain the model for every `--snapshot-update`.
        if is_snapshot_update:
            key = f"{m}_{h}_{n}"

            # Delete the model from the cache if it exists.
            if key in models:
                del models[key]

            _model = _RRR(m, h, n)
            assert not _model.is_cached

            # Train the model.
            optimizer = torch.optim.Adam(_model.parameters(), lr=0.01)
            for _ in range(5000):
                optimizer.zero_grad()
                outputs = _model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

            # Cache the model
            models = update_stored_models(_model, m, h, n)

        # Always reload the model from cache so we have reproducible results
        # between full/snapshot tests.
        model = _RRR(m, h, n)
        assert model.is_cached

        return model

    return maybe_train_model


@pytest.fixture
def DummyNaNModel():
    class _DN(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(2, 1)
            self.counter = 0
            with torch.no_grad():
                self.linear.weight.fill_(1.0)
                self.linear.bias.fill_(0.0)

        def forward(self, x):
            self.counter += 1
            if self.counter > 100:
                with torch.no_grad():
                    self.linear.weight.fill_(float("inf"))
            return self.linear(x)

    return _DN


# --- Shared dataset fixtures ---
@pytest.fixture
def generated_normalcrossing_dataset():
    """Shared dataset: normal input with small Gaussian noise."""
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 2, size=(num_samples,))
    y = sigma * torch.normal(0, 1, size=(num_samples,))
    train_data = TensorDataset(x, y)

    # Add deterministic generator for DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(
        train_data, batch_size=num_samples, shuffle=True, generator=generator
    )
    return train_dataloader, train_data, x, y


@pytest.fixture
def generated_linedot_normalcrossing_dataset():
    """Shared dataset: single-dim x, noise y for line-plus-dot tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_samples = 1000
    x = torch.normal(0, 2, size=(num_samples,))
    y = sigma * torch.normal(0, 1, size=(num_samples,))
    train_data = TensorDataset(x, y)

    # Add deterministic generator for DataLoader shuffling
    generator = torch.Generator()
    generator.manual_seed(42)
    train_dataloader = DataLoader(
        train_data, batch_size=num_samples, shuffle=True, generator=generator
    )
    return train_dataloader, train_data, x, y


@pytest.fixture
def run_llc_estimator():
    """
    Returns a callable to run LLCEstimator + sampling and return the estimator.
    Usage:
        estimator = run_llc_estimator(
            model, loader, sampling_method, lr,
            nbeta=None, num_chains=1, num_draws=100, seed=42,
            sampling_method_kwargs=None,
            **llc_init_kwargs
        )
    """

    def _run(
        model,
        loader,
        sampling_method,
        lr,
        nbeta=None,
        num_chains=1,
        num_draws=100,
        seed=42,
        sampling_method_kwargs=None,
        **llc_init_kwargs,
    ):
        if nbeta is None:
            nbeta = default_nbeta(loader)
        init_loss = get_init_loss_multi_batch(
            loader, num_chains, model, evaluate_mse, device="cpu"
        )
        estimator = LLCEstimator(
            num_chains=num_chains,
            num_draws=num_draws,
            nbeta=nbeta,
            init_loss=init_loss,
            **llc_init_kwargs,
        )
        torch.manual_seed(seed)
        sm_kwargs = {"lr": lr, "nbeta": nbeta}
        if sampling_method_kwargs:
            sm_kwargs.update(sampling_method_kwargs)
        _sample(
            model,
            loader,
            evaluate=evaluate_mse,
            sampling_method_kwargs=sm_kwargs,
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            num_burnin_steps=0,
            callbacks=[estimator],
            verbose=False,
            seed=seed,
        )
        return estimator

    return _run


# --- Snapshot fixture ---
@pytest.fixture
def is_snapshot_update(request):
    return request.config.getoption("--snapshot-update")
