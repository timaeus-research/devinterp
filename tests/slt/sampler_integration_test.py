import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from devinterp.slt import sample
from devinterp.slt.llc import LLCEstimator


class PolyModel(nn.Module):
    def __init__(self, powers=[1, 2]):
        super(PolyModel, self).__init__()
        self.powers = torch.tensor(powers)
        self.weights = nn.Parameter(
            torch.tensor(
                torch.zeros_like(self.powers, dtype=torch.float32), requires_grad=True
            )
        )

    def forward(self, x):
        return x * torch.prod(self.weights**self.powers)


class LinePlusDotModel(nn.Module):
    def __init__(self, dim=2):
        super(LinePlusDotModel, self).__init__()
        self.weights = nn.Parameter(
            torch.zeros(dim, dtype=torch.float32), requires_grad=True
        )

    def forward(self, x):
        return x * (self.weights[0] - 1) * (torch.sum(self.weights**2) ** 2)


@pytest.fixture
def generated_dataset():
    torch.manual_seed(42)
    np.random.seed(42)
    sigma = 0.25
    num_train_samples = 1000
    batch_size = num_train_samples
    x = torch.normal(0, 2, size=(num_train_samples,))
    y = sigma * torch.normal(0, 1, size=(num_train_samples,))
    train_data = TensorDataset(x, y)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_loader, train_data, x, y


@pytest.mark.parametrize("sampling_method", [SGLD, SGNHT])
@pytest.mark.parametrize("model", [LinePlusDotModel(), PolyModel()])
def test_llc_ordinality_2d(generated_dataset, sampling_method, model):
    train_loader, train_data, _, _ = generated_dataset
    criterion = F.mse_loss
    lr = 0.0001
    num_chains = 10
    num_draws = 5_000
    llcs = []
    sample_points = [[0.0, 0.0], [1.0, 0.0]]
    for sample_point in sample_points:
        model.weights = nn.Parameter(
            torch.tensor(sample_point, dtype=torch.float32, requires_grad=True)
        )
        llc_estimator = LLCEstimator(
            num_chains=num_chains, num_draws=num_draws, n=len(train_data)
        )
        sample(
            model,
            train_loader,
            criterion=criterion,
            optimizer_kwargs=dict(
                lr=lr,
                bounding_box_size=0.2,
                num_samples=len(train_data),
            ),
            sampling_method=sampling_method,
            num_chains=num_chains,
            num_draws=num_draws,
            callbacks=[llc_estimator],
        )
        llcs += [llc_estimator.sample()]
    assert llcs[0]["llc/mean"] < llcs[1]["llc/mean"]
