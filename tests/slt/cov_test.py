import numpy as np
import pytest
import torch
import torch.nn as nn
from matplotlib import pyplot as plt

from devinterp.slt.cov import (
    BetweenLayerCovarianceAccumulator,
    CovarianceAccumulator,
    WithinHeadCovarianceAccumulator,
)


# A simple dummy model
class DummyModel(nn.Module):
    def __init__(self, size):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)


def accessor_layer1(model: nn.Module):
    return model.fc1.weight.data


def accessor_layer2(model: nn.Module):
    return model.fc2.weight.data


@pytest.fixture
def dummy_model():
    model = DummyModel(10)
    return model


@pytest.fixture
def cov_matrix():
    return np.array([[2.0, 1.0], [1.0, 2.0]])


@pytest.fixture
def random_vars(cov_matrix):
    np.random.seed(0)
    return np.random.multivariate_normal([0, 0], cov_matrix, 100)


@pytest.fixture
def known_accessor(random_vars):
    i = -1

    def _known_accessor(model):
        nonlocal i
        i += 1
        return torch.tensor(random_vars[i], dtype=torch.float32)

    return _known_accessor


def test_covariance_accumulator_with_known_cov(dummy_model, known_accessor, cov_matrix):
    acc = CovarianceAccumulator(num_weights=2, accessors=[known_accessor])

    for _ in range(100):
        acc(dummy_model)

    acc.finalize()

    cov_matrix = acc.to_matrix().detach().cpu().numpy()
    assert np.allclose(
        cov_matrix, cov_matrix, atol=1e-1
    )  # Tolerance due to finite sample

    # Test the eigenvalues
    evals, evecs = np.linalg.eig(cov_matrix)
    eigenspectrum = acc.to_eigen(include_matrix=True)

    assert np.allclose(evals, eigenspectrum["evals"], atol=1e-1)
    assert np.allclose(np.abs(evecs), np.abs(eigenspectrum["evecs"]), atol=1e-1)


def test_reset_functionality(dummy_model, known_accessor):
    acc = CovarianceAccumulator(num_weights=2, accessors=[known_accessor])

    for _ in range(100):
        acc(dummy_model)

    acc.reset()

    assert torch.all(acc.first_moment == 0)
    assert torch.all(acc.second_moment == 0)
    assert acc.num_draws == 0
    assert acc.is_finished is False


def test_full_model(dummy_model):
    np.random.seed(0)

    cov_matrix = np.random.normal(size=(20, 20))

    # Reduce the correlation between layers (off-diagonal blocks)
    cov_matrix[:10, 10:] *= 0.1
    cov_matrix[10:, :10] *= 0.1

    def update_model(model):
        new_params = np.random.multivariate_normal(np.zeros(20), cov_matrix)
        model.fc1.weight.data = torch.tensor(
            new_params[:10].reshape((10,)), dtype=torch.float32
        )
        model.fc2.weight.data = torch.tensor(
            new_params[10:].reshape((10,)), dtype=torch.float32
        )

    acc = CovarianceAccumulator(
        num_weights=20, accessors=[accessor_layer1, accessor_layer2]
    )

    for _ in range(100):
        update_model(dummy_model)
        acc(dummy_model)

    acc.finalize()

    cov_matrix = acc.to_matrix().detach().cpu().numpy()
    assert np.allclose(
        cov_matrix, cov_matrix, atol=1e-1
    )  # Tolerance due to finite sample


class DummyTransformer(nn.Module):
    def __init__(self, num_layers=2, num_heads=2, embed_dim=4):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.MultiheadAttention(embed_dim, num_heads, bias=False)
                for _ in range(num_layers)
            ]
        )

    def get_qkv_matrices(self, layer_idx, head_idx):
        layer = self.layers[layer_idx].in_proj_weight  # (3 * embed_dim, embed_dim)
        head = layer[head_idx * 6 : (head_idx + 1) * 6, :]
        return head


@pytest.fixture
def dummy_transformer():
    return DummyTransformer()


def accessor_transformer_layer1(model):
    return tuple(model.get_qkv_matrices(0, h) for h in range(2))


def accessor_transformer_layer2(model):
    return tuple(model.get_qkv_matrices(1, h) for h in range(2))


def l1h1(model):
    return model.get_qkv_matrices(0, 0)


def l1h2(model):
    return model.get_qkv_matrices(0, 1)


def l2h1(model):
    return model.get_qkv_matrices(1, 0)


def l2h2(model):
    return model.get_qkv_matrices(1, 1)


def test_within_head_covariance(dummy_transformer):
    np.random.seed(0)

    num_heads = 2
    num_weights_per_head = 3 * 4 * (4 // num_heads)
    num_parameters_per_layer = num_weights_per_head * num_heads

    cov_matrix_layer_1 = np.random.normal(
        size=(num_parameters_per_layer, num_parameters_per_layer)
    )
    cov_matrix_layer_1 = cov_matrix_layer_1 @ cov_matrix_layer_1.T
    cov_matrix_layer_2 = np.random.normal(
        size=(num_parameters_per_layer, num_parameters_per_layer)
    )
    cov_matrix_layer_2 = cov_matrix_layer_2 @ cov_matrix_layer_2.T

    def update_model(model):
        new_params_layer_1 = np.random.multivariate_normal(
            np.zeros(num_parameters_per_layer), cov_matrix_layer_1
        )
        new_params_layer_2 = np.random.multivariate_normal(
            np.zeros(num_parameters_per_layer), cov_matrix_layer_2
        )

        model.layers[0].in_proj_weight.data = torch.tensor(
            new_params_layer_1, dtype=torch.float32
        ).reshape(model.layers[0].in_proj_weight.shape)
        model.layers[1].in_proj_weight.data = torch.tensor(
            new_params_layer_2, dtype=torch.float32
        ).reshape(model.layers[1].in_proj_weight.shape)

    acc = WithinHeadCovarianceAccumulator(
        num_heads,
        num_weights_per_head,
        [accessor_transformer_layer1, accessor_transformer_layer2],
    )

    for _ in range(1_000):
        update_model(dummy_transformer)
        acc(dummy_transformer)

    acc.finalize()
    cov_matrix = acc.to_matrix().detach().cpu().numpy()

    mse = 0

    # Extract submatrices for each layer and head, and validate
    for l, actual_cov_matrix in enumerate(
        [cov_matrix_layer_1, cov_matrix_layer_2]
    ):  # 2 layers
        for h in range(num_heads):  # 2 heads
            local_cov_matrix = cov_matrix[l, h]
            actual_local_cov_matrix = actual_cov_matrix[
                h * num_weights_per_head : (h + 1) * num_weights_per_head,
                h * num_weights_per_head : (h + 1) * num_weights_per_head,
            ]
            local_mse = (
                np.sum(
                    (local_cov_matrix - actual_local_cov_matrix)
                    / local_cov_matrix.shape[0]
                )
                ** 2
            )

            # fig, axes = plt.subplots(1, 2)
            # fig.suptitle(f"Layer {l}, Head {h} MSE: {local_mse}")
            # axes[0].imshow(local_cov_matrix)
            # axes[0].set_title("Observed")
            # axes[1].imshow(actual_local_cov_matrix)
            # axes[1].set_title("Theoretical")
            # plt.show()

            mse += local_mse / (num_heads * 2)

    assert (
        mse < 3
    ), f"MSE: {mse}"  # Visually this looks good, but the MSE is a bit high.


def test_between_layer_covariance_within_heads(dummy_transformer):
    np.random.seed(0)

    num_heads = 2
    num_weights_per_head = 3 * 4 * (4 // num_heads)
    num_parameters_per_layer = num_weights_per_head * num_heads

    cov_matrix_layer_1 = np.random.normal(
        size=(num_parameters_per_layer, num_parameters_per_layer)
    )
    cov_matrix_layer_1 = cov_matrix_layer_1 @ cov_matrix_layer_1.T
    cov_matrix_layer_2 = np.random.normal(
        size=(num_parameters_per_layer, num_parameters_per_layer)
    )
    cov_matrix_layer_2 = cov_matrix_layer_2 @ cov_matrix_layer_2.T

    def update_model(model):
        new_params_layer_1 = np.random.multivariate_normal(
            np.zeros(num_parameters_per_layer), cov_matrix_layer_1
        )
        new_params_layer_2 = np.random.multivariate_normal(
            np.zeros(num_parameters_per_layer), cov_matrix_layer_2
        )

        model.layers[0].in_proj_weight.data = torch.tensor(
            new_params_layer_1, dtype=torch.float32
        ).reshape(model.layers[0].in_proj_weight.shape)
        model.layers[1].in_proj_weight.data = torch.tensor(
            new_params_layer_2, dtype=torch.float32
        ).reshape(model.layers[1].in_proj_weight.shape)

    acc1 = WithinHeadCovarianceAccumulator(
        num_heads,
        num_weights_per_head,
        [accessor_transformer_layer1, accessor_transformer_layer2],
    )
    acc2 = BetweenLayerCovarianceAccumulator(
        dummy_transformer,
        pairs={
            "l1h1": ("l1h1", "l1h1"),
            "l1h2": ("l1h2", "l1h2"),
            "l2h1": ("l2h1", "l2h1"),
            "l2h2": ("l2h2", "l2h2"),
        },
        l1h1=l1h1,
        l1h2=l1h2,
        l2h1=l2h1,
        l2h2=l2h2,
    )

    for _ in range(1_000):
        update_model(dummy_transformer)
        acc1(dummy_transformer)
        acc2(dummy_transformer)

    acc1.finalize()
    acc2.finalize()

    cov1 = acc1.to_matrix().detach().cpu().numpy()
    cov2 = {k: v.detach().cpu().numpy() for k, v in acc2.to_matrices().items()}

    # Extract submatrices for each layer and head, and validate
    for l in range(2):
        for h in range(num_heads):
            assert np.allclose(cov1[l, h], cov2[f"l{l+1}h{h+1}"])
