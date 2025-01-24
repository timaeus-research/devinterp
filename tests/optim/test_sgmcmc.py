from copy import deepcopy

import numpy as np
import pytest
import torch
import torch.nn as nn
from devinterp.optim.preconditioner import RMSpropPreconditioner
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgmcmc import SGMCMC
from syrupy.assertion import SnapshotAssertion

BATCH_SIZE = 3
WIDTH = 5
STEPS = 5


def create_model(bias=True):
    return nn.Sequential(
        nn.Linear(WIDTH, WIDTH, bias=bias),
        nn.Linear(WIDTH, WIDTH, bias=bias),
        nn.Linear(WIDTH, WIDTH, bias=bias),
    )


def create_paired_models(seed=0):
    """Helper function for creating identical model pairs"""
    torch.manual_seed(seed)

    model1 = create_model()
    model2 = deepcopy(model1)
    return model1, model2


def create_task():
    """Helper function for creating test data"""
    data = torch.randn(BATCH_SIZE, WIDTH)
    target = torch.randn(BATCH_SIZE, WIDTH)
    criterion = nn.MSELoss()
    return data, target, criterion


def run_optimization_steps(model, optimizer, data, target, criterion, steps=2, seed=42):
    """Helper function to run optimization steps"""
    torch.manual_seed(seed)

    for _ in range(steps):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


def compare_parameters(model1, model2, atol=1e-5):
    """Helper function to compare model parameters"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=atol), f"Parameters differ: {p1} vs {p2}"


def compare_metrics(optimizer_sgld, optimizer_sgmcmc, metrics):
    """Helper function to compare optimizer metrics"""
    for i, (g1, g2) in enumerate(
        zip(optimizer_sgld.param_groups, optimizer_sgmcmc.param_groups)
    ):
        for metric in metrics:
            if metric == "noise":
                for a, b in zip(optimizer_sgld.noise[i], g2["metrics"]["noise"]):
                    assert a.shape == b.shape, "Noise tensors differ"
                    assert torch.allclose(a, b), "Noise tensors differ"
            elif metric == "dws":
                pi = 0
                for j, a in enumerate(g2["metrics"]["dws"]):
                    b = optimizer_sgld.dws[pi]
                    pi += 1
                    assert a.shape == b.shape, "DWS tensors differ in shape"
                    assert torch.allclose(a, b, atol=1e-4), "DWS metrics differ"
            else:
                assert torch.allclose(
                    g2["metrics"][metric],
                    getattr(optimizer_sgld, metric),
                ), f"Metric {metric} differs"


def _serialize(obj, precision=2):
    """Helper function to recursively convert values to serializable format with rounding"""
    if isinstance(obj, (float, np.floating)):
        return np.format_float_positional(
            obj, precision=precision, unique=False, fractional=False
        )
    elif isinstance(obj, torch.Tensor):
        return _serialize(obj.detach().cpu().numpy(), precision)
    elif isinstance(obj, np.ndarray):
        if obj.flatten().shape == (1,):
            return _serialize(float(obj.flatten()[0]), precision)
        else:
            return [_serialize(x, precision) for x in obj]
    elif isinstance(obj, dict):
        return {k: _serialize(v, precision) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize(x, precision) for x in obj]
    return obj


def serialize_model_state(model, precision=2):
    """Helper to convert model parameters to serializable format with rounding"""
    model_state = {}
    for name, param in model.named_parameters():
        model_state[name] = param.detach().cpu().numpy()
    return _serialize(model_state, precision)


def serialize_metrics(optimizer, precision=2):
    """Helper to convert optimizer metrics to serializable format with rounding"""
    metrics = {}
    for group in optimizer.param_groups:
        if "metrics" in group:
            metrics.update(_serialize(group["metrics"], precision))
    return metrics


@pytest.mark.parametrize("lr", [1e-1, 1e-2, 1e-3, 1e-4])
@pytest.mark.parametrize("nbeta", [1.0, 10.0])
@pytest.mark.parametrize("localization", [0.0, 0.1])
@pytest.mark.parametrize("bounding_box_size", [None, 0.1])
@pytest.mark.parametrize("weight_decay", [0.0, 0.05])
def test_SGMCMC_vs_SGLD(
    lr, nbeta, localization, bounding_box_size, weight_decay, snapshot
):
    model1, model2 = create_paired_models()
    data, target, criterion = create_task()

    kwargs = dict(
        lr=lr,
        noise_level=1.0,
        nbeta=nbeta,
        localization=localization,
        bounding_box_size=bounding_box_size,
        weight_decay=weight_decay,
    )

    metrics = [
        "noise_norm",
        "grad_norm",
        "weight_norm",
        "noise",
        "dws",
        "localization_loss",
        "distance",
    ]

    optimizer_sgld = SGLD(
        model1.parameters(),
        **kwargs,
        save_noise=True,
        save_mala_vars=True,
        noise_norm=True,
        grad_norm=True,
        weight_norm=True,
        distance=True,
    )

    optimizer_sgmcmc = SGMCMC.sgld(
        model2.parameters(),
        **kwargs,
        metrics=metrics,
    )

    run_optimization_steps(model1, optimizer_sgld, data, target, criterion)
    run_optimization_steps(model2, optimizer_sgmcmc, data, target, criterion)

    compare_parameters(model1, model2)
    compare_metrics(optimizer_sgld, optimizer_sgmcmc, metrics)

    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
        "optimizer_sgld": serialize_metrics(optimizer_sgld),
        "optimizer_sgmcmc": serialize_metrics(optimizer_sgmcmc),
    }

    assert state == snapshot(
        name=f"test_SGMCMC_vs_SGLD_{lr}_{nbeta}_{localization}_{bounding_box_size}_{weight_decay}"
    )


@pytest.mark.parametrize("lr", [1e-1])
@pytest.mark.parametrize("nbeta", [1.0])
@pytest.mark.parametrize("localization", [0.0, 0.1])
@pytest.mark.parametrize("bounding_box_size", [None, 0.1])
@pytest.mark.parametrize("weight_decay", [0.0])
@pytest.mark.parametrize("optimize_over", ["scalar", "tensor"])
def test_optimize_over(
    lr,
    nbeta,
    localization,
    bounding_box_size,
    weight_decay,
    optimize_over,
    snapshot: SnapshotAssertion,
):
    # Create identical models
    model1, model2 = create_paired_models()

    original_params = [p.clone() for p in model1.parameters()]

    torch.manual_seed(42)
    kwargs = dict(
        lr=lr,
        noise_level=1.0,
        nbeta=nbeta,
        localization=localization,
        bounding_box_size=bounding_box_size,
        weight_decay=weight_decay,
    )

    metrics = [
        "noise_norm",
        "grad_norm",
        "weight_norm",
        "noise",
        "dws",
        "localization_loss",
        "distance",
    ]

    torch.manual_seed(42)

    if optimize_over == "tensor":
        optimize_over_params = [
            torch.randint(0, 2, p.shape).bool() for p in model1.parameters()
        ]
    elif optimize_over == "scalar":
        optimize_over_params = [
            torch.randint(0, 2, (1,)).bool() for p in model1.parameters()
        ]

    # Setup optimizers with equivalent parameters
    optimizer_sgld = SGLD(
        [
            {"params": p, "optimize_over": opt}
            for p, opt in zip(model1.parameters(), optimize_over_params)
        ],
        **kwargs,
        save_noise=True,
        save_mala_vars=True,
        noise_norm=True,
        grad_norm=True,
        weight_norm=True,
        distance=True,
    )

    optimizer_sgmcmc = SGMCMC(
        [
            {"params": (p,), "optimize_over": (opt,)}
            for p, opt in zip(model2.parameters(), optimize_over_params)
        ],
        **kwargs,
        metrics=metrics,
    )

    # Create test data
    data, target, criterion = create_task()

    # Set same random seed for both optimizers
    run_optimization_steps(model1, optimizer_sgld, data, target, criterion)
    run_optimization_steps(model2, optimizer_sgmcmc, data, target, criterion)

    compare_parameters(model1, model2)

    if optimize_over == "tensor":
        for p1, p2, mask in zip(
            original_params, model2.parameters(), optimize_over_params
        ):
            assert torch.allclose(
                p1[~mask], p2[~mask], atol=1e-5
            ), f"Masked parameters differ: {p1} vs {p2} for mask {mask}"

    elif optimize_over == "scalar":
        for p1, p2, mask in zip(
            original_params, model2.parameters(), optimize_over_params
        ):
            if not mask:
                assert torch.allclose(
                    p1, p2, atol=1e-5
                ), f"Masked parameters differ: {p1} vs {p2}"

    # Compare parameters
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-5), f"Parameters differ: {p1} vs {p2}"

    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
    }

    assert state == snapshot(
        name=f"test_optimize_over_{lr}_{nbeta}_{localization}_{bounding_box_size}_{weight_decay}_{optimize_over}"
    )


@pytest.mark.parametrize("lr", [1e-1, 1e-2])
def test_SGMCMC_deterministic(lr, snapshot):
    """Test that SGMCMC behaves like SGD when noise is disabled"""
    model1, model2 = create_paired_models()
    data, target, criterion = create_task()

    optimizer_sgd = torch.optim.SGD(model1.parameters(), lr=lr)
    optimizer_sgmcmc = SGMCMC.sgld(
        model2.parameters(),
        lr=2 * lr,  # Double LR to match SGMCMC's 0.5 factor in gradient update
        noise_level=0.0,
        nbeta=1.0,
        localization=0.1,
    )

    run_optimization_steps(model1, optimizer_sgd, data, target, criterion, steps=1)
    run_optimization_steps(model2, optimizer_sgmcmc, data, target, criterion, steps=1)

    compare_parameters(model1, model2, atol=1e-6)

    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
    }

    assert state == snapshot(name=f"test_SGMCMC_deterministic_{lr}")


@pytest.mark.parametrize("lr", [1e-1, 1e-2])
@pytest.mark.parametrize("diffusion_factor", [0.0, 0.01])
@pytest.mark.parametrize("bounding_box_size", [None, 0.1])
def test_SGMCMC_vs_SGNHT(lr, diffusion_factor, bounding_box_size, snapshot):
    """Test that SGMCMC with SGNHT settings matches expected behavior"""
    model1, model2 = create_paired_models()
    data, target, criterion = create_task()

    # Setup optimizers
    optimizer_sgnht = SGMCMC.sgnht(
        model1.parameters(),
        lr=lr,
        diffusion_factor=diffusion_factor,
        bounding_box_size=bounding_box_size,
        nbeta=1.0,
    )

    optimizer_sgmcmc = SGMCMC.sgnht(
        model2.parameters(),
        lr=lr,
        diffusion_factor=diffusion_factor,
        bounding_box_size=bounding_box_size,
        nbeta=1.0,
    )

    run_optimization_steps(
        model1, optimizer_sgnht, data, target, criterion, steps=STEPS, seed=42
    )
    run_optimization_steps(
        model2, optimizer_sgmcmc, data, target, criterion, steps=STEPS, seed=42
    )

    compare_parameters(model1, model2, atol=1e-6)

    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
    }

    assert state == snapshot(
        name=f"test_SGMCMC_vs_SGNHT_{lr}_{diffusion_factor}_{bounding_box_size}"
    )


def test_SGMCMC_bounding_box(snapshot):
    """Test that SGMCMC respects the bounding box constraint"""
    model = create_model()
    data, target, criterion = create_task()

    # Create optimizer with small bounding box
    box_size = 0.01
    optimizer = SGMCMC(
        model.parameters(),
        lr=1.0,  # Large learning rate to ensure parameters would move outside box
        noise_level=1.0,
        nbeta=1.0,
        prior=None,
        bounding_box_size=box_size,
    )

    run_optimization_steps(model, optimizer, data, target, criterion, steps=10)

    # Check that all parameters stay within the bounding box
    for p in model.parameters():
        initial_param = optimizer.state[p]["initial_param"]
        assert torch.all(
            torch.abs(p.data - initial_param) <= box_size + 1e-6
        ), f"Parameter exceeded bounding box: diff={torch.abs(p.data - initial_param).max()}, box_size={box_size}"

    state = {
        "model": serialize_model_state(model),
    }

    assert state == snapshot(name=f"test_SGMCMC_bounding_box_{box_size}")


def test_SGMCMC_param_groups(snapshot):
    """Test that SGMCMC correctly handles parameter groups with different settings"""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    model = create_model()

    # Create two parameter groups with different learning rates and noise levels
    param_groups = [
        {"params": model[0].parameters(), "lr": 0.1, "noise_level": 1.0},
        {"params": model[1].parameters(), "lr": 0.0, "noise_level": 0.0},
    ]

    optimizer = SGMCMC(param_groups)

    # Verify that parameters were assigned to correct groups
    assert len(optimizer.param_groups) == 2
    assert next(model[0].parameters()) in optimizer.param_groups[0]["params"]
    assert next(model[1].parameters()) in optimizer.param_groups[1]["params"]

    data, target, criterion = create_task()

    # Store initial parameters
    params_before = [
        {name: p.clone() for name, p in layer.named_parameters()} for layer in model
    ]

    run_optimization_steps(model, optimizer, data, target, criterion, steps=1)

    # Verify that parameters in different groups updated differently
    for group_idx, layer_params in enumerate(params_before):
        for name, p_before in layer_params.items():
            p_after = dict(model[group_idx].named_parameters())[name]
            param_diff = (p_after - p_before).abs().mean()

            # Group 0 should have larger updates due to higher learning rate
            if group_idx == 0:
                assert param_diff > 0.01, "First group parameters didn't update enough"
            else:
                assert param_diff == 0.0, "Second group parameters updated too much"

    state = {
        "model": serialize_model_state(model),
    }

    assert state == snapshot(name="test_SGMCMC_param_groups")


@pytest.mark.parametrize("lr_ratio", [3.0, 0.5])
@pytest.mark.parametrize("noise_ratio", [3.0, 0.5])
def test_SGMCMC_vs_SGLD_param_groups(lr_ratio, noise_ratio, snapshot):
    """Test that SGMCMC and SGLD behave identically with different parameter groups"""
    # Create identical models
    model1 = create_model()
    model2 = deepcopy(model1)

    # Create two parameter groups with different learning rates and noise levels
    base_lr = 1e-4
    base_noise = 0.0
    param_groups1 = [
        {
            "params": list(model1[0:2].parameters()),
            "lr": 0,
            "noise_level": base_noise,
            "localization": 0.1,
        },
        {
            "params": list(model1[2].parameters()),
            "lr": base_lr * lr_ratio,
            "noise_level": base_noise * noise_ratio,
            "localization": 0.5,
        },
    ]
    param_groups2 = [
        {
            "params": list(model2[0:2].parameters()),
            "lr": 0,
            "noise_level": base_noise,
            "localization": 0.1,
        },
        {
            "params": list(model2[2].parameters()),
            "lr": base_lr * lr_ratio,
            "noise_level": base_noise * noise_ratio,
            "localization": 0.5,
        },
    ]

    metrics = [
        "noise_norm",
        "grad_norm",
        "weight_norm",
        "distance",
    ]  # , "noise"]  # , "dws"]

    # Setup optimizers with equivalent parameters
    optimizer_sgld = SGLD(
        param_groups1,
        nbeta=1.0,
        save_noise=True,
        save_mala_vars=True,
        noise_norm=True,
        grad_norm=True,
        weight_norm=True,
        distance=True,
    )

    optimizer_sgmcmc = SGMCMC(
        param_groups2,
        nbeta=1.0,
        metrics=metrics,
    )

    # Check hyperparameters in groups
    for g1, g2 in zip(optimizer_sgld.param_groups, optimizer_sgmcmc.param_groups):
        assert g1["lr"] == g2["lr"], f"LRs differ: {g1['lr']} vs {g2['lr']}"
        assert (
            g1["noise_level"] == g2["noise_level"]
        ), f"Noise levels differ: {g1['noise_level']} vs {g2['noise_level']}"
        assert (
            g1["nbeta"] == g2["nbeta"]
        ), f"nbeta differ: {g1['nbeta']} vs {g2['nbeta']}"
        assert (
            g1["localization"] == g2["localization"]
        ), f"Localizations differ: {g1['localization']} vs {g2['localization']}"

    data, target, criterion = create_task()

    run_optimization_steps(
        model1, optimizer_sgld, data, target, criterion, steps=STEPS, seed=42
    )
    run_optimization_steps(
        model2, optimizer_sgmcmc, data, target, criterion, steps=STEPS, seed=42
    )

    compare_parameters(model1, model2, atol=1e-4)

    # Compare metrics (Need this custom comparison because of different storage)
    for i, (g1, g2) in enumerate(
        zip(optimizer_sgld.param_groups, optimizer_sgmcmc.param_groups)
    ):
        for metric in metrics:
            if metric == "noise":
                # Compare noise tensors for each parameter group
                for a, b in zip(
                    optimizer_sgld.noise[i],
                    g2["metrics"]["noise"],
                ):
                    assert a.shape == b.shape, "Noise tensors differ"
                    assert torch.allclose(a, b), "Noise tensors differ"
            elif metric == "dws":
                pi = 0
                for j, a in enumerate(g2["metrics"]["dws"]):
                    b = optimizer_sgld.dws[pi]
                    pi += 1
                    assert a.shape == b.shape, "DWS tensors differ in shape"
                    assert torch.allclose(a, b, atol=1e-4), "DWS metrics differ"
            else:
                assert torch.allclose(
                    g2["metrics"][metric],
                    g1[metric],
                ), f"Metric {metric} differs"

    state = {
        "optimizer_sgld": serialize_metrics(optimizer_sgld),
        "optimizer_sgmcmc": serialize_metrics(optimizer_sgmcmc),
    }

    assert state == snapshot(name="test_SGMCMC_vs_SGLD_param_groups")


@pytest.mark.parametrize("lr", [1e-1, 1e-2])
@pytest.mark.parametrize("alpha", [0.9, 0.99])
@pytest.mark.parametrize("eps", [1e-8, 0.1])
@pytest.mark.parametrize("add_grad_correction", [False, True])
def test_SGMCMC_rmsprop_sgld(lr, alpha, eps, add_grad_correction, snapshot):
    """Test that SGMCMC with RMSprop preconditioning behaves as expected"""
    # Skip tests with grad correction until implemented
    if add_grad_correction:
        pytest.skip("Gradient correction not yet implemented for RMSprop")

    model = create_model()

    # Setup optimizer
    optimizer = SGMCMC.rmsprop_sgld(
        model.parameters(),
        lr=lr,
        alpha=alpha,
        eps=eps,
        add_grad_correction=add_grad_correction,
        nbeta=1.0,
        metrics=["grad_norm", "noise_norm", "weight_norm"],
    )

    # Verify RMSprop preconditioner settings
    for group in optimizer.param_groups:
        assert isinstance(group["preconditioner"], RMSpropPreconditioner)
        assert group["preconditioner"].alpha == alpha
        assert group["preconditioner"].eps == eps
        assert group["preconditioner"].add_grad_correction == add_grad_correction

    data, target, criterion = create_task()

    # Take multiple optimization steps
    for step in range(5):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Verify that square_avg is being updated in state
        for p in model.parameters():
            state = optimizer.state[p]
            assert "square_avg" in state
            assert isinstance(state["square_avg"], torch.Tensor)
            assert not torch.isnan(state["square_avg"]).any()
            assert not torch.isinf(state["square_avg"]).any()

        # Verify metrics are being tracked
        assert optimizer.metrics["grad_norm"] >= 0
        assert optimizer.metrics["noise_norm"] >= 0
        assert optimizer.metrics["weight_norm"] >= 0

    state = {
        "optimizer": serialize_metrics(optimizer),
    }

    assert state == snapshot(name="test_SGMCMC_rmsprop_sgld")


@pytest.mark.parametrize("lr", [1e-1, 1e-2, 1e-3, 1e-4])
def test_SGMCMC_rmsprop_equals_sgld(lr, snapshot):
    """Test that RMSprop-preconditioned SGLD equals regular SGLD when alpha=0 and eps=1"""
    model1, model2 = create_paired_models()
    data, target, criterion = create_task()

    # Common parameters
    kwargs = dict(
        lr=lr,
        noise_level=1.0,
        nbeta=1.0,
        localization=0.1,
        weight_decay=0.0,
    )

    metrics = ["grad_norm", "noise_norm", "weight_norm", "noise"]

    # Regular SGLD
    optimizer_sgld = SGLD(
        model1.parameters(),
        **kwargs,
        metrics=metrics,
    )

    # RMSprop SGLD with identity-like settings
    optimizer_rmsprop = SGMCMC.rmsprop_sgld(
        model2.parameters(),
        alpha=1.0,  # No momentum
        eps=1.0,  # Makes preconditioner act like identity
        add_grad_correction=False,
        **kwargs,
        metrics=metrics,
    )

    # Verify optimizers behave identically
    run_optimization_steps(
        model1, optimizer_sgld, data, target, criterion, steps=STEPS, seed=42
    )
    run_optimization_steps(
        model2, optimizer_rmsprop, data, target, criterion, steps=STEPS, seed=42
    )

    compare_parameters(model1, model2, atol=1e-6)
    compare_metrics(optimizer_sgld, optimizer_rmsprop, metrics)

    state = {
        "model1": serialize_model_state(model1),
        "model2": serialize_model_state(model2),
        "optimizer_sgld": serialize_metrics(optimizer_sgld),
        "optimizer_rmsprop": serialize_metrics(optimizer_rmsprop),
    }

    assert state == snapshot(name="test_SGMCMC_rmsprop_equals_sgld")
