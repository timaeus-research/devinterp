import pytest
import torch
from devinterp.optim.prior import CompositePrior, GaussianPrior, UniformPrior


def test_gaussian_prior_initialization():
    # Test with default zero center
    prior = GaussianPrior(localization=1.0, center=None)
    params = [torch.randn(2, 3), torch.randn(4)]
    state = prior.initialize(params)

    assert all(state[p]["prior_center"] is None for p in params)

    # Test with initial values as center
    prior = GaussianPrior(localization=1.0, center="initial")
    state = prior.initialize(params)

    assert all(torch.equal(state[p]["prior_center"], p) for p in params)

    # Test with explicit centers
    centers = [torch.zeros(2, 3), torch.zeros(4)]
    prior = GaussianPrior(localization=1.0, center=centers)
    state = prior.initialize(params)

    assert all(
        torch.equal(state[p]["prior_center"], c) for p, c in zip(params, centers)
    )


def test_gaussian_prior_grad():
    prior = GaussianPrior(localization=2.0)
    param = torch.tensor([1.0, 2.0, 3.0])
    state = {"prior_center": None}

    # Test gradient with zero center
    grad = prior.grad(param, state)
    expected = 2.0 * param
    assert torch.allclose(grad, expected)

    # Test gradient with non-zero center
    state["prior_center"] = torch.tensor([0.5, 1.0, 1.5])
    grad = prior.grad(param, state)
    expected = 2.0 * (param - state["prior_center"])
    assert torch.allclose(grad, expected)


def test_gaussian_prior_distance_sq():
    prior = GaussianPrior(localization=1.0)
    param = torch.tensor([1.0, 2.0, 3.0])

    # Test distance without state (zero-centered)
    dist = prior.distance_sq(param, state={"prior_center": None})
    expected = (param**2).sum()
    assert torch.equal(dist, expected)

    # Test distance with center
    state = {"prior_center": torch.tensor([0.5, 1.0, 1.5])}
    dist = prior.distance_sq(param, state)
    expected = ((param - state["prior_center"]) ** 2).sum()
    assert torch.equal(dist, expected)


def test_composite_prior():
    params = [torch.randn(2)]

    prior1 = GaussianPrior(localization=1.0, center=[params[0]])
    prior2 = GaussianPrior(localization=2.0, center=None)
    composite = CompositePrior([prior1, prior2])

    # Test key assignment
    assert prior1.key == "prior_center_0"
    assert prior2.key == "prior_center_1"

    # Test initialization
    state = composite.initialize(params)
    assert "prior_center_0" in state[params[0]]
    assert "prior_center_1" in state[params[0]]
    assert torch.allclose(state[params[0]]["prior_center_0"], params[0])
    assert state[params[0]]["prior_center_1"] is None

    # Test gradient combination
    param = torch.tensor([1.0, 2.0])
    state = {
        "prior_center_0": None,
        "prior_center_1": None,
    }
    grad = composite.grad(param, state)
    expected = 3.0 * param  # Sum of both priors (1.0 + 2.0) * param
    assert torch.allclose(grad, expected)


def test_composite_prior_single():
    # Test that CompositePrior returns the single prior when given only one
    prior = GaussianPrior(localization=1.0, center=None)
    composite = CompositePrior([prior])
    assert composite is prior


def test_composite_prior_empty():
    # Test that CompositePrior raises error when given empty list
    assert isinstance(CompositePrior([]), UniformPrior)


def test_custom_key():
    # Test custom key initialization
    prior = GaussianPrior(localization=1.0, key="custom_key")
    param = torch.randn(3)
    state = {"custom_key": None}

    grad = prior.grad(param, state)
    assert torch.allclose(grad, param)

    # Test that distance_sq uses custom key
    state["custom_key"] = torch.zeros(3)
    dist = prior.distance_sq(param, state)
    assert torch.equal(dist, (param**2).sum())


def test_uniform_prior():
    # Test initialization
    prior = UniformPrior()
    params = [torch.randn(2, 3), torch.randn(4)]
    state = prior.initialize(params)
    assert state == {}  # UniformPrior should return an empty state

    # Test gradient
    param = torch.tensor([1.0, 2.0, 3.0])
    grad = prior.grad(param, {})
    assert torch.equal(grad, torch.zeros_like(param))  # Gradient should be zero


def test_composite_prior_with_uniform():
    # Test that CompositePrior returns UniformPrior when only UniformPriors are provided
    uniform1 = UniformPrior()
    uniform2 = UniformPrior()
    composite = CompositePrior([uniform1, uniform2])
    assert isinstance(composite, UniformPrior)

    # Test mixture of Uniform and Gaussian priors
    gaussian = GaussianPrior(localization=1.0)
    composite = CompositePrior([uniform1, gaussian, uniform2])
    assert composite is gaussian

    # Test that single Gaussian with Uniforms returns just the Gaussian
    composite = CompositePrior([uniform1, gaussian])
    assert composite is gaussian  # Should return the single Gaussian directly

    # Test all UniformPriors returns a UniformPrior
    composite = CompositePrior([UniformPrior(), UniformPrior(), UniformPrior()])
    assert isinstance(composite, UniformPrior)
