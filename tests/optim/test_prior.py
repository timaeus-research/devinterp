import torch
from devinterp.optim.prior import CompositePrior, GaussianPrior, UniformPrior


def test_gaussian_prior_initialization_zero_center():
    prior = GaussianPrior(localization=1.0, center=None)
    params = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
    state = prior.initialize(params)

    assert all(state[p]["prior_center"] is None for p in params)


def test_gaussian_prior_initialization_initial_center():
    prior = GaussianPrior(localization=1.0, center="initial")
    params = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
    state = prior.initialize(params)

    torch.testing.assert_close(
        state[params[0]]["prior_center"],
        torch.tensor([1.0, 2.0, 3.0]),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        state[params[1]]["prior_center"],
        torch.tensor([4.0, 5.0]),
        atol=0,
        rtol=0,
    )

    # Centers must be independent copies, not aliases
    params[0][0] = 999.0
    torch.testing.assert_close(
        state[params[0]]["prior_center"],
        torch.tensor([1.0, 2.0, 3.0]),
        atol=0,
        rtol=0,
    )


def test_gaussian_prior_initialization_explicit_centers():
    centers = [torch.tensor([10.0, 20.0]), torch.tensor([30.0])]
    prior = GaussianPrior(localization=1.0, center=centers)
    params = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    state = prior.initialize(params)

    torch.testing.assert_close(
        state[params[0]]["prior_center"], torch.tensor([10.0, 20.0]), atol=0, rtol=0
    )
    torch.testing.assert_close(
        state[params[1]]["prior_center"], torch.tensor([30.0]), atol=0, rtol=0
    )


def test_gaussian_prior_grad_zero_centered():
    prior = GaussianPrior(localization=2.0, center=None)
    param = torch.tensor([1.0, 2.0, 3.0])
    state = {"prior_center": None}

    grad = prior.grad(param, state)
    torch.testing.assert_close(grad, torch.tensor([2.0, 4.0, 6.0]), atol=0, rtol=0)


def test_gaussian_prior_grad_with_center():
    prior = GaussianPrior(localization=2.0, center="initial")
    param = torch.tensor([1.0, 2.0, 3.0])
    state = {"prior_center": torch.tensor([0.5, 1.0, 1.5])}

    grad = prior.grad(param, state)
    torch.testing.assert_close(grad, torch.tensor([1.0, 2.0, 3.0]), atol=0, rtol=0)


def test_composite_prior():
    p = torch.tensor([1.0, 2.0])

    prior1 = GaussianPrior(localization=1.0, center=[p.clone()])
    prior2 = GaussianPrior(localization=2.0, center=None)
    composite = CompositePrior([prior1, prior2])

    assert prior1.key == "prior_center_0"
    assert prior2.key == "prior_center_1"

    state = composite.initialize([p])
    assert "prior_center_0" in state[p]
    assert "prior_center_1" in state[p]
    torch.testing.assert_close(
        state[p]["prior_center_0"], torch.tensor([1.0, 2.0]), atol=0, rtol=0
    )
    assert state[p]["prior_center_1"] is None


def test_composite_prior_grad():
    param = torch.tensor([1.0, 2.0])
    prior1 = GaussianPrior(localization=1.0, center=None)
    prior2 = GaussianPrior(localization=2.0, center=None)
    composite = CompositePrior([prior1, prior2])
    state = {
        "prior_center_0": None,
        "prior_center_1": None,
    }

    grad = composite.grad(param, state)
    # (1.0 + 2.0) * [1, 2] = [3, 6]
    torch.testing.assert_close(grad, torch.tensor([3.0, 6.0]), atol=0, rtol=0)


def test_composite_prior_single():
    prior = GaussianPrior(localization=1.0, center=None)
    composite = CompositePrior([prior])
    assert len(composite.priors) == 1

    param = torch.tensor([1.0, 2.0])
    state = composite.initialize([param])
    grad = composite.grad(param, state[param])
    torch.testing.assert_close(grad, torch.tensor([1.0, 2.0]), atol=0, rtol=0)


def test_composite_prior_empty():
    composite = CompositePrior([])
    assert len(composite.priors) == 0

    param = torch.tensor([1.0, 2.0])
    grad = composite.grad(param, {})
    torch.testing.assert_close(grad, torch.tensor([0.0, 0.0]), atol=0, rtol=0)


def test_uniform_prior():
    prior = UniformPrior()
    params = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
    state = prior.initialize(params)
    assert state == {}

    param = torch.tensor([1.0, 2.0, 3.0])
    grad = prior.grad(param, {})
    torch.testing.assert_close(grad, torch.tensor([0.0, 0.0, 0.0]), atol=0, rtol=0)


def test_composite_prior_filters_uniform():
    uniform = UniformPrior()
    gaussian = GaussianPrior(localization=1.0)

    composite = CompositePrior([uniform, gaussian, UniformPrior()])
    assert len(composite.priors) == 1
    assert composite.priors[0] is gaussian

    composite = CompositePrior([UniformPrior(), UniformPrior()])
    assert len(composite.priors) == 0
