import torch
from devinterp.optim.metrics import Metrics


def test_add_sum_squared_and_sqrt_norms():
    m = Metrics()
    m.add_sum_squared_(
        scaled_grad=torch.tensor([3.0, 4.0]),
        unscaled_grad=torch.tensor([3.0, 4.0]),
        localization=torch.tensor([1.0]),
        weight_decay=torch.tensor([2.0]),
        noise=torch.tensor([0.0, 5.0]),
        distance=torch.tensor([6.0, 8.0]),
    )
    m.sqrt_norms_()

    torch.testing.assert_close(m.scaled_grad, torch.tensor([5.0]), atol=0, rtol=0)
    torch.testing.assert_close(m.unscaled_grad, torch.tensor([5.0]), atol=0, rtol=0)
    torch.testing.assert_close(m.localization, torch.tensor([1.0]), atol=0, rtol=0)
    torch.testing.assert_close(m.weight_decay, torch.tensor([2.0]), atol=0, rtol=0)
    torch.testing.assert_close(m.noise, torch.tensor([5.0]), atol=0, rtol=0)
    torch.testing.assert_close(m.distance, torch.tensor([10.0]), atol=0, rtol=0)


def test_accumulates_across_calls():
    m = Metrics()
    m.add_sum_squared_(
        scaled_grad=torch.tensor([3.0]),
        unscaled_grad=torch.tensor([5.0]),
        localization=torch.tensor([6.0]),
        weight_decay=torch.tensor([1.0]),
        noise=torch.tensor([2.0]),
        distance=torch.tensor([3.0]),
    )
    m.add_sum_squared_(
        scaled_grad=torch.tensor([4.0]),
        unscaled_grad=torch.tensor([12.0]),
        localization=torch.tensor([8.0]),
        weight_decay=torch.tensor([1.0]),
        noise=torch.tensor([2.0]),
        distance=torch.tensor([4.0]),
    )
    m.add_dot_products_(
        scaled_grad=torch.tensor([1.0, 2.0]),
        prior=torch.tensor([3.0, 4.0]),
        noise=torch.tensor([5.0, 6.0]),
    )
    m.add_dot_products_(
        scaled_grad=torch.tensor([1.0]),
        prior=torch.tensor([1.0]),
        noise=torch.tensor([1.0]),
    )
    m.sqrt_norms_()

    # sqrt(3² + 4²) = 5
    torch.testing.assert_close(m.scaled_grad, torch.tensor([5.0]), atol=0, rtol=0)
    # sqrt(5² + 12²) = 13
    torch.testing.assert_close(m.unscaled_grad, torch.tensor([13.0]), atol=0, rtol=0)
    # sqrt(6² + 8²) = 10
    torch.testing.assert_close(m.localization, torch.tensor([10.0]), atol=0, rtol=0)
    # sqrt(1² + 1²) = √2
    torch.testing.assert_close(
        m.weight_decay, torch.tensor([2.0]).sqrt(), atol=0, rtol=0
    )
    # sqrt(2² + 2²) = 2√2
    torch.testing.assert_close(m.noise, torch.tensor([8.0]).sqrt(), atol=0, rtol=0)
    # sqrt(3² + 4²) = 5
    torch.testing.assert_close(m.distance, torch.tensor([5.0]), atol=0, rtol=0)
    # dot_grad_prior: (1*3 + 2*4) + (1*1) = 11 + 1 = 12
    torch.testing.assert_close(m.dot_grad_prior, torch.tensor([12.0]), atol=0, rtol=0)
    # dot_grad_noise: (1*5 + 2*6) + (1*1) = 17 + 1 = 18
    torch.testing.assert_close(m.dot_grad_noise, torch.tensor([18.0]), atol=0, rtol=0)
    # dot_prior_noise: (3*5 + 4*6) + (1*1) = 39 + 1 = 40
    torch.testing.assert_close(m.dot_prior_noise, torch.tensor([40.0]), atol=0, rtol=0)


def test_zero():
    m = Metrics()
    m.scaled_grad = torch.tensor([7.0])
    m.unscaled_grad = torch.tensor([6.0])
    m.localization = torch.tensor([3.0])
    m.weight_decay = torch.tensor([2.0])
    m.noise = torch.tensor([1.0])
    m.distance = torch.tensor([5.0])
    m.dot_grad_prior = torch.tensor([0.5])
    m.dot_grad_noise = torch.tensor([0.3])
    m.dot_prior_noise = torch.tensor([0.1])
    m.numel = 42

    m.zero_()

    z = torch.tensor([0.0])
    torch.testing.assert_close(m.scaled_grad, z, atol=0, rtol=0)
    torch.testing.assert_close(m.unscaled_grad, z, atol=0, rtol=0)
    torch.testing.assert_close(m.localization, z, atol=0, rtol=0)
    torch.testing.assert_close(m.weight_decay, z, atol=0, rtol=0)
    torch.testing.assert_close(m.noise, z, atol=0, rtol=0)
    torch.testing.assert_close(m.distance, z, atol=0, rtol=0)
    torch.testing.assert_close(m.dot_grad_prior, z, atol=0, rtol=0)
    torch.testing.assert_close(m.dot_grad_noise, z, atol=0, rtol=0)
    torch.testing.assert_close(m.dot_prior_noise, z, atol=0, rtol=0)
    assert m.numel == 0


def test_to_returns_copy():
    m = Metrics(scaled_grad=torch.tensor([3.0]), numel=5)
    m2 = m.to("cpu")

    assert m2 is not m
    assert m2.numel == 5
    torch.testing.assert_close(m2.scaled_grad, torch.tensor([3.0]), atol=0, rtol=0)


def test_prior_combines_localization_and_weight_decay():
    m = Metrics()
    m.localization = torch.tensor([3.0])
    m.weight_decay = torch.tensor([4.0])

    # sqrt(3² + 4²) = 5
    torch.testing.assert_close(m.prior, torch.tensor([5.0]), atol=0, rtol=0)


def test_dot_products():
    m = Metrics()
    sg = torch.tensor([1.0, 2.0, 3.0])
    prior = torch.tensor([4.0, 5.0, 6.0])
    noise = torch.tensor([7.0, 8.0, 9.0])

    m.add_dot_products_(scaled_grad=sg, prior=prior, noise=noise)

    # 1*4 + 2*5 + 3*6 = 32
    torch.testing.assert_close(m.dot_grad_prior, torch.tensor([32.0]), atol=0, rtol=0)
    # 1*7 + 2*8 + 3*9 = 50
    torch.testing.assert_close(m.dot_grad_noise, torch.tensor([50.0]), atol=0, rtol=0)
    # 4*7 + 5*8 + 6*9 = 122
    torch.testing.assert_close(m.dot_prior_noise, torch.tensor([122.0]), atol=0, rtol=0)


def test_aggregate_norms_and_dots():
    m1 = Metrics(
        scaled_grad=torch.tensor([3.0]),
        unscaled_grad=torch.tensor([3.0]),
        localization=torch.tensor([0.0]),
        weight_decay=torch.tensor([0.0]),
        noise=torch.tensor([0.0]),
        distance=torch.tensor([0.0]),
        dot_grad_prior=torch.tensor([10.0]),
        dot_grad_noise=torch.tensor([0.0]),
        dot_prior_noise=torch.tensor([0.0]),
        numel=5,
    )
    m2 = Metrics(
        scaled_grad=torch.tensor([4.0]),
        unscaled_grad=torch.tensor([4.0]),
        localization=torch.tensor([0.0]),
        weight_decay=torch.tensor([0.0]),
        noise=torch.tensor([0.0]),
        distance=torch.tensor([0.0]),
        dot_grad_prior=torch.tensor([20.0]),
        dot_grad_noise=torch.tensor([0.0]),
        dot_prior_noise=torch.tensor([0.0]),
        numel=3,
    )
    result = Metrics.aggregate([m1, m2])

    # Norms: sqrt(3² + 4²) = 5
    torch.testing.assert_close(result.scaled_grad, torch.tensor([5.0]), atol=0, rtol=0)
    # Dots: additive
    torch.testing.assert_close(
        result.dot_grad_prior, torch.tensor([30.0]), atol=0, rtol=0
    )
    assert result.numel == 8
