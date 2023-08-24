import torch
import numpy as np
import torch_testing as tt

from model import to_token_sequence, from_token_sequence

from tasks import RegressionSequenceDistribution, SingletonTaskDistribution
from tasks import DiscreteTaskDistribution, GaussianTaskDistribution

from baselines import dmmse_predictor, ridge_predictor


# # # TEST MODELS MODULE (incomplete)


def test_to_token_sequence():
    xs = torch.asarray(
        [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    )
    ys = torch.asarray(
        [ [ [1],     [2],     [6],     ] ]
    )
    toks = to_token_sequence(xs, ys)
    expected_toks = torch.asarray(
      [ [ [ 0, 1, 2, 3, ]
        , [ 1, 0, 0, 0, ]
        , [ 0, 2, 3, 4, ]
        , [ 2, 0, 0, 0, ]
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    )
    tt.assert_equal(toks, expected_toks)
    

def test_from_token_sequence():
    toks = torch.asarray(
      [ [ [ 0, 1, 2, 3, ]
        , [ 1, 0, 0, 0, ]
        , [ 0, 2, 3, 4, ]
        , [ 2, 0, 0, 0, ]
        , [ 0, 6, 5, 4, ]
        , [ 6, 0, 0, 0, ]
        ] ]
    )
    xs, ys = from_token_sequence(toks)
    expected_xs = torch.asarray(
        [ [ [1,2,3], [2,3,4], [6,5,4], ] ]
    )
    expected_ys = torch.asarray(
        [ [ [1],     [2],     [6],     ] ]
    )
    tt.assert_equal(xs, expected_xs)
    tt.assert_equal(ys, expected_ys)


def test_to_from_token_sequence_roundtrip():
    xs = torch.randn(10, 16, 8)
    ys = torch.randn(10, 16, 1)
    toks = to_token_sequence(xs, ys)
    xs_, ys_ = from_token_sequence(toks)
    tt.assert_equal(xs, xs_)
    tt.assert_equal(ys, ys_)


# # # TEST TASKS MODULE


def test_singleton_task_distribution():
    task = torch.ones(4)
    distr = SingletonTaskDistribution(task)
    task_sample = torch.ones(10, 4)
    tt.assert_equal(distr.task, task)
    tt.assert_equal(distr.sample_tasks(1), torch.ones(1, 4))
    tt.assert_equal(distr.sample_tasks(10), torch.ones(10, 4))


def test_regression_data_generation_zero_function():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.zeros(4)),
        noise_variance=0,
    )
    _xs, ys = data.get_batch(num_examples=16, batch_size=64)
    tt.assert_equal(ys, torch.zeros(64, 16, 1))


def test_regression_data_generation_shape():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.ones(4)),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=16, batch_size=64)
    assert tuple(xs.shape) == (64, 16, 4)
    assert tuple(ys.shape) == (64, 16, 1)


def test_regression_data_generation_sum_function():
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(torch.ones(4)),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=16, batch_size=64)
    # check contents of batch: each y should be x0+x1+x2+x3
    tt.assert_equal(ys, xs.sum(axis=-1, keepdim=True))


def test_regression_data_generation_arange():
    B, K, D = 8, 8, 16
    task = torch.arange(D, dtype=torch.float32)
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(task),
        noise_variance=0,
    )
    xs, ys = data.get_batch(num_examples=K, batch_size=B)
    # one-by-one each y should be sum_i i*x_i
    for b in range(B):
        for k in range(K):
            y = ys[b, k, 0]
            x = xs[b, k, :]
            assert y - (task @ x) < 1e-4 # close enough?


def test_regression_data_generation_zero_plus_variance():
    B, K, D = 1024, 4096, 1
    task = torch.zeros(D)
    data = RegressionSequenceDistribution(
        task_distribution=SingletonTaskDistribution(task),
        noise_variance=4., # standard deviation 2
    )
    _xs, ys = data.get_batch(num_examples=K, batch_size=B)
    # the ys should be gaussian with variance 4 / stddev 2 
    sample_var, sample_mean = torch.var_mean(ys)
    assert abs(sample_mean - 0) < 1e-2
    assert abs(sample_var - 4.) < 1e-2
    # Note: the test uses Bessel's correction, but we actually know the mean,
    # so could just compute the sample variance without correction with known
    # mean zero and maybe this would fail with slightly lower probability.
    # Anyway, I expect this will be fine but we will see.


def test_discrete_task_distribution():
    D, M, N = 4, 16, 128
    distr = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    tasks = distr.sample_tasks(n=N)
    assert tuple(tasks.shape) == (N, D)
    for n in range(N):
        assert tasks[n] in distr.tasks


def test_gaussian_task_distribution():
    D, N = 64, 128
    distr = GaussianTaskDistribution(task_size=D)
    tasks = distr.sample_tasks(n=N)
    assert tuple(tasks.shape) == (N, D)


# # # TEST BASELINES MODULE


def test_dmsse_predictor_first_is_uniform():
    B, K, D, M, V = 256, 1, 4, 16, .25
    T = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    ws = T.tasks
    _, ws_hat = dmmse_predictor(
        xs,
        ys,
        prior=T,
        noise_variance=V,
        return_ws_hat=True,
    )
    # now for K=1 there is no context for the first predictor so it's just
    # going to be the uniform average over all the tasks
    tt.assert_allclose(ws_hat, ws.mean(axis=0).expand(B, K, D))


def test_dmsse_predictor_python_loops():
    B, K, D, M, V = 256, 2, 4, 8, .25
    T = DiscreteTaskDistribution(task_size=D, num_tasks=M)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    ws = T.tasks
    _, ws_hat = dmmse_predictor(
        xs,
        ys,
        prior=T,
        noise_variance=V,
        return_ws_hat=True,
    )
    # now do a straight-forward non-vectorised implementation of eqn 7 and
    # compare
    for b in range(B):
        for k in range(K):
            # compute denominator
            denom = 0.
            for l in range(M):
                # compute inner sum
                # note: there is no need for k-1 in the loop guard because
                # we are counting k in range(K) from 0, this k is actually
                # the 'number of contextual examples to consider' (0, 1, ...,
                # k-1)
                loss_sofar = 0.
                for j in range(k):
                    loss_sofar += (ys[b,j,0] - ws[l] @ xs[b,j]).square()
                denom += np.exp(-1/(2*V) * loss_sofar)
            # compute numerator
            numer = torch.zeros(D)
            for i in range(M):
                # compute inner sum
                loss_sofar = 0.
                for j in range(k):
                    loss_sofar += (ys[b,j,0] - ws[i] @ xs[b,j]).square()
                numer += np.exp(-1/(2*V) * loss_sofar) * ws[i]
            # combine
            wk_hat_expected = numer / denom
            # check
            wk_hat_actual = ws_hat[b, k]
            tt.assert_allclose(wk_hat_expected, wk_hat_actual, atol=1e-5)


def test_ridge_predictor_first_is_zero():
    B, K, D, V = 256, 1, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now for K=1 there is no context for the first predictor so it's just
    # going to be the prior average weight, i.e., 0
    tt.assert_allclose(ws_hat, torch.zeros(B, K, D))


def test_ridge_predictor_second_is_easy():
    B, K, D, V = 256, 2, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now for K=2 there is a single vector of context for the second
    # predictor so we can do a single low-dimensional update and check the
    # dimensions all check out (also here I use inv instead of solve)
    for b in range(B):
        k = 1
        x = xs[b, k-1, :]               # B K D, slice      -> D
        y = ys[b, k-1, 0]               # B K 1, slice      -> .
        xxT = x.outer(x)                # D outer D         -> D D
        LHS = xxT + V * torch.eye(D)    # D D + (. . * D D) -> D D
        inv = torch.linalg.inv(LHS)     # inv(D D)          -> D D
        RHS = x * y                     # D * .             -> D
        wk_hat_expected = inv @ RHS     # D D @ D           -> D
        # check
        wk_hat_actual = ws_hat[b, k, :] # B K D, slice      -> D
        tt.assert_allclose(wk_hat_expected, wk_hat_actual, atol=1e-5)


def test_ridge_predictor_python_loops():
    B, K, D, V = 256, 1, 4, .25
    T = GaussianTaskDistribution(task_size=D)
    xs, ys = RegressionSequenceDistribution(
        task_distribution=T,
        noise_variance=V,
    ).get_batch(
        num_examples=K,
        batch_size=B,
    )
    _, ws_hat = ridge_predictor(xs, ys, noise_variance=V, return_ws_hat=True)
    # now do a straight-forward non-vectorised implementation of eqn 8 and
    # compare
    for b in range(B):
        for k in range(K):
            # note: there is no need for k-1 in the indexing because we are
            # already counting k in range(K) from 0, thus this k is actually
            # the 'number of contextual examples to consider': 0, 1, ..., k-1
            Xk = xs[b, :k, :]
            yk = ys[b, :k, :]
            LHS = Xk.T @ Xk + V * torch.eye(D)
            RHS = Xk.T @ yk
            # wk = inv(LHS) @ RHS <--> wk = solve(LHS @ wk == RHS)
            wk_hat_expected = torch.linalg.solve(LHS, RHS).view(D)
            # check
            wk_hat_actual = ws_hat[b, k]
            tt.assert_allclose(wk_hat_expected, wk_hat_actual)


