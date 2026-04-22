import numpy as np
import pytest

from ssp_bayes_opt.blr import BayesianLinearRegression


def test_blr_init():
    b = BayesianLinearRegression(10)
    assert b.m.shape == (10, 1)
    assert b.S.shape == (10, 10)
    assert b.input_dim == 10


def test_blr_update_changes_posterior():
    rng = np.random.default_rng(0)
    dim = 20
    b = BayesianLinearRegression(dim)
    m0, S0 = b.m.copy(), b.S.copy()

    xs = rng.normal(size=(5, dim))
    ys = rng.normal(size=(5, 1))
    b.update(xs, ys)

    assert not np.allclose(b.m, m0)
    assert not np.allclose(b.S, S0)


def test_blr_predict_shape():
    rng = np.random.default_rng(1)
    dim = 20
    b = BayesianLinearRegression(dim)
    train_x = rng.normal(size=(10, dim))
    train_y = rng.normal(size=(10, 1))
    b.update(train_x, train_y)

    test_x = rng.normal(size=(7, dim))
    mu, var = b.predict(test_x)
    assert mu.shape == (1, 7)  # mu is (1, n) from m.T @ phi.T
    assert var.shape == (7,)


def test_blr_predict_positive_variance():
    rng = np.random.default_rng(2)
    dim = 15
    b = BayesianLinearRegression(dim)
    train_x = rng.normal(size=(8, dim))
    train_y = rng.normal(size=(8, 1))
    b.update(train_x, train_y)

    test_x = rng.normal(size=(5, dim))
    _, var = b.predict(test_x)
    assert np.all(var > 0)


def test_blr_regression_accuracy():
    rng = np.random.default_rng(3)
    dim = 30
    true_w = rng.normal(size=(dim, 1))

    xs = rng.normal(size=(200, dim))
    ys = xs @ true_w + 0.01 * rng.normal(size=(200, 1))

    b = BayesianLinearRegression(dim)
    b.update(xs, ys)

    test_xs = rng.normal(size=(50, dim))
    test_ys = test_xs @ true_w
    mu, _ = b.predict(test_xs)
    assert np.mean((mu.reshape(-1) - test_ys.reshape(-1)) ** 2) < 0.1


def test_blr_sequential_updates_match_batch():
    """Sequential rank-1 updates should give the same posterior as one batch update.

    Both objects are initialised with an explicit beta so that the noise
    precision doesn't differ between the single-sample and full-batch estimates.
    """
    rng = np.random.default_rng(4)
    dim = 10
    xs = rng.normal(size=(6, dim))
    ys = rng.normal(size=(6, 1))
    beta = 1.0

    b_batch = BayesianLinearRegression(dim, beta=beta)
    b_batch.update(xs, ys)

    b_seq = BayesianLinearRegression(dim, beta=beta)
    for i in range(len(xs)):
        b_seq.update(xs[i:i+1], ys[i:i+1])

    assert np.allclose(b_batch.m, b_seq.m, atol=1e-6)
    assert np.allclose(b_batch.S, b_seq.S, atol=1e-6)
