import numpy as np
import pytest

from ssp_bayes_opt.agents.kernels.sinc_kernel import SincKernel, d_sinc


def test_d_sinc_finite_difference():
    xs = np.linspace(-1, 1, 10000)
    d_sinc_est = np.divide(np.diff(np.sinc(xs)), np.diff(xs))
    assert np.allclose(d_sinc_est, d_sinc(xs)[1:], atol=1e-3)


def test_kernel_shape():
    k = SincKernel()
    xs = np.random.default_rng(0).random((100, 2))
    ys = np.random.default_rng(1).random((40, 2))
    K = k(xs, ys)
    assert K.shape == (100, 40)


def test_kernel_symmetry():
    k = SincKernel()
    xs = np.random.default_rng(2).random((50, 2))
    K = k(xs, xs)
    assert np.allclose(K, K.T)


def test_kernel_diagonal_ones():
    k = SincKernel()
    xs = np.random.default_rng(3).random((50, 2))
    K = k(xs, xs)
    assert np.allclose(np.diag(K), 1.0)


def test_kernel_gradient_shape():
    k = SincKernel()
    xs = np.array([[0.0, 0.0], [0.5, 0.5]])
    K, K_grad = k(xs, eval_gradient=True)
    assert K_grad.shape == (2, 2, 1)


def test_kernel_gradient_values():
    k = SincKernel()
    xs = np.array([[0.0, 0.0], [0.5, 0.5]])
    K, K_grad = k(xs, eval_gradient=True)
    dists = xs[:, None, :] - xs[None, :, :]
    sinc_mat = np.sinc(dists)
    d_sinc_mat = d_sinc(dists)
    expected = (d_sinc_mat[:, :, 0] * sinc_mat[:, :, 1] * (-dists[:, :, 0])
                + sinc_mat[:, :, 0] * d_sinc_mat[:, :, 1] * (-dists[:, :, 1]))
    assert np.allclose(expected.flatten(), K_grad.flatten())


def test_kernel_in_gp_classifier():
    from sklearn.datasets import load_iris
    from sklearn.gaussian_process import GaussianProcessClassifier
    from sklearn.gaussian_process.kernels import RBF

    X, y = load_iris(return_X_y=True)
    sinc_score = GaussianProcessClassifier(
        kernel=SincKernel(), random_state=0).fit(X, y).score(X, y)
    rbf_score = GaussianProcessClassifier(
        kernel=1.0 * RBF(1.0), random_state=0).fit(X, y).score(X, y)
    assert np.allclose(sinc_score, rbf_score, atol=2e-2), (
        f'RBF score {rbf_score:.3f} vs sinc score {sinc_score:.3f}')
