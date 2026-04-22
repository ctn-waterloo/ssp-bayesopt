import numpy as np
import pytest

from ssp_bayes_opt.agents import GPAgent, GPUCBAgent

BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])
N_INIT = 10


def branin_hoo(x):
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6., 10., 1 / (8 * np.pi)
    return -(a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
             + s * (1 - t) * np.cos(x[:, 0]) + s)


@pytest.fixture
def init_data():
    rng = np.random.default_rng(42)
    xs = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(N_INIT, 2))
    ys = branin_hoo(xs).reshape(-1, 1)
    return xs, ys


def test_static_gp_init(init_data):
    xs, ys = init_data
    agt = GPAgent(xs, ys, kernel_type='matern', updating=False)
    assert hasattr(agt.gp, 'kernel_')
    ls = agt.gp.kernel_.length_scale
    assert np.all(np.isfinite(ls))


def test_gp_updating_init(init_data):
    xs, ys = init_data
    agt = GPAgent(xs, ys, kernel_type='matern', updating=True)
    mu, var, phi = agt.eval(xs)
    assert mu.shape[0] == N_INIT
    assert np.all(var >= 0)


def test_gp_ucb_init(init_data):
    xs, ys = init_data
    agt = GPUCBAgent(xs, ys, kernel_type='matern', updating=False)
    min_func, jac = agt.acquisition_func()
    assert callable(min_func)
    assert jac is None


def test_gp_update(init_data):
    xs, ys = init_data
    agt = GPAgent(xs, ys, kernel_type='matern', updating=True)
    rng = np.random.default_rng(99)
    x_new = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(1, 2))
    y_new = branin_hoo(x_new).reshape(1, 1)
    mu, var, _ = agt.eval(x_new)
    agt.update(x_new, y_new, float(var.flat[0]))
    assert agt.xs.shape[0] == N_INIT + 1
