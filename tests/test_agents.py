"""Pytest tests for each BO agent type.

All agents are tested in fast mode:
  - decoder_method='from-set' (avoids TensorFlow training)
  - explicit length_scale (avoids 20-restart GP hyperparameter fitting)
  - ssp_dim=51 (small enough to be fast)
"""
import numpy as np
import pytest

from ssp_bayes_opt import sspspace
from ssp_bayes_opt.agents import SSPAgent, GPAgent, GPUCBAgent, RFFAgent
from ssp_bayes_opt.agents.ssp_traj_agent import SSPTrajectoryAgent
from ssp_bayes_opt.agents.ssp_multi_agent import SSPMultiAgent

SSP_DIM = 51
BOUNDS = np.array([[-5.0, 5.0], [-5.0, 5.0]])
N_INIT = 8


def _init_data(rng=None, n=N_INIT, bounds=BOUNDS):
    if rng is None:
        rng = np.random.default_rng(0)
    xs = rng.uniform(bounds[:, 0], bounds[:, 1], size=(n, bounds.shape[0]))
    ys = (-(xs[:, 0] ** 2 + xs[:, 1] - 11) ** 2
          - (xs[:, 0] + xs[:, 1] ** 2 - 7) ** 2).reshape(-1, 1)
    return xs, ys


def _hex_space(bounds=BOUNDS):
    return sspspace.HexagonalSSPSpace(
        bounds.shape[0], ssp_dim=SSP_DIM,
        domain_bounds=bounds, length_scale=1.0, rng=0)


# ---------------------------------------------------------------------------
# SSPAgent
# ---------------------------------------------------------------------------

class TestSSPAgent:
    def setup_method(self):
        self.xs, self.ys = _init_data()
        self.space = _hex_space()
        self.agt = SSPAgent(
            self.xs, self.ys, self.space,
            decoder_method='from-set', length_scale=1.0)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == N_INIT
        assert var.shape[-1] == N_INIT

    def test_acquisition_func_callable(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)

    def test_update_does_not_crash(self):
        x_new, y_new = _init_data(n=1)
        mu, var, _ = self.agt.eval(x_new)
        self.agt.update(x_new, y_new, float(var.flat[0]), step_num=1)

    def test_decode_roundtrip(self):
        x = np.array([[1.0, -2.0]])
        phi = self.space.encode(x)
        x_rec = self.agt.decode(phi)
        assert x_rec.shape == x.shape

    def test_encode_shape(self):
        phi = self.agt.encode(self.xs)
        assert phi.shape == (N_INIT, self.agt.ssp_dim)


# ---------------------------------------------------------------------------
# GPAgent
# ---------------------------------------------------------------------------

class TestGPAgent:
    def setup_method(self):
        self.xs, self.ys = _init_data()
        self.agt = GPAgent(
            self.xs, self.ys, kernel_type='matern',
            updating=True, gamma_c=1.0)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == N_INIT

    def test_acquisition_func_callable(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)
        assert jac is None

    def test_update_does_not_crash(self):
        x_new, y_new = _init_data(n=1)
        mu, var, _ = self.agt.eval(x_new)
        self.agt.update(x_new, y_new, float(var.flat[0]), step_num=1)

    def test_eval_returns_nonneg_variance(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert np.all(var >= 0)


# ---------------------------------------------------------------------------
# GPUCBAgent
# ---------------------------------------------------------------------------

class TestGPUCBAgent:
    def setup_method(self):
        self.xs, self.ys = _init_data()
        self.agt = GPUCBAgent(
            self.xs, self.ys, kernel_type='matern',
            updating=True, gamma_c=1.0)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == N_INIT

    def test_acquisition_func_callable(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)


# ---------------------------------------------------------------------------
# RFFAgent
# ---------------------------------------------------------------------------

class TestRFFAgent:
    def setup_method(self):
        self.xs, self.ys = _init_data()
        self.agt = RFFAgent(
            self.xs, self.ys, ssp_dim=SSP_DIM,
            kernel_type='rbf', gamma_c=1.0)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == N_INIT

    def test_acquisition_func_and_gradient(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)
        assert callable(jac)
        x_test = self.xs[0]
        val = min_func(x_test)
        grad = jac(x_test)
        assert np.isfinite(val).all()
        assert grad.shape == x_test.shape

    def test_update_does_not_crash(self):
        x_new, y_new = _init_data(n=1)
        mu, var, _ = self.agt.eval(x_new)
        self.agt.update(x_new, y_new, float(var.flat[0]), step_num=1)

    def test_encode_shape(self):
        phi = self.agt.encode(self.xs)
        assert phi.shape == (N_INIT, self.agt.dim)  # RFFAgent uses self.dim


# ---------------------------------------------------------------------------
# SSPTrajectoryAgent
# ---------------------------------------------------------------------------

class TestSSPTrajectoryAgent:
    def setup_method(self):
        self.traj_len = 3
        self.x_dim = 2
        self.n_init = 5
        rng = np.random.default_rng(1)
        # xs shape: (n_init, traj_len, x_dim)
        raw_xs = rng.uniform(-5, 5, size=(self.n_init, self.traj_len, self.x_dim))
        self.xs = raw_xs.reshape(self.n_init, -1)
        self.ys = rng.uniform(-1, 0, size=(self.n_init, 1))

        self.agt = SSPTrajectoryAgent(
            self.xs, self.ys,
            x_dim=self.x_dim,
            traj_len=self.traj_len,
            domain_bounds=BOUNDS,
            decoder_method='from-set',
            length_scale=1.0,
            ssp_dim=SSP_DIM,
        )

    def test_encode_shape(self):
        phi = self.agt.encode(self.xs)
        assert phi.shape == (self.n_init, self.agt.ssp_dim)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == self.n_init

    def test_acquisition_func_callable(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)

    def test_update_does_not_crash(self):
        rng = np.random.default_rng(99)
        x_new = rng.uniform(-5, 5, size=(1, self.traj_len * self.x_dim))
        y_new = rng.uniform(-1, 0, size=(1, 1))
        mu, var, _ = self.agt.eval(x_new)
        self.agt.update(x_new, y_new, float(var.flat[0]), step_num=1)


# ---------------------------------------------------------------------------
# SSPMultiAgent
# ---------------------------------------------------------------------------

class TestSSPMultiAgent:
    def setup_method(self):
        self.n_agents = 2
        self.traj_len = 3
        self.x_dim = 2
        self.n_init = 5
        rng = np.random.default_rng(2)
        # xs shape expected: (n_init, traj_len * n_agents * x_dim)
        self.xs = rng.uniform(-5, 5, size=(self.n_init, self.traj_len * self.n_agents * self.x_dim))
        self.ys = rng.uniform(-1, 0, size=(self.n_init, 1))

        self.agt = SSPMultiAgent(
            self.xs, self.ys,
            n_agents=self.n_agents,
            x_dim=self.x_dim,
            traj_len=self.traj_len,
            domain_bounds=BOUNDS,
            decoder_method='from-set',
            length_scale=1.0,
            ssp_dim=SSP_DIM,
        )

    def test_encode_shape(self):
        phi = self.agt.encode(self.xs)
        assert phi.shape == (self.n_init, self.agt.ssp_dim)

    def test_eval_shape(self):
        mu, var, phi = self.agt.eval(self.xs)
        assert mu.shape[-1] == self.n_init

    def test_acquisition_func_callable(self):
        min_func, jac = self.agt.acquisition_func()
        assert callable(min_func)

    def test_update_does_not_crash(self):
        rng = np.random.default_rng(99)
        x_new = rng.uniform(-5, 5, size=(1, self.traj_len * self.n_agents * self.x_dim))
        y_new = rng.uniform(-1, 0, size=(1, 1))
        mu, var, _ = self.agt.eval(x_new)
        self.agt.update(x_new, y_new, float(var.flat[0]), step_num=1)
