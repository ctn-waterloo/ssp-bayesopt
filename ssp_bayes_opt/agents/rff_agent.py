import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
import scipy
from .. import blr

from .agent import Agent


class RFFAgent(Agent):
    """BO agent using Random Fourier Features (RFF) to approximate a GP kernel.

    Fits a scaled RBF or Matérn kernel to the initial data, then samples RFF
    weights to construct a fixed feature map. BayesianLinearRegression is used
    over this feature map for efficient inference.

    Reference: Bradford et al. (2018), equations 26-28.

    Parameters
    ----------
    init_xs : np.ndarray, shape (n, d)
    init_ys : np.ndarray, shape (n, 1)
    ssp_dim : int
        Number of RFF features (dimensionality of the feature map).
    kernel_type : {'rbf', 'matern'}
    gamma_c : float or Callable
    beta_ucb : float
    var_decay : float or Callable
        Added to beta_ucb after each update (note: additive, unlike SSPAgent's
        multiplicative decay).
    """

    def __init__(self, init_xs, init_ys, ssp_dim, kernel_type='rbf',
                 gamma_c=1.0, beta_ucb=np.log(2 / 1e-6), var_decay=0.,
                 **kwargs):
        super().__init__()

        (num_pts, data_dim) = init_xs.shape
        self.data_dim = data_dim
        self.init_xs = init_xs
        self.init_ys = init_ys
        self.dim = ssp_dim

        k_scaling = ConstantKernel(1, (0.1, 10))
        if kernel_type == 'matern':
            k_cov = Matern(nu=2.5, length_scale_bounds=(
                1 / np.sqrt(init_xs.shape[0] + 1), 1e5))
            kernel_nu = 2.5
        elif kernel_type == 'rbf':
            k_cov = RBF(1, length_scale_bounds=(
                1 / np.sqrt(init_xs.shape[0] + 1), 1e5))
            kernel_nu = np.inf
        else:
            raise NotImplementedError(f'kernel_type {kernel_type!r} not supported')
        fit_gp = GaussianProcessRegressor(
            kernel=k_scaling * k_cov, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=20, random_state=0)
        fit_gp.fit(self.init_xs, self.init_ys)
        lengthscales = fit_gp.kernel_.k2.length_scale
        scaling = fit_gp.kernel_.k1.constant_value
        print('Selected Lengthscale: ', lengthscales)

        alpha = scaling ** 2
        self.sqrt_2_alpha_over_m = np.sqrt(2 * alpha / self.dim)
        if np.isinf(kernel_nu):
            def p_w(size):
                return np.random.normal(loc=0, scale=1 / lengthscales, size=size)
        else:
            def p_w(size):
                return scipy.stats.t.rvs(
                    loc=0, scale=1 / lengthscales, df=kernel_nu, size=size)
        self.W = p_w(size=(self.dim, data_dim))
        self.B = np.random.uniform(0, 2 * np.pi, size=(self.dim, 1))

        init_phis = self.encode(init_xs)
        self.blr = blr.BayesianLinearRegression(self.dim)
        self.blr.update(init_phis, np.array(init_ys))
        self.constraint_ssp = np.zeros_like(self.blr.m)

        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.var_weight = beta_ucb
        self.var_decay = var_decay

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.var_weight * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi

    def initial_guess(self):
        """Return an initial RFF feature guess (sampled from BLR posterior)."""
        return self.blr.sample()

    def acquisition_func(self):
        """Return (min_func, gradient) in x space."""
        optim_norm_margin = 4

        def min_func(x, m=self.blr.m, sigma=self.blr.S,
                     gamma=self.gamma_t, beta_inv=1 / self.blr.beta,
                     norm_margin=optim_norm_margin):
            phi = self.encode(x).T
            val = phi.T @ m
            mi = self.var_weight * np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()

        def gradient(x, m=self.blr.m, sigma=self.blr.S,
                     gamma=self.gamma_t, beta_inv=1 / self.blr.beta):
            x = np.atleast_2d(x)
            Wx_plus_B = np.dot(self.W, x.T) + self.B
            phi_x = self.sqrt_2_alpha_over_m * np.cos(Wx_plus_B)
            sin_Wx_plus_B = np.sin(Wx_plus_B)
            sigma_phi = sigma @ phi_x
            sigma_sq_x = beta_inv + phi_x.T @ sigma_phi
            sigma_x = np.sqrt(sigma_sq_x)
            grad_mu = -self.sqrt_2_alpha_over_m * ((m * sin_Wx_plus_B).T @ self.W)
            grad_sigma = -(self.sqrt_2_alpha_over_m / sigma_x) * (
                (sigma_phi * sin_Wx_plus_B).T @ self.W)
            return -(grad_mu + self.var_weight * grad_sigma).flatten()

        return min_func, gradient

    def update(self, x_t: np.ndarray, y_t: np.ndarray, sigma_t: float, step_num=0):
        """Incorporate a new observation into the BLR posterior."""
        x_val = x_t
        y_val = y_t
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])

        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)

        if isinstance(self.var_decay, (int, float)):
            self.var_weight = self.var_weight + self.var_decay
        elif callable(self.var_decay):
            self.var_weight = self.var_weight + self.var_decay(step_num)
        else:
            msg = f'unable to use {self.var_weight}, expected number or callable'
            print(msg)
            raise RuntimeError(msg)

        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c * sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number or callable'
            print(msg)
            raise RuntimeError(msg)

    def encode(self, x):
        x = np.atleast_2d(x)
        phi = self.sqrt_2_alpha_over_m * np.cos(np.dot(self.W, x.T) + self.B).T
        return phi
