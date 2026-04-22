import numpy as np
import warnings
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor

from .agent import Agent
from .kernels import SincKernel


class GPAgent(Agent):
    """Gaussian Process-based BO agent with Mutual Information acquisition.

    When updating=False the kernel hyperparameters are fitted once using
    20-restart optimisation and then held fixed. When updating=True the kernel
    is used as-is (no initial fitting) but is re-fitted to all data after each
    new observation.

    Parameters
    ----------
    init_xs : np.ndarray, shape (n, d)
    init_ys : np.ndarray, shape (n, 1)
    kernel_type : {'sinc', 'matern'}
    updating : bool
        If True, re-fit the GP after every new observation.
    gamma_c : float or Callable
        Scaling factor for the MI gamma accumulator.
    beta_ucb : float
        Variance weight in the acquisition function.
    """

    def __init__(self, init_xs, init_ys, kernel_type='sinc',
                 updating=True, gamma_c=1.0, beta_ucb=np.log(2 / 1e-6),
                 **kwargs):
        super().__init__()
        self.xs = init_xs
        self.ys = init_ys

        if updating:
            kern = Matern(nu=2.5) if kernel_type == 'matern' else SincKernel()
        else:
            if kernel_type == 'matern':
                fit_kern = Matern(nu=2.5)
            else:
                fit_kern = SincKernel(
                    length_scale_bounds=(1 / np.sqrt(init_xs.shape[0] + 1), 1e5))
            fit_gp = GaussianProcessRegressor(
                kernel=fit_kern, alpha=1e-6, normalize_y=True,
                n_restarts_optimizer=20, random_state=0)
            fit_gp.fit(self.xs, self.ys)
            if kernel_type == 'matern':
                kern = Matern(nu=2.5, length_scale=np.exp(fit_gp.kernel_.theta),
                              length_scale_bounds='fixed')
            else:
                kern = SincKernel(length_scale=np.exp(fit_gp.kernel_.theta),
                                  length_scale_bounds='fixed')

        self.gp = GaussianProcessRegressor(
            kernel=kern, alpha=1e-6, normalize_y=True,
            n_restarts_optimizer=5, random_state=None)
        self.gp.fit(self.xs, self.ys)

        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.sqrt_alpha = beta_ucb

    def eval(self, xs):
        mu, std = self.gp.predict(xs, return_std=True)
        var = std ** 2
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi

    def update(self, x_t, y_t, sigma_t, step_num=0):
        self.xs = np.vstack((self.xs, x_t))
        self.ys = np.vstack((self.ys, y_t))

        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c * sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number or callable'
            print(msg)
            raise RuntimeError(msg)

        # REVIEW: O(n^3) GP refit on every update; expensive for large n.
        self.gp.fit(self.xs, self.ys)

    def acquisition_func(self):
        def min_func(x, gp=self.gp, gamma_t=self.gamma_t,
                     sqrt_alpha=self.sqrt_alpha):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu, std = gp.predict(x.reshape([1, -1]), return_std=True)
                var = std ** 2
                phi = sqrt_alpha * (np.sqrt(var + gamma_t) - np.sqrt(gamma_t))
                return -(mu + phi).flatten()

        return min_func, None


class GPUCBAgent(GPAgent):
    """GP agent using Upper Confidence Bound (UCB) acquisition.

    UCB: mu(x) + sqrt(beta) * sigma(x), where beta=beta_ucb.
    """

    def __init__(self, init_xs, init_ys, kernel_type='sinc',
                 updating=True, gamma_c=1.0, beta_ucb=np.log(2 / 1e-6),
                 **kwargs):
        super().__init__(init_xs, init_ys, kernel_type=kernel_type,
                         updating=updating, gamma_c=gamma_c,
                         beta_ucb=beta_ucb, **kwargs)
        self.beta_ucb = beta_ucb

    def acquisition_func(self):
        def min_func(x, gp=self.gp, beta_ucb=self.beta_ucb):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu, std = gp.predict(x.reshape([1, -1]), return_std=True)
                phi = np.sqrt(beta_ucb) * std
                return -(mu + phi).flatten()

        return min_func, None
