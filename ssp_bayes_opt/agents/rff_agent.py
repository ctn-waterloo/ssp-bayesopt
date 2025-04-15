import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, RBF
import scipy
from .. import blr

from .agent import Agent

class RFFAgent(Agent):
    def __init__(self, init_xs, init_ys,
                 ssp_dim,
                 kernel_type='matern',
                 gamma_c=1.0,
                 beta_ucb=np.log(2/1e-6),
                 var_decay=0.,
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
                            1/np.sqrt(init_xs.shape[0]+1),
                            1e5)
                        )
            kernel_nu = 2.5
        elif kernel_type == 'rbf':
            k_cov = RBF(1, length_scale_bounds=(
                            1/np.sqrt(init_xs.shape[0]+1),
                            1e5)
                        )
            kernel_nu = np.inf
        else:
            raise NotImplementedError
        fit_gp = GaussianProcessRegressor(
            kernel=k_scaling * k_cov,
            alpha=1e-6,
            normalize_y=True,
            n_restarts_optimizer=20,
            random_state=0,
        )
        fit_gp.fit(self.init_xs, self.init_ys)
        lengthscales = fit_gp.kernel_.k2.length_scale
        scaling = fit_gp.kernel_.k1.constant_value


        #From https://github.com/michaelosthege/pyrff/blob/8f5c3b57f5a74a449a55c01e3b2f214792f8a6ea/pyrff/rff.py
        # see Bradford 2018, equation 26
        alpha = scaling ** 2
        # just to avoid re-computing it all the time:
        self.sqrt_2_alpha_over_m = np.sqrt(2 * alpha / self.dim)
        # construct function to sample p(w) (see [Bradford, 2018], equations 27 and 28)
        if np.isinf(kernel_nu):
            def p_w(size: tuple) -> np.ndarray:
                return np.random.normal(loc=0, scale=1 / lengthscales, size=size)
        else:
            def p_w(size: tuple) -> np.ndarray:
                return scipy.stats.t.rvs(loc=0, scale=1 / lengthscales, df=kernel_nu, size=size)
        self.W = p_w(size=(self.dim, data_dim))
        self.B = np.random.uniform(0, 2 * np.pi, size=(self.dim, 1))

        # Encode the initial sample points
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.dim)
        self.blr.update(init_phis, np.array(init_ys))
        self.constraint_ssp = np.zeros_like(self.blr.m)

        # Acq. fun params
        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.var_weight = beta_ucb
        self.var_decay = var_decay
    ### end __init__

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.var_weight * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi

    def initial_guess(self):
        '''
        The initial guess for optimizing the acquisition function.
        '''
        # Return an initial guess from either the distribution or
        # From the approximate solution of dot(m,x) + x^T Sigma x
        return self.blr.sample()

    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''

        optim_norm_margin = 4

        def min_func(x, m=self.blr.m,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta,
                        norm_margin=optim_norm_margin):

            phi = self.encode(x).T
            val = phi.T @ m
            mi = self.var_weight * np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(x, m=self.blr.m,
                      sigma=self.blr.S,
                      gamma=self.gamma_t, # don't have right now
                      beta_inv=1/self.blr.beta):
            x = np.atleast_2d(x)
            Wx_plus_B = np.dot(self.W, x.T) + self.B
            phi_x = self.sqrt_2_alpha_over_m * np.cos(Wx_plus_B)  # shape (D,)
            sin_Wx_plus_B = np.sin(Wx_plus_B)

            mu_x = m.T @ phi_x
            sigma_phi = sigma @ phi_x

            sigma_sq_x = beta_inv + phi_x.T @ sigma_phi
            sigma_x = np.sqrt(sigma_sq_x)

            grad_mu = - self.sqrt_2_alpha_over_m * ((m * sin_Wx_plus_B).T @ self.W)
            grad_sigma = - (self.sqrt_2_alpha_over_m / sigma_x) * ((sigma_phi * sin_Wx_plus_B).T @ self.W)

            retval = grad_mu + self.var_weight * grad_sigma
            return -retval.flatten()


        return min_func, gradient



    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float, step_num=0):
        '''
        Updates the state of the Bayesian Linear Regression.
        '''
    
        x_val = x_t
        y_val = y_t
#         y_val = self.transformer.transform(y_t)
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])
        ### end if

        # Update BLR
        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)

        # Update var_weight
        if isinstance(self.var_decay, (int, float)):
            self.var_weight = self.var_weight + self.var_decay
        elif callable(self.var_decay):
            self.var_weight = self.var_weight + self.var_decay(step_num)
        else:
            msg = f'unable to use {self.var_weight}, expected number or callable'
            print(msg)
            raise RuntimeError(msg)
        
        # Update gamma
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
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

