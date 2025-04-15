import numpy as np
import warnings
# import GP modules
from sklearn.gaussian_process.kernels import Matern, RBF
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.preprocessing import StandardScaler 

from .agent import Agent

from ..util import DiscretizedFunction


class MIAcquisitionFunc:
    def __init__(self, sqrt_alpha):
        self.gamma_t = 0
        self.sqrt_alpha = sqrt_alpha

    def __call__(self, mean, var):
        return mean + self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))

class DiscretizedDomainAgent(Agent):
    def __init__(self, init_xs, init_ys,
            bounds=None,
            gamma_c=1.0,
            beta_ucb=np.log(2/1e-6),
            **kwargs):
        super().__init__()
        # Store observations
        self.xs = init_xs
        self.ys = init_ys
        # create the gp
        
        
        ## Create the GP to use during optimization.
        ## fit to the initial values
#         fit_kern = Matern(nu=2.5) 
#         self.bins_per_dim = kwargs.get('bins_per_dim', None)
        self.lenscale = kwargs['length_scale']
        if self.lenscale is None or self.lenscale < 0:
            fit_kern = Matern(
                    nu=2.5,
                    length_scale=np.ones((self.xs.shape[1],)),
                    length_scale_bounds=(1e-5,1e5),
            ) 
            fit_gp = GaussianProcessRegressor(
                    kernel=fit_kern,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=20,
                    random_state=0,
                    )
            fit_gp.fit(self.xs, self.ys)
            self.lenscale = np.abs(fit_gp.kernel_.theta)
        ### end if
#         assert np.any(self.lenscale > 0), f'Error: Negative length scale, {self.lenscale}, {fit_gp.kernel_.theta}, {fit_kern.length_scale}.'
#         assert self.bins_per_dim >= 10, f'Error: Choosing improbably small number of bins per dimension: {self.bins_per_dim}.'

        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.sqrt_alpha = beta_ucb
        self.disc_func = DiscretizedFunction(
                bounds, 
                self.lenscale,
#                 self.bins_per_dim,
                init_sum_f = np.mean(self.ys),
                init_sum_f2 = np.var(self.ys) + np.mean(self.ys)**2, #np.sum(self.ys**2),
                pseudo_counts = 1, #len(self.ys),
                )
        self.acq_func = MIAcquisitionFunc(sqrt_alpha=self.sqrt_alpha)
    ### end __init__

    def eval(self, xs):
        mu, var = self.disc_func.predict(xs)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi 

    def update(self, x_t, y_t, sigma_t, step_num=0):
        self.xs = np.vstack((self.xs, x_t))
        self.ys = np.vstack((self.ys, y_t))
        # self.gamma_t = self.gamma_t + sigma_t
        
        # Update gamma
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)

        self.acq_func.gamma_t = np.copy(self.gamma_t)
        self.disc_func.update(x_t, y_t)
    ### end update

    def sample(self):
        x, val = self.disc_func.sample(self.acq_func)
        soln = lambda : None
        soln.x = x
        soln.fun = -val
        return soln

    def acquisition_func(self):
        def min_func(x,
                     func=self.disc_func,
                     gamma_t=self.gamma_t,
                     sqrt_alpha=self.sqrt_alpha):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu, var = func.predict(x.reshape([1,-1]))
                phi = sqrt_alpha * (np.sqrt(var + gamma_t) - np.sqrt(gamma_t))
                return -(mu + phi).flatten()

        return min_func, None
