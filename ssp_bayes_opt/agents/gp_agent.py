import numpy as np

# import GP modules
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.preprocessing import StandardScaler 

from .agent import Agent

class GPAgent(Agent):
    def __init__(self, init_xs, init_ys, gamm_c = 1.0, updating=True):
        super().__init__()
        # Store observations
        self.xs = init_xs
        self.ys = init_ys
        # create the gp

        ## Create the GP to use during optimization.
        if updating:
            kern = Matern(nu=2.5)
        else:
            ## fit to the initial values
            fit_gp = GaussianProcessRegressor(
                        kernel=Matern(nu=2.5),
                        alpha=1e-6,
                        normalize_y=True,
                        n_restarts_optimizer=5,
                        random_state=None,
                    )
            fit_gp.fit(self.xs, self.ys)
            kern = Matern(nu=2.5,
                          length_scale=np.exp(fit_gp.kernel_.theta),
                          length_scale_bounds='fixed')
        ### end if
        self.gp = GaussianProcessRegressor(
                    kernel=kern,
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=5,
                    random_state=None,
                )
        self.gp.fit(self.xs, self.ys)

        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
        self.gamma_c = gamma_c
    ### end __init__

    def eval(self, xs):
        mu, std = self.gp.predict(xs, return_std=True)
        var = std**2
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi 

    def update(self, x_t, y_t, sigma_t, step_num=0, info=None):
        self.xs = np.vstack((self.xs, x_t))
        self.ys = np.vstack((self.ys, y_t))
        #self.gamma_t = self.gamma_t + sigma_t
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)
    
        self.gp.fit(self.xs, self.ys)
    ### end update
        

    def acquisition_func(self):
        def min_func(x,
                     gp=self.gp,
                     gamma_t=self.gamma_t,
                     sqrt_alpha=self.sqrt_alpha):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                mu, std = gp.predict(x.reshape([1,-1]), return_std=True)
                var = std**2
                phi = sqrt_alpha * (np.sqrt(var + gamma_t) - np.sqrt(gamma_t))
                return -(mu + phi).flatten()

        return min_func, None
