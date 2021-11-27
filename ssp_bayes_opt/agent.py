import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import warnings

# import GP modules
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor 

from . import ssp
from . import blr

import functools

class Agent:
    def __init__(self):
        pass

    def eval(self, xs):
        pass

    def update(self, x_t, y_t, sigma_t):
        pass

    def acquisition_func(self):
        pass

class GPAgent:
    def __init__(self, init_xs, init_ys):

        # Store observations
        self.xs = init_xs
        self.ys = init_ys
        # create the gp
        ## TODO instantiate scikitlearn regressor.
        self.gp = GaussianProcessRegressor(
                kernel=Matern(nu=2.5),
                alpha=1e-6,
                normalize_y=True,
                n_restarts_optimizer=5,
                random_state=None,
                )

        # fit to the initial values
        self.gp.fit(self.xs, self.ys)
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)

        self._params = self.gp.get_params()

    def eval(self, xs):
        mu, std = self.gp.predict(xs, return_std=True)
        var = std**2
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi 

    def update(self, x_t, y_t, sigma_t):
        self.xs = np.vstack((self.xs, x_t))
        self.ys = np.vstack((self.ys, y_t))
        self.gamma_t = self.gamma_t + sigma_t
    
        self.gp.fit(self.xs, self.ys)
        
        # Reset the parameters after an update.
        self.gp.set_params(**(self._params))

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

class SSPAgent:
    def __init__(self, init_xs, init_ys, n_scales=3, n_rotates=2, scale_min=0.8, scale_max=3.4):
  
        (num_pts, data_dim) = init_xs.shape

        # Create the simplex.
        self.ptrs, K_scale_rotates = ssp.HexagonalBasis(dim=data_dim)
#         self.ptrs, K_scale_rotates = ssp.RandomBasis(dim=data_dim, d=128)
        self.ptrs = np.vstack(self.ptrs)
        self.ssp_dim = self.ptrs.shape[1]

        # Optimize the length scales
        self.length_scale = self._optimize_lengthscale(init_xs, init_ys)
        print('Selected Lengthscale: ', self.length_scale)

        # Encode the initial sample points 
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_dim)

        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)

        # Cache for the input xs.
        self.phis = None

    ### end __init__

    def encode(self, xs):
        return self._encode(self.ptrs, xs, length_scale=self.length_scale)

    def _optimize_lengthscale(self, init_xs, init_ys):

        ls_0 = 20. * np.ones((init_xs.shape[1],))

        def min_func(length_scale):
            init_phis = self._encode(self.ptrs, init_xs, np.abs(length_scale))
            b = blr.BayesianLinearRegression(self.ssp_dim)
            b.update(init_phis, init_ys)
            mu, var = b.predict(init_phis)
            
            diff = init_ys.flatten() - mu.flatten()
#             err = np.sum(np.divide(np.power(diff, 2), var**2))
            err = np.sum(np.power(diff, 2))
            return err
        ### end min_func

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B')
        return np.abs(retval.x)


    def eval(self, xs):
        if self.phis is None:
            self.phis = self.encode(xs)
        ### end if
        mu, var = self.blr.predict(self.phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return mu, var, phi

    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        def optim_func(x, m=self.blr.m,
                       sigma=self.blr.S,
                       gamma=self.gamma_t,
                       beta_inv=self.blr.beta,
                       ptrs = self.ptrs,
                       ):
#             ptr = self.encode(x.reshape(1,-1))
            ptr = ssp.vector_encode(ptrs, x.reshape(1,-1))
            val = ptr @ m
            mi = np.sqrt(gamma + beta_inv + ptr @ sigma @ ptr.T) - np.sqrt(gamma)
            return -(val + mi).flatten()
        ### end optim_func

        def jac_func(x, m=self.blr.m,
                     sigma=self.blr.S,
                     gamma=self.gamma_t,
                     beta_inv=self.blr.beta,
                     ptrs=self.ptrs,
                     ):
#             ptr = self.encode(x.reshape(1,-1))
            ptr = ssp.vector_encode(ptrs, x.reshape(1,-1))
            sqr = (ptr @ sigma @ ptr.T) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sigma @ ptr.T / scale)
            return retval
        ### end gradient
        return optim_func, None #jac_func 
    ### end select_optimal

    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float):
        '''
        Updates the state of the Bayesian Linear Regression.
        '''
    
        x_val = x_t
        y_val = y_t
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])
        ### end if
    
        # Update BLR
        # TODO: use vector encode
#         ptr = ssp.vector_encode(ptrs, x.reshape(1,-1))
        phi = self.encode(x_val)
        self.blr.update(phi, y_val)
        
        # Update gamma
        self.gamma_t = self.gamma_t + sigma_t


    def _encode(self, ptrs, x, length_scale=None):
        (num_pts, x_dim) = x.shape

        outputs = np.zeros((num_pts, self.ssp_dim))

        for i in range(num_pts):
            vs = [ssp.encode(p,x[i,p_idx] / length_scale[p_idx]) for p_idx, p in enumerate(ptrs)]
            outputs[i,:] = functools.reduce(ssp.bind, vs)
        ### end for
        return outputs

