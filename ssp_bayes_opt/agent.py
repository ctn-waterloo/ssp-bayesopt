import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# import GPy
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

class SSPAgent:
    def __init__(self, init_xs, init_ys, n_scales=1, n_rotates=1, scale_min=0.8, scale_max=3.4):
  
        self.num_restarts = 10
        (num_pts, data_dim) = init_xs.shape


        # Create the SSP axis vectors
#         self.ptrs = ssp.make_hex_unitary(data_dim, 
#                     n_scales=n_scales, n_rotates=n_rotates, 
#                     scale_min=scale_min, scale_max=scale_max)


        # Create the simplex.
        self.ptrs, K_scale_rotates = ssp.HexagonalBasis(dim=data_dim)
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

        ls_0 = 4. * np.ones((init_xs.shape[1],))

        def min_func(length_scale):
            init_phis = self._encode(self.ptrs, init_xs, np.abs(length_scale))
            W = np.linalg.pinv(init_phis) @ init_ys
            mu = np.dot(init_phis,W)
            diff = init_ys - mu.T
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

    def select_optimal(self):
        '''
        return objective_func, jacobian_func
        '''

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
        return optim_func, jac_func 

        # Optimize the function.
#         solns = []
#         vals = []
#         phis = []
#         for _ in range(self.num_restarts):
#             ## Create initial Guess
# #             try:
# #                 phi_init = np.random.multivariate_normal(self.blr.m.flatten(), self.blr.S).reshape(-1,1)
# #             except np.linalg.LinAlgError as e:
# #                 print(e)
# #                 phi_init = -self.blr.S_inv @ self.blr.m
#             phi_init = np.random.uniform(low=-5, high=5, size=(2,))
# 
# #             soln = minimize(optim_func, phi_init, jac=gradient, method='L-BFGS-B')
#             soln = minimize(optim_func, phi_init, method='L-BFGS-B', bounds=bounds)
#             vals.append(-soln.fun)
#             solns.append(np.copy(soln.x))
# 
#         best_val_idx = np.argmax(vals)
#         best_soln = solns[best_val_idx]
#         best_score = vals[best_val_idx]
# 
# 
#         return best_soln.reshape(1,-1), best_score, 0#best_phi
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

