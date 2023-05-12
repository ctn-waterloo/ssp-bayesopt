import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import warnings

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor 

from .. import sspspace 
from .. import blr
from .kernels import SincKernel


from .agent import Agent



class SSPDiscreteContextualAgent(Agent):
    def __init__(self, init_xs, init_ys, init_context: int, context_size: int, ssp_space=None, context_space = None,
                 decoder_method='network-optim',gamma_c=1., **kwargs):
        super().__init__()
  
        self.context_dim = 1
        self.context_size = context_size
        num_pts = init_xs.shape[0]
        self.data_dim = init_xs.shape[1] - self.context_dim

        if ssp_space is None:
            ssp_space = sspspace.HexagonalSSPSpace(
                                self.data_dim,
                                ssp_dim=151, 
                                n_rotates=5, 
                                n_scales=5, 
                                scale_min=2*np.pi/np.sqrt(6) - 0.5, 
                                scale_max=2*np.pi/np.sqrt(6) + 0.5, 
                                domain_bounds=None, 
                                length_scale=5,
            )
        if context_space is None:
            context_space = sspspace.SPSpace(
                                self.context_size,
                                ssp_space.ssp_dim
            )
        ### end if
        self.ssp_space = ssp_space
        self.context_space = context_space
        
        self.context = self.context_space.encode(init_context)
        self.context_raw = init_context

        # Optimize the length scales
        if not 'length_scale' in kwargs or kwargs.get('length_scale') < 0:
            ls = self._optimize_lengthscale(init_xs[:,:self.data_dim], init_ys)
            self.ssp_space.update_lengthscale(ls)
        else:
            self.ssp_space.update_lengthscale(kwargs.get('length_scale', 4))
            
        ### end if
        print('Selected Lengthscale: ', ssp_space.length_scale)
        
        

        # Encode the initial sample points 
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_space.ssp_dim)

        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
        
        if (decoder_method=='network') | (decoder_method=='network-optim'):
            self.ssp_space.train_decoder_net();
            self.init_samples=None
        else:
            self.init_samples = self.ssp_space.get_sample_pts_and_ssps(2**17,'length-scale')
        self.decoder_method = decoder_method
        self.gamma_c = gamma_c
    ### end __init__


    def _optimize_lengthscale(self, init_xs, init_ys):

        ## fit to the initial values
        fit_gp = GaussianProcessRegressor(
                    kernel=SincKernel(
                        length_scale_bounds=(
                            1/np.sqrt(init_xs.shape[0]+1), 
                            1e5)
                        ),
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=20,
                    random_state=0,
                )
        fit_gp.fit(init_xs, init_ys)
        lenscale = np.exp(fit_gp.kernel_.theta)
        return lenscale
    ### end _optimize_lengthscale

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return mu, var, phi

    def initial_guess(self):
        '''
        The initial guess for optimizing the acquisition function.
        '''
        # Return an initial guess from either the distribution or
        # From the approximate solution of dot(m,x) + x^T Sigma x
        x = self.blr.sample()
        return self.ssp_space.bind(self.ssp_space.invert(self.context), x.T).T


    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''

        optim_norm_margin = 4
        C = self.context_space.get_binding_matrix(self.context)
        m = C @ self.blr.m
        sigma= C.T @ self.blr.S @ C
        def min_func(phi, m=m,
                        sigma=sigma,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta,
                        norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if phi_norm > norm_margin:
                phi = norm_margin * phi / phi_norm
            ### end if
            val = phi.T @ m
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(phi, m=m,
                      sigma=sigma,
                      gamma=self.gamma_t,
                      beta_inv=1/self.blr.beta,
                      norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if phi_norm > norm_margin:
                phi = norm_margin * phi / phi_norm
            ### end if
            sig_phi = sigma @ phi
            sqr = (phi.T @ sig_phi ) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sig_phi / scale)
            return retval

        return min_func, gradient

    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float, step_num=0, info=None):
        '''
        Updates the state of the Bayesian Linear Regression.
        '''
    
        x_val = x_t
        y_val = y_t
        self.context = self.context_space.encode(info)
        self.context_raw = info
        
#         y_val = self.transformer.transform(y_t)
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])
        ### end if

        # Update BLR
        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)
        
        # Update gamma
        #self.gamma_t = self.gamma_t + sigma_t
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)

    def encode(self, x):
        if x.shape[1] == self.data_dim:
            return self.ssp_space.bind(self.ssp_space.encode(x), self.context)
        else:
            return self.ssp_space.bind(self.ssp_space.encode(x[:,:self.data_dim]), 
                                       self.context_space.encode(x[:,-1].astype(int)))
    
    
    def decode(self,ssp):
        action = self.ssp_space.decode(ssp,method=self.decoder_method,samples=self.init_samples)
        return np.concatenate([action, self.context_raw.reshape(-1)])



class SSPContinuousContextualAgent(Agent):
    def __init__(self, init_xs, init_ys, init_context, ssp_space=None, context_ssp_space = None,
                 decoder_method='network-optim',gamma_c=1., **kwargs):
        super().__init__()
  
        self.context_dim = len(init_context)
        num_pts = init_xs.shape[0]
        self.data_dim = init_xs.shape[1] - self.context_dim

        if ssp_space is None:
            ssp_space = sspspace.HexagonalSSPSpace(
                                self.data_dim,
                                ssp_dim=151, 
                                n_rotates=5, 
                                n_scales=5, 
                                scale_min=2*np.pi/np.sqrt(6) - 0.5, 
                                scale_max=2*np.pi/np.sqrt(6) + 0.5, 
                                domain_bounds=None, 
                                length_scale=5,
            )
        if context_ssp_space is None:
            context_ssp_space = sspspace.RandomSSPSpace(
                                self.context_dim,
                                ssp_dim=151,
                                domain_bounds=None, 
                                length_scale=5,
            )
        ### end if
        self.ssp_space = ssp_space
        self.context_ssp_space = context_ssp_space
        
        self.context = self.context_ssp_space.encode(init_context)
        self.context_raw = init_context

        # Optimize the length scales
        if not 'length_scale' in kwargs or kwargs.get('length_scale') < 0:
            ls = self._optimize_lengthscale(init_xs, init_ys)
            self.ssp_space.update_lengthscale(ls)
            self.context_ssp_space.update_lengthscale(ls)
        else:
            self.ssp_space.update_lengthscale(kwargs.get('length_scale', 4))
            self.context_ssp_space.update_lengthscale(kwargs.get('context_length_scale', 4))
            
        ### end if
        print('Selected Lengthscale: ', ssp_space.length_scale)
        
        

        # Encode the initial sample points 
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_space.ssp_dim)

        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
        
        if (decoder_method=='network') | (decoder_method=='network-optim'):
            self.ssp_space.train_decoder_net();
            self.context_ssp_space.train_decoder_net();
            self.init_samples=None
        else:
            self.init_samples = self.ssp_space.get_sample_pts_and_ssps(2**17,'length-scale')
            self.context_init_samples = self.context_ssp_space.get_sample_pts_and_ssps(2**17,'length-scale')
        self.decoder_method = decoder_method
        self.gamma_c = gamma_c
    ### end __init__


    def _optimize_lengthscale(self, init_xs, init_ys):

        ## fit to the initial values
        fit_gp = GaussianProcessRegressor(
                    kernel=SincKernel(
                        length_scale_bounds=(
                            1/np.sqrt(init_xs.shape[0]+1), 
                            1e5)
                        ),
                    alpha=1e-6,
                    normalize_y=True,
                    n_restarts_optimizer=20,
                    random_state=0,
                )
        fit_gp.fit(init_xs, init_ys)
        lenscale = np.exp(fit_gp.kernel_.theta)
        return lenscale
    ### end _optimize_lengthscale

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return mu, var, phi

    def initial_guess(self):
        '''
        The initial guess for optimizing the acquisition function.
        '''
        # Return an initial guess from either the distribution or
        # From the approximate solution of dot(m,x) + x^T Sigma x
        x = self.blr.sample()
        return self.ssp_space.bind(self.ssp_space.invert(self.context), x.T).T


    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''

        optim_norm_margin = 4
        C = self.ssp_space.get_binding_matrix(self.context)
        m = C @ self.blr.m
        sigma= C.T @ self.blr.S @ C
        def min_func(phi, m=m,
                        sigma=sigma,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta,
                        norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if phi_norm > norm_margin:
                phi = norm_margin * phi / phi_norm
            ### end if
            val = phi.T @ m
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(phi, m=m,
                      sigma=sigma,
                      gamma=self.gamma_t,
                      beta_inv=1/self.blr.beta,
                      norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if phi_norm > norm_margin:
                phi = norm_margin * phi / phi_norm
            ### end if
            sig_phi = sigma @ phi
            sqr = (phi.T @ sig_phi ) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sig_phi / scale)
            return retval

        return min_func, gradient

    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float, step_num=0, info=None):
        '''
        Updates the state of the Bayesian Linear Regression.
        '''
    
        x_val = x_t
        y_val = y_t
        self.context = self.context_ssp_space.encode(info)
        self.context_raw = info
        
#         y_val = self.transformer.transform(y_t)
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])
        ### end if

        # Update BLR
        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)
        
        # Update gamma
        #self.gamma_t = self.gamma_t + sigma_t
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)

    def encode(self, x):
        if x.shape[1] == self.data_dim:
            return self.ssp_space.bind(self.ssp_space.encode(x), self.context)
        else:
            return self.ssp_space.bind(self.ssp_space.encode(x[:,:self.data_dim]), 
                                       self.context_ssp_space.encode(x[:,-self.context_dim:]))
    
    
    def decode(self,ssp):
        action = self.ssp_space.decode(ssp,method=self.decoder_method,samples=self.init_samples)
        return np.concatenate([action, self.context_raw.reshape(-1)])
