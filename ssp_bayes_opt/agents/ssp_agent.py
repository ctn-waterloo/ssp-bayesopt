import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor 

from .. import sspspace 
from .. import blr
from .kernels import SincKernel

# from pymanopt.function import numpy as function_decorator
# from pymanopt.manifolds import Sphere

from .agent import Agent


class SSPAgent(Agent):
    def __init__(self, init_xs, init_ys, ssp_space,
                 decoder_method='network-optim',
                 gamma_c=1.0,
                 beta_ucb=np.log(2/1e-6),
                 var_decay=0.,
                 **kwargs):
        super().__init__()
  
        (num_pts, data_dim) = init_xs.shape
        self.data_dim = data_dim
        self.init_xs = init_xs
        self.init_ys = init_ys
        self.decoder_method = decoder_method

        # Set-up the space and decoder: these are action space dependent
        self._set_ssp_space(ssp_space=ssp_space, **kwargs)
        self._set_decoder()

        # Encode the initial sample points
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))
        self.constraint_ssp = np.zeros_like(self.blr.m)

        # Acq. fun params
        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.var_weight = beta_ucb
        self.var_decay = var_decay
    ### end __init__


    def _set_ssp_space(self, ssp_space, seed=0, **kwargs):
        if ssp_space is None:
            ssp_space = sspspace.HexagonalSSPSpace(self.data_dim, ssp_dim=kwargs.get('ssp_dim', 100),
                 domain_bounds=kwargs.get('domain_bounds', None),
                length_scale=1, rng=seed)

        self.ssp_space = ssp_space
        self.ssp_dim = ssp_space.ssp_dim

        # Optimize the length scales
        if not 'length_scale' in kwargs or kwargs.get('length_scale') < 0:
            self.ssp_space.update_lengthscale(self._optimize_lengthscale(self.init_xs, self.init_ys))
        else:
            self.ssp_space.update_lengthscale(kwargs.get('length_scale', 4))
        ### end if
        print('Selected Lengthscale: ', ssp_space.length_scale)


    def _set_decoder(self):
        if (self.decoder_method == 'network') | (self.decoder_method == 'network-optim'):
            self.ssp_space.train_decoder_net();
            self.init_samples = None
        else:
            self.init_samples = self.get_init_samples(self.ssp_space)

    def length_scale(self):
        return self.ssp_space.length_scale

    def _optimize_lengthscale(self, init_xs, init_ys):

        ## fit to the initial values
        fit_gp = GaussianProcessRegressor(
                    kernel=SincKernel(
                        length_scale=1.,#np.ones(init_xs.shape[1]),
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

    def get_init_samples(self, ssp_space):
        n_ls_method = [2 * int(np.ceil((b[1] - b[0]) / ssp_space.length_scale[b_idx])) for b_idx, b in enumerate(ssp_space.domain_bounds)]
        if (np.prod(n_ls_method)*ssp_space.ssp_dim > 1e7) or ('optim' not in self.decoder_method):
            samples = ssp_space.get_sample_pts_and_ssps(
                    np.min([100,int(np.ceil((1e7/ssp_space.ssp_dim)**(1/ssp_space.domain_dim)))]),
                'grid')
        else:
            samples = ssp_space.get_sample_pts_and_ssps(1,'length-scale')
        return samples

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

    def untrusted(self, x, badness=-1):
        '''
        Updates the domain constraints for the optimization.
        TODO: modify to permit multiple updates at once

        Parameters
        ----------
        x : np.ndarray
            points to be excluded from the optimization.
            For now assuming one data point per call of untrusted


        badness : float
            The scale to be applied to the x points.  For now
            assuming that one scalar value is applied per point in
            x
        '''
        phi = self.encode(x)
        # TODO: modify to running average of ssps.
        # Could exceed the scale of the mean values
        # if not careful.
        self.constraint_ssp += badness * phi

    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''

        optim_norm_margin = 4

        # @function_decorator(Sphere(self.ssp_dim))
        def min_func(phi, m=self.blr.m,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta,
                        norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if phi_norm > norm_margin:
                phi = norm_margin * phi / phi_norm
            ### end if
            val = phi.T @ m
            mi = self.var_weight * np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()

        # @function_decorator(Sphere(self.ssp_dim))
        def gradient(phi, m=self.blr.m,
                      sigma=self.blr.S,
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
            retval = -(m.flatten() + self.var_weight * sig_phi / scale)
            return retval


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
        return self.ssp_space.encode(x)
    
    def decode(self,ssp):
        return self.ssp_space.decode(ssp,method=self.decoder_method,samples=self.init_samples)
