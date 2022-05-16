import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

# import GP modules
from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.preprocessing import StandardScaler 

from . import sspspace
from . import blr

import functools
import warnings



def factory(agent_type, init_xs, init_ys, **kwargs):

    data_dim = init_xs.shape[1]
    # Initialize the agent
    agt = None
    if agent_type=='ssp-hex':
        ssp_space = sspspace.HexagonalSSPSpace(**kwargs)
        agt = SSPAgent(init_xs, init_ys,ssp_space) 
    elif agent_type=='ssp-rand':
        ssp_space = sspspace.RandomSSPSpace(**kwargs)
        agt = SSPAgent(init_xs, init_ys,ssp_space) 
    elif agent_type == 'gp':
        agt = GPAgent(init_xs, init_ys)
    elif agent_type == 'static-gp':
        agt = GPAgent(init_xs, init_ys, updating=True, **kwargs)
    else:
        raise RuntimeWarning(f'Undefined agent type {agent_type}')
    return agt


class Agent:
    def __init__(self):
        pass

    def eval(self, xs):
        pass

    def update(self, x_t, y_t, sigma_t):
        pass

    def acquisition_func(self):
        pass

    def length_scale(self):
        pass


class PassthroughScaler:
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x

class SSPAgent(Agent):
    def __init__(self, init_xs, init_ys, ssp_space=None, **kwargs):
        super().__init__()
  
        (num_pts, data_dim) = init_xs.shape
        self.data_dim = data_dim

#         self.scaler = StandardScaler()
        self.scaler = PassthroughScaler()

        if ssp_space is None:
            ssp_space = sspspace.HexagonalSSPSpace(data_dim,
                                    ssp_dim=151, 
                                    n_rotates=5, 
                                    n_scales=5, 
                                    scale_min=2*np.pi/np.sqrt(6) - 0.5,
                                    scale_max=2*np.pi/np.sqrt(6) + 0.5,
                                    domain_bounds=None, 
                                    length_scale=5,
            )
        
        self.ssp_space = ssp_space
        # Optimize the length scales
        if not 'length_scale' in kwargs or kwargs.get('length_scale') < 0:
            self.ssp_space.update_lengthscale(self._optimize_lengthscale(init_xs, init_ys))
        else:
            self.ssp_space.update_lengthscale(kwargs.get('length_scale', 4))
        ### end if
        print('Selected Lengthscale: ', self.ssp_space.length_scale)

        # Encode the initial sample points 
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_space.ssp_dim)

        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
        
#         self.init_samples = self.ssp_space.get_sample_pts_and_ssps(300**data_dim,'grid')
        self.init_samples = self.ssp_space.get_sample_pts_and_ssps(300**data_dim,'length-scale')

        # Cache for the input xs.
#         self.phis = None

    ### end __init__

    def _optimize_lengthscale(self, init_xs, init_ys):
        from .kernels import SincKernel

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


    def _optimize_lengthscale_v1(self, init_xs, init_ys):
        ls_0 = np.array([[7]]) 
        self.scaler.fit(init_ys)

        def min_func(length_scale, xs=init_xs, ys=self.scaler.transform(init_ys),
                        ssp_space=self.ssp_space):
            errors = []
            kfold = KFold(n_splits=min(xs.shape[0], 50))
            ssp_space.update_lengthscale(length_scale)

#             phis = ssp_space.encode(xs)
#             b = blr.BayesianLinearRegression(ssp_space.ssp_dim)
#             b.update(phis, ys)
#             mu, var = b.predict(phis)
#             diff = ys.flatten() - mu.flatten()
#             loss = -0.5*np.log(var) - 0.5*np.divide(np.power(diff,2),var)
#             err_val = np.sum(loss) - xs.shape[0] * np.log(2*np.pi) / 2
#             return err_val

            for train_idx, test_idx in kfold.split(xs):
                train_x, test_x = xs[train_idx], xs[test_idx]
                train_y, test_y = ys[train_idx], ys[test_idx]

                train_phis = ssp_space.encode(train_x)
                test_phis = ssp_space.encode(test_x)

                b = blr.BayesianLinearRegression(ssp_space.ssp_dim)
                b.update(train_phis, train_y)
                mu, var = b.predict(test_phis)
                diff = test_y.flatten() - mu.flatten()
                loss = -0.5*np.log(var) - 0.5*np.divide(np.power(diff,2),var)
                errors.append(np.sum(loss))
            ### end for
            err_val = np.sum(errors) - xs.shape[0] / (2*np.log(2*np.pi))
            return err_val
        ### end min_func

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B',
                          bounds=[(1/np.sqrt(init_xs.shape[0]),None)],
                          )
        return np.abs(retval.x) 


    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return self.scaler.inverse_transform(mu), var, phi

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
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        optim_norm_margin = 4

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
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


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
            retval = -(m.flatten() + sig_phi / scale)
            return retval

        return min_func, gradient
    

    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float):
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
        y_val = self.scaler.transform(y_val)
    
        # Update BLR
        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)
        
        # Update gamma
        self.gamma_t = self.gamma_t + sigma_t

    def encode(self,x):
        return self.ssp_space.encode(x)
    
    def decode(self,ssp):
#         return self.ssp_space.decode(ssp,method='from-set',samples=self.init_samples)
        return self.ssp_space.decode(ssp,method='direct-optim',samples=self.init_samples)

    def length_scale(self):
        return self.ssp_space.length_scale

class SSPTrajectoryAgent(Agent):
    def __init__(self, x_dim, traj_len, init_trajs, init_ys, ssp_x_space=None, ssp_t_space=None):
        super().__init__()
        self.num_restarts = 10
        (num_pts, data_dim) = init_trajs.shape
        self.data_dim = data_dim
        self.x_dim= x_dim
        self.traj_len = traj_len
        self.scaler = PassthroughScaler()
        if ssp_x_space is None:
            ssp_x_space = sspspace.HexagonalSSPSpace(x_dim,ssp_dim=151, n_rotates=5, n_scales=5, 
                 scale_min=0.1, scale_max=3,
                 domain_bounds=None, length_scale=5)
        if ssp_t_space is None:
            ssp_t_space = sspspace.RandomSSPSpace(1,ssp_dim=ssp_x_space.ssp_dim,
                 domain_bounds=np.array([[0,traj_len]]), length_scale=5)
        
        self.ssp_x_space = ssp_x_space
        self.ssp_t_space = ssp_t_space
        
        # Encode timestamps
        self.timestep_ssps = self.ssp_t_space.encode(np.linspace(0,traj_len,traj_len))
        
        # Encode the initial sample points 
        init_phis = self.encode(init_trajs)

        self.blr = blr.BayesianLinearRegression(self.ssp_space.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
    
        self.init_samples = self.ssp_space.get_sample_pts_and_ssps(400,'grid')
        
    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return self.scaler.inverse_transform(mu), var, phi

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
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        def min_func(phi, m=self.blr.m,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta):
            val = phi.T @ m
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(phi, m=self.blr.m,
                      sigma=self.blr.S,
                      gamma=self.gamma_t,
                      beta_inv=1/self.blr.beta):
            sqr = (phi.T @ sigma @ phi) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sigma @ phi / scale)
            return retval

        return min_func, gradient
    
    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float):
        '''
        Updates the state of the Bayesian Linear Regression.
        '''
    
        x_val = x_t
        y_val = y_t
        if len(x_t.shape) < 2:
            x_val = x_t.reshape(1, x_t.shape[0])
            y_val = y_t.reshape(1, y_t.shape[0])
        y_val = self.scaler.transform(y_val)
    
        # Update BLR
        phi = np.atleast_2d(self.encode(x_val).squeeze())
        self.blr.update(phi, y_val)
        
        # Update gamma
        self.gamma_t = self.gamma_t + sigma_t

    def encode(self,x):
        x = np.atleast_2d(x)
        S = np.zeros((self.ssp_x_space.ssp_dim,x.shape[0]))
        x = x.reshape(-1,self.traj_len,self.x_dim)
        for j in range(self.traj_len):
            S += self.ssp_x_space.bind(self.timestep_ssps[j,:] , self.ssp_x_space.encode(x[:,j,:]))
        return S.T
    
        
    def decode(self,ssp):
        decoded_traj = np.zeros((len(self.traj_len),self.x_dim))
        for j in range(self.traj_len):
            query = self.ssp_x_space.bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , ssp)
            decoded_traj[j,:] = self.ssp_x_space.decode(query, method='direct-optim',samples=self.init_samples)
        return decoded_traj.reshape(-1)

class GPAgent(Agent):
    def __init__(self, init_xs, init_ys, updating=True):
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
    ### end __init__

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

    def length_scale(self):
        return np.exp(self.gp.kernel_.theta)
