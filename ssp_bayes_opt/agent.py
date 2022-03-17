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

from scipy.stats import qmc


def factory(agent_type, init_xs, init_ys, **kwargs):

    data_dim = init_xs.shape[1]
    # Initialize the agent
    agt = None
    if agent_type=='ssp-hex':
        ssp_space = sspspace.HexagonalSSPSpace(data_dim, **kwargs)
        agt = SSPAgent(init_xs, init_ys,ssp_space) 
    elif agent_type=='ssp-rand':
        ssp_space = sspspace.RandomSSPSpace(data_dim, **kwargs)
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
        '''
        Returns:
        --------
        
            objective_func : Callable
                Returns the acquisition function that is being optimized
                by the bayesian optimization.

            jacobian_func : Callable | None
                Returns the gradient of the acquisition function, or None
                if we are too lazy to implementing/think it won't help.
        '''
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
    def __init__(self, init_xs, init_ys, ssp_space=None):
        super().__init__()
  
        self.num_restarts = 10
        (num_pts, data_dim) = init_xs.shape
        self.data_dim = data_dim

#         self.scaler = StandardScaler()
        self.scaler = PassthroughScaler()

        if ssp_space is None:
            ssp_space = sspspace.HexagonalSSPSpace(data_dim,ssp_dim=151, n_rotates=5, n_scales=5, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=None, length_scale=5)
        
        self.ssp_space = ssp_space
        # Optimize the length scales
        #self.ssp_space.update_lengthscale(self._optimize_lengthscale(init_xs, init_ys))
        self.ssp_space.update_lengthscale(4)
        print('Selected Lengthscale: ', self.ssp_space.length_scale)

        # Encode the initial sample points 
        init_phis = self.encode(init_xs)

        self.blr = blr.BayesianLinearRegression(self.ssp_space.ssp_dim)

        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
        
        self.init_samples = self.ssp_space.get_sample_pts_and_ssps(300**data_dim,'grid')

        # Cache for the input xs.
#         self.phis = None

    ### end __init__

    def _optimize_lengthscale(self, init_xs, init_ys):
        ls_0 = np.array([[8.]]) 
        self.scaler.fit(init_ys)

        def min_func(length_scale, xs=init_xs, ys=self.scaler.transform(init_ys),
                        ssp_space=self.ssp_space):
            errors = []
            kfold = KFold(n_splits=min(xs.shape[0], 50))
            ssp_space.update_lengthscale(length_scale)

            for train_idx, test_idx in kfold.split(xs):
                train_x, test_x = xs[train_idx], xs[test_idx]
                train_y, test_y = ys[train_idx], ys[test_idx]

                train_phis = ssp_space.encode(train_x)
                test_phis = ssp_space.encode(test_x)

                b = blr.BayesianLinearRegression(ssp_space.ssp_dim)
                b.update(train_phis, train_y)
                mu, var = b.predict(test_phis)
                diff = test_y.flatten() - mu.flatten()
                loss = -0.5*np.log(var) - np.divide(np.power(diff,2),var)
                errors.append(np.sum(-loss))
            ### end for
            return np.sum(errors)
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
                      beta_inv=1/self.blr.beta): # same thing with beta_inv as above
            sqr = (phi.T @ sigma @ phi) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sigma @ phi / scale)
            return retval

        return min_func, gradient
    
    def acquisition_func_v2(self):
        '''
        return objective_func, jacobian_func
        '''

        def optim_func(x, m=self.blr.m,
                       sigma=self.blr.S,
                       gamma=self.gamma_t,
                       beta_inv=1/self.blr.beta
                       ):
            ptr = self.encode(x)
            val = ptr @ m
            mi = np.sqrt(gamma + beta_inv + ptr @ sigma @ ptr.T) - np.sqrt(gamma)
            return -(val + mi).flatten()
        ### end optim_func

        def jac_func(x, m=self.blr.m,
                     sigma=self.blr.S,
                     gamma=self.gamma_t,
                     beta_inv=1/self.blr.beta
                     ):
            ptr, grad_ptr = self.ssp_space.encode_and_deriv(x)
            sqr = (ptr @ sigma @ ptr.T) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = grad_ptr.squeeze().T @ -(m + sigma @ ptr.T / scale) 
            return retval
        ### end gradient
        return optim_func, jac_func 

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
        return self.ssp_space.decode(ssp,method='from-set',samples=self.init_samples)

class SSPTrajectoryAgent(Agent):
    def __init__(self,  n_init, func, x_dim=1, traj_len=1,
                 ssp_x_space=None, ssp_t_space=None, ssp_dim=151,domain_bounds=None,length_scale=4):
        super().__init__()
        self.num_restarts = 10
        self.data_dim = x_dim*traj_len
        self.x_dim= x_dim
        self.traj_len = traj_len
        self.scaler = PassthroughScaler()
        if domain_bounds is not None:
            domain_bounds = np.array([np.min(domain_bounds[:,0])*np.ones(x_dim), 
                             np.max(domain_bounds[:,1])*np.ones(x_dim)]).T
        if ssp_x_space is None:
            ssp_x_space = sspspace.HexagonalSSPSpace(x_dim,ssp_dim=ssp_dim,
                 scale_min=0.1, scale_max=3,
                 domain_bounds=domain_bounds, length_scale=length_scale)
        if ssp_t_space is None:
            ssp_t_space = sspspace.RandomSSPSpace(1,ssp_dim=ssp_x_space.ssp_dim,
                 domain_bounds=np.array([[0,traj_len]]), length_scale=1)
        
        self.ssp_x_space = ssp_x_space
        self.ssp_t_space = ssp_t_space
        
        # Encode timestamps
        self.timestep_ssps = self.ssp_t_space.encode(np.linspace(0,traj_len,traj_len).reshape(-1,1))
        
        # Encode the initial sample points 
        init_trajs = self.sample_trajectories(n_init)
        init_phis = self.encode(init_trajs)
        init_ys =  np.array([func(np.atleast_2d(x)) for x in init_trajs]).reshape((n_init,-1))
        self.init_xs = init_trajs
        self.init_ys = init_ys

        self.blr = blr.BayesianLinearRegression(self.ssp_x_space.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
    
        self.init_samples = self.ssp_x_space.get_sample_pts_and_ssps(10000,'grid')
        
    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return self.scaler.inverse_transform(mu), var, phi

    def sample_trajectories(self, num_points=10):
        sampler = qmc.Sobol(d=self.x_dim) 
        u_sample_points = sampler.random(num_points*self.traj_len)
        sample_points = qmc.scale(u_sample_points, self.ssp_x_space.domain_bounds[:,0], 
                                  self.ssp_x_space.domain_bounds[:,1])
        return sample_points.reshape(num_points, self.traj_len*self.x_dim)

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
        S = np.zeros((x.shape[0], self.ssp_x_space.ssp_dim))
        x = x.reshape(-1,self.traj_len,self.x_dim)
        for j in range(self.traj_len):
            S += self.ssp_x_space.bind(self.timestep_ssps[j,:] , self.ssp_x_space.encode(x[:,j,:]))
        return S
    
        
    def decode(self,ssp):
        decoded_traj = np.zeros((self.traj_len,self.x_dim))
        for j in range(self.traj_len):
            query = self.ssp_x_space.bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , ssp)
            decoded_traj[j,:] = self.ssp_x_space.decode(query, method='from-set',samples=self.init_samples)
        return decoded_traj.reshape(-1)

class UCBSSPAgent(SSPAgent):
    def __init__(self, init_xs, init_ys, ssp_space=None):
        super.__init__(init_xs, init_ys, ssp_space=ssp_space)

    def initial_guess(self):
        '''
        The initial guess for optimizing the acquisition function.
        '''
        # Return an initial guess from either the distribution or
        # From the approximate solution of dot(m,x) + x^T Sigma x
#         return -self.blr.S_inv @ self.blr.m
        return self.blr.sample()

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * np.sqrt(var) 
        return self.scaler.inverse_transform(mu), var, phi

    def acquisition_func(self):
        '''
        Returns:
        --------
        
            objective_func : Callable
                Returns the acquisition function that is being optimized
                by the bayesian optimization.

            jacobian_func : Callable | None
                Returns the gradient of the acquisition function, or None
                if we are too lazy to implementing/think it won't help.
        '''
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        def min_func(phi, m=self.blr.m,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta):
            val = phi.T @ m
            std = np.sqrt(beta_inv + phi.T @ sigma @ phi)
            return -(val + std).flatten()


        def gradient(phi, m=self.blr.m,
                      sigma=self.blr.S,
                      gamma=self.gamma_t,
                      beta_inv=1/self.blr.beta):
            sqr = (phi.T @ sigma @ phi) 
            scale = np.sqrt(sqr + beta_inv)
            retval = -(m.flatten() + sigma @ phi / scale)
            return retval

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
