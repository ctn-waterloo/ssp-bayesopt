import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

from .. import sspspace
from .. import blr

from .agent import Agent

class SSPTrajectoryAgent(Agent):
    def __init__(self, init_xs, init_ys, x_dim=1, traj_len=1,
                 ssp_x_space=None, ssp_t_space=None, ssp_dim=151,
                 domain_bounds=None,length_scale=4):
        super().__init__()
        self.num_restarts = 10
        self.data_dim = x_dim*traj_len
        self.x_dim= x_dim
        self.traj_len = traj_len
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
        init_phis = self.encode(init_xs)


        self.init_xs = init_xs
        self.init_ys = init_ys

        self.blr = blr.BayesianLinearRegression(self.ssp_x_space.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))

        self.contraint_ssp = np.zeros_like(self.blr.m)

        # MI params
        self.gamma_t = 0
        self.sqrt_alpha = np.log(2/1e-6)
    
        self.init_samples = self.ssp_x_space.get_sample_pts_and_ssps(10000,'grid')
        
    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return mu, var, phi

#     def sample_trajectories(self, num_points=10):
#         sampler = qmc.Sobol(d=self.x_dim) 
#         u_sample_points = sampler.random(num_points*self.traj_len)
#         sample_points = qmc.scale(u_sample_points, self.ssp_x_space.domain_bounds[:,0], 
#                                   self.ssp_x_space.domain_bounds[:,1])
#         return sample_points.reshape(num_points, self.traj_len*self.x_dim)

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
        # if not careful.  Alternative approach: 
        # badness is -1 * self.blr.eval(phi)
        self.constraint_ssp += badness * phi

    def acquisition_func(self):
        '''
        return objective_func, jacobian_func
        '''
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        def min_func(phi, m=self.blr.m + self.constraint_ssp,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta):
            val = phi.T @ m
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(phi, m=self.blr.m + self.constraint_ssp,
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
        y_val = y_val
    
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
