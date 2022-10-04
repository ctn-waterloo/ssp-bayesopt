import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

from .. import sspspace
from .. import blr

from .agent import Agent

class SSPMultiAgent(Agent):
    def __init__(self, init_xs, init_ys, n_agents, x_dim=1, traj_len=1,
                 ssp_x_spaces=None, ssp_t_space=None,
                 ssp_dim=151,
                 domain_bounds=None, length_scale=4,
                 gamma_c=1.0,
                 init_pos=None,
                 decoder_method='network-optim'):
        super().__init__()
        self.num_restarts = 10
        self.n_agents = n_agents
        self.data_dim = x_dim*traj_len*n_agents
        self.x_dim= x_dim
        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        self.traj_len = traj_len
        if domain_bounds is not None:
            domain_bounds = np.array([np.min(domain_bounds[:,0])*np.ones(x_dim*n_agents), 
                             np.max(domain_bounds[:,1])*np.ones(x_dim*n_agents)]).T
        if not isinstance(length_scale, (list, tuple, np.ndarray)):
            length_scale = [length_scale]*n_agents
       
       
        if ssp_x_spaces is None:
            ssp_x_spaces=[]
            for i in range(n_agents):
                ssp_x_spaces.append( sspspace.HexagonalSSPSpace(x_dim,ssp_dim=ssp_dim,
                                                                scale_min=0.1, scale_max=3,
                                                                domain_bounds=domain_bounds[i*x_dim:(i+1)*x_dim,:], 
                                                                length_scale=length_scale[i]) )
        if ssp_t_space is None:
            ssp_t_space =  sspspace.RandomSSPSpace(1,ssp_dim=ssp_x_spaces[0].ssp_dim,
                                domain_bounds=np.array([[0,self.traj_len]]), length_scale=1) 
        agent_sps = sspspace.RandomSSPSpace(n_agents, ssp_dim=ssp_x_spaces[0].ssp_dim, length_scale=1).axis_matrix.T
        self.ssp_x_spaces = ssp_x_spaces
        self.ssp_t_space = ssp_t_space
        self.agent_sps = agent_sps
        self.ssp_dim = ssp_x_spaces[0].ssp_dim
        # Encode timestamps
        self.timestep_ssps = self.ssp_t_space.encode(
                                                    np.linspace(0,
                                                                self.traj_len,
                                                                self.traj_len
                                                                ).reshape(-1,1)
                                                    )
        ###
        
        # Encode the initial sample points 
        init_phis = self.encode(init_xs)


        self.init_xs = init_xs
        self.init_ys = init_ys

        # optres = self._optimize_lengthscale(init_xs, init_ys)
        # for i in range(n_agents):
        #     self.ssp_x_spaces[i].update_lengthscale(optres[0])
        #self.ssp_t_space.update_lengthscale(optres[1])
        self.length_scales = length_scale

        self.blr = blr.BayesianLinearRegression(self.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))


        self.constraint_ssp = np.zeros_like(self.blr.m)

        #if not self.init_pos is None:
            # constraint_val = self.ssp_x_space.bind(
            #                             self.timestep_ssps[0,:],
            #                             self.ssp_x_space.encode(self.init_pos)
            #                         )
            # # Transpose on constraint_val because the ssp_space expects
            # # data to be organized with samples in rows 
            # # but BLR expects samples in columns.
            # self.constraint_ssp += constraint_val.T
        ### end if

        # MI params
        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.sqrt_alpha = np.log(2/1e-6)
        

        if (decoder_method=='network') | (decoder_method=='network-optim'):
            for i in range(n_agents):
                self.ssp_x_spaces[i].train_decoder_net();
            self.init_samples=[None]*n_agents
        else:
            init_samples = []
            for i in range(n_agents):
                init_samples.append( self.ssp_x_spaces[i].get_sample_pts_and_ssps(2**17, method='length-scale') )
            self.init_samples = init_samples
        self.decoder_method = decoder_method
    
    def length_scale(self):
        return self.length_scales

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

    def _optimize_lengthscale(self, init_trajs, init_ys):
        ls_0 = np.array([[4.],[4.]]) 

        def min_func(length_scale, xs=init_trajs, ys=init_ys,
                        ssp_x_space=self.ssp_x_space,ssp_t_space=self.ssp_t_space):
            errors = []
            kfold = KFold(n_splits=min(xs.shape[0], 50))
            ssp_x_space.update_lengthscale(length_scale[0])
            ssp_t_space.update_lengthscale(length_scale[1])
            for train_idx, test_idx in kfold.split(xs):
                train_x, test_x = xs[train_idx], xs[test_idx]
                train_y, test_y = ys[train_idx], ys[test_idx]

                train_phis = self.encode(train_x)
                test_phis = self.encode(test_x)

                b = blr.BayesianLinearRegression(ssp_x_space.ssp_dim)
                b.update(train_phis, train_y)
                mu, var = b.predict(test_phis)
                diff = test_y.flatten() - mu.flatten()
                loss = -0.5*np.log(var) - np.divide(np.power(diff,2),var)
                errors.append(np.sum(-loss))
            ### end for
            return np.sum(errors)
        ### end min_func

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B',
                          bounds=[(1/np.sqrt(init_trajs.shape[0]),None),(1/np.sqrt(init_trajs.shape[0]),None)],
                          )
        return np.abs(retval.x) 

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
        # TODO: Currently returning (objective_func, None) to be fixed when 
        # I finish the derivation

        optim_norm_margin = 5
        def min_func(phi, m=self.blr.m,# + self.constraint_ssp,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        beta_inv=1/self.blr.beta,
                        norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if np.abs(phi_norm - norm_margin) >= 1:
                phi = norm_margin * phi / phi_norm
            ### end if

            val = phi.T @ m
            mi = np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma)
            return -(val + mi).flatten()


        def gradient(phi, m=self.blr.m,# + self.constraint_ssp,
                      sigma=self.blr.S,
                      gamma=self.gamma_t,
                      beta_inv=1/self.blr.beta,
                      norm_margin=optim_norm_margin):

            phi_norm = np.linalg.norm(phi)
            if np.abs(phi_norm - norm_margin) >= 1:
                phi = norm_margin * phi / phi_norm
            ### end if
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
#         print('!!!encoding vector mag: ', np.linalg.norm(phi))
        self.blr.update(phi, y_val)
        
        # Update gamma
        self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        print(f'!!!!! gamma_t = {self.gamma_t}')

    def encode(self,x):
        '''
        Translates a trajectory x into an SSP representation.
        

        Parameters:
        -----------
        x : np.ndarray
            A (s, l, d) numpy array specifying s trajectories
            of length l.
        '''
        enc_x = np.atleast_2d(x)
        S = np.zeros((x.shape[0], self.ssp_dim))
        
        enc_x = enc_x.reshape(-1,self.n_agents,self.traj_len,self.x_dim)
        for i in range(self.n_agents):
            Si = np.zeros((x.shape[0], self.ssp_dim))
            for j in range(self.traj_len):
                #print(enc_x.shape)
                #print(enc_x[:,i,j,:].shape)
                #print(self.ssp_x_spaces[i].encode(enc_x[:,i,j,:]).shape)
                Si += self.ssp_x_spaces[i].bind(self.timestep_ssps[j,:], 
                                       self.ssp_x_spaces[i].encode(enc_x[:,i,j,:]))
            S += self.ssp_x_spaces[i].bind(self.agent_sps[i,:], Si)
        return S
    
        
    def decode(self,ssp):
        decoded_traj = np.zeros((self.n_agents, self.traj_len,self.x_dim))
        for i in range(self.n_agents):
            sspi = self.ssp_x_spaces[i].bind(self.ssp_x_spaces[i].invert(self.agent_sps[i,:]), ssp)
            queries = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(self.timestep_ssps) , sspi)
            decoded_traj[i,:,:] = self.ssp_x_spaces[i].decode(queries, 
                                                              method=self.decoder_method,
                                                              samples=self.init_samples[i])
            # for j in range(self.traj_len):
            #     query = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , sspi)
            #     decoded_traj[i,j,:] = self.ssp_x_spaces[i].decode(query, method=self.decoder_method,samples=self.init_samples[i])
        return decoded_traj.reshape(-1)
