import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

from .. import sspspace
from .. import blr

from .agent import Agent

# Made for use with NAS benchmarks
# https://github.com/google-research/nasbench
class SSPNASGraphAgent(Agent):
    def __init__(self, init_xs, init_ys,
                 max_nodes=9, max_edges=7, num_ops = 3,
                 ssp_dim=151,
                 gamma_c=1.0,
                 beta_ucb=np.log(2/1e-6),
                 init_pos=None,
                 seed=None, **kwargs):
        super().__init__()
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.num_ops = num_ops + 1 #for input
        self.x_dim = int(0.5*(self.max_edges-1)*self.max_edges) # assuming max size
        self.threshold = (1/self.max_nodes + 1/self.max_edges)*0.3
        # self.ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        # self.fixed_ops = ['input', 'output']

        self.ssp_dim = ssp_dim
        self.num_restarts = 10
        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        # domain_bounds = np.array([np.zeros(self.max_nodes*self.max_edges),
        #                      self.num_ops*np.ones(self.max_nodes*self.max_edges)]).T
       
        self.spspace = sspspace.SPSpace(self.max_edges + self.num_ops + 2,
                                        dim=ssp_dim, seed=seed)
        self.node_sps = self.spspace.vectors[:self.max_edges]
        self.inverse_node_sps = self.spspace.inverse_vectors[:self.max_edges]
        self.ops_sps = self.spspace.vectors[self.max_edges:self.max_edges + self.num_ops ]
        self.inverse_ops_sps = self.spspace.inverse_vectors[:self.max_edges:self.max_edges + self.num_ops]
        self.op_slot_sp = self.spspace.vectors[self.num_ops].reshape(1,-1)
        self.inverse_op_slot_sp = self.spspace.inverse_vectors[self.num_ops].reshape(1,-1)
        self.target_slot_sp = self.spspace.vectors[self.num_ops+1].reshape(1,-1)
        self.inverse_target_slot_sp = self.spspace.inverse_vectors[self.num_ops+1].reshape(1,-1)
        self.ssp_dim = self.spspace.dim
        
        # Encode the initial sample points 
        init_phis = self.encode(init_xs)
        norms = np.linalg.norm(init_phis, axis=1)

        self.phi_norm_bounds = [norms.min(), norms.max()]
#         print('!!! norm_bounds', self.phi_norm_bounds)

        self.init_xs = init_xs
        self.init_ys = init_ys

        self.blr = blr.BayesianLinearRegression(self.ssp_dim)
        self.blr.update(init_phis, np.array(init_ys))
        self.constraint_ssp = np.zeros_like(self.blr.m)

        # MI params
        self.gamma_t = 0
        self.gamma_c = gamma_c
        self.sqrt_alpha = beta_ucb 

        # self.decoder_method = decoder_method
    
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

        def min_func(phi, m=self.blr.m,# + self.constraint_ssp,
                        sigma=self.blr.S,
                        gamma=self.gamma_t,
                        sqrt_alpha=self.sqrt_alpha,
                        beta_inv=1/self.blr.beta,
                        norm_margin=self.phi_norm_bounds):

            phi_norm = np.linalg.norm(phi)
            phi_norm_scale = np.mean(norm_margin) / phi_norm
            phi = phi_norm_scale * phi 

            val = phi.T @ m
            mi = sqrt_alpha * (np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma))
            return -(val + mi).flatten()


        def gradient(phi, m=self.blr.m,# + self.constraint_ssp,
                      sigma=self.blr.S,
                      gamma=self.gamma_t,
                      sqrt_alpha=self.sqrt_alpha,
                      beta_inv=1/self.blr.beta,
                      norm_margin=self.phi_norm_bounds):

            phi_norm = np.linalg.norm(phi)
            phi_norm_scale = np.mean(norm_margin) / phi_norm
            phi = phi_norm_scale * phi 

            sqr = (phi.T @ sigma @ phi) 
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sqrt_alpha * sigma @ phi / scale)
            return retval

        return min_func, gradient
    
    def update(self, x_t:np.ndarray, y_t:np.ndarray, sigma_t:float, step_num=0):
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

        phi_norm = np.linalg.norm(phi)
        if phi_norm < self.phi_norm_bounds[0]:
            self.phi_norm_bounds[0] = phi_norm
        if phi_norm > self.phi_norm_bounds[1]:
            self.phi_norm_bounds[1] = phi_norm
    
        self.blr.update(phi, y_val)
        
        # Update gamma
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)


    def encode(self,G):
        '''
        Translates a graph x into an SSP representation.
        

        Parameters:
        -----------
        x : np.ndarray
            A (s, l, d) numpy array specifying a graph G with particular structure
        '''
        G = np.atleast_2d(G)
        S = np.zeros((G.shape[0], self.ssp_dim))
        for n in range(G.shape[0]): # not vectorized
            for i in range(self.max_edges-1):
                layer_i = G[n,int((self.max_edges - 1 + 0.5 * (1 - i)) * i):int((self.max_edges - 1 - 0.5 * i) * (i + 1))]
                if np.sum(layer_i) > 0:
                    if i == 0:
                        op_i = 1
                    else:
                        op_i = int(np.round(np.mean(layer_i[layer_i>0]))) # what operation is happening on this layer?
                    _S = self.spspace.bind(self.op_slot_sp, self.ops_sps[op_i-1][None,:]) # bind & bundle that in
                    target_bundle = np.sum(self.node_sps[np.where(layer_i>0)[0],:], axis=0, keepdims=True) # target layers bundle
                    _S += self.spspace.bind(self.target_slot_sp, target_bundle)
                    S[n,:] += self.spspace.bind(self.node_sps[i][None,:], _S).flatten()
        return S
    
        
    def decode(self,ssp):
        ssp = np.atleast_2d(ssp)
        n_nodes = 0
        decoded_graphs = np.zeros((ssp.shape[0], self.x_dim))
        for n in range(ssp.shape[0]):
            decoded_graph = np.zeros((self.max_edges, self.max_edges))
            for i in range(self.max_edges - 1):
                query_i = self.spspace.bind(self.inverse_node_sps[i][None, :], ssp[n,:].reshape(1,-1))
                op_query = self.spspace.bind(self.inverse_op_slot_sp, query_i)
                if i == 0: # input layer is special
                    op_i = 1
                else:
                    op_i = 2 + np.argmax(np.sum(self.ops_sps[1:] * op_query, axis=-1))
                # if op_i>4:
                #     print(op_i)
                target_query = self.spspace.bind(self.inverse_target_slot_sp, query_i)
                sims = np.sum(self.node_sps[i:] * target_query, axis=-1)
                for j,sim in enumerate(sims):
                    if sim >= self.threshold:
                        n_nodes += 1
                        decoded_graph[i,j+i] = op_i
                    if n_nodes > self.max_nodes:
                        break
            decoded_graphs[n,:] = np.concatenate([decoded_graph[i,i+1:] for i in range(decoded_graph.shape[0]-1)])
        return decoded_graphs
