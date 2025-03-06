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
                 max_conns=9, max_layers=7, num_ops=3,
                 ssp_dim=151,
                 length_scale=1.,
                 gamma_c=1.0,
                 beta_ucb=np.log(2/1e-6),
                 alpha_decay=1.0,
                 init_pos=None,
                 seed=None,
                 **kwargs):
        super().__init__()
        self.max_conns = max_conns
        self.max_layers = max_layers
        self.num_ops = num_ops + 2 #for input and output
        self.x_dim = int(0.5*(self.max_layers-1)*self.max_layers) # assuming max size
        self.threshold = 0.3#(1/self.max_nodes + 1/self.max_edges)*0.3
        # self.ops = ['conv1x1-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3']
        # self.fixed_ops = ['input', 'output']

        self.ssp_dim = ssp_dim
        self.num_restarts = 10
        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        # domain_bounds = np.array([np.zeros(self.max_nodes*self.max_layers),
        #                      self.num_ops*np.ones(self.max_nodes*self.max_layers)]).T

        # self.ssp_space = sspspace.HexagonalSSPSpace(1, ssp_dim, seed=seed, length_scale=length_scale)
        # self.ssp_dim = self.ssp_space.ssp_dim
        # self.sp_space = sspspace.SPSpace(self.num_ops + 2,
        #                                 dim=self.ssp_dim, seed=seed)
        #
        # self.node_sps = self.ssp_space.encode(np.arange(1,self.max_layers+1)[:,None])
        # self.inverse_node_sps = self.ssp_space.encode(-np.arange(1,self.max_layers+1)[:, None])
        #
        # self.ops_sps = self.sp_space.vectors[:self.num_ops ]
        # self.inverse_ops_sps = self.sp_space.inverse_vectors[:self.num_ops]
        # self.op_slot_sp = self.sp_space.vectors[self.num_ops].reshape(1,-1)
        # self.inverse_op_slot_sp = self.sp_space.inverse_vectors[self.num_ops].reshape(1,-1)
        # self.target_slot_sp = self.sp_space.vectors[self.num_ops+1].reshape(1,-1)
        # self.inverse_target_slot_sp = self.sp_space.inverse_vectors[self.num_ops+1].reshape(1,-1)


        self.sp_space = sspspace.SPSpace(self.max_layers + self.num_ops + 3,
                                        dim=ssp_dim, seed=seed)
        self.layer_sps = self.sp_space.vectors[:self.max_layers]
        self.inverse_layer_sps = self.sp_space.inverse_vectors[:self.max_layers]
        self.ops_sps = self.sp_space.vectors[self.max_layers:self.max_layers + self.num_ops]
        self.inverse_ops_sps = self.sp_space.inverse_vectors[self.max_layers:self.max_layers + self.num_ops]
        self.op_slot_sp = self.sp_space.vectors[self.max_layers + self.num_ops].reshape(1, -1)
        self.inverse_op_slot_sp = self.sp_space.inverse_vectors[self.max_layers + self.num_ops].reshape(1, -1)
        self.target_slot_sp = self.sp_space.vectors[self.max_layers + self.num_ops + 1].reshape(1, -1)
        self.inverse_target_slot_sp = self.sp_space.inverse_vectors[self.max_layers + self.num_ops + 1].reshape(1, -1)
        self.other_sp = self.sp_space.vectors[self.max_layers + self.num_ops + 2].reshape(1, -1)
        self.ssp_dim = self.sp_space.dim

        self.identity = self.sp_space.identity()[None, :]

        
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
        self.alpha_decay = alpha_decay

        # self.decoder_method = decoder_method
    
    def length_scale(self):
        return self.length_scales

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
        return self.blr.sample()


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

        self.sqrt_alpha = self.sqrt_alpha * self.alpha_decay
        
        # Update gamma
        if isinstance(self.gamma_c, (int, float)):
            self.gamma_t = self.gamma_t + self.gamma_c*sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)


    # def encode(self,G):
    #     '''
    #     Translates a graph x into an SSP representation.
    #
    #
    #     Parameters:
    #     -----------
    #     x : np.ndarray
    #         A (s, l, d) numpy array specifying a graph G with particular structure
    #     '''
    #     G = np.atleast_2d(G)
    #     S = np.zeros((G.shape[0], self.ssp_dim))
    #
    #     for n in range(G.shape[0]): # not vectorized
    #         S2 = self.identity
    #         for i in range(self.max_layers-1):
    #             layer_i = G[n,int((self.max_layers - 1 + 0.5 * (1 - i)) * i):int((self.max_layers - 1 - 0.5 * i) * (i + 1))]
    #             if np.sum(layer_i) > 0:
    #                 if i == 0:
    #                     op_i = 1
    #                 else:
    #                     op_i = int(np.round(np.mean(layer_i[layer_i>0]))) # what operation is happening on this layer?
    #                 _S = self.sp_space.bind(self.op_slot_sp, self.ops_sps[op_i-1][None,:]) # bind & bundle that in
    #                 target_bundle = np.sum(self.layer_sps[np.where(layer_i>0)[0],:], axis=0, keepdims=True) # target layers bundle
    #                 _S += self.sp_space.bind(self.target_slot_sp, target_bundle)
    #                 S[n,:] += self.sp_space.bind(self.layer_sps[i][None,:], _S).flatten()
    #                 S2 = self.sp_space.bind(S2, S[n,:])
    #         S[n, :] += 0.1*S2.flatten()
    #         S[n, :] /= np.linalg.norm(S[n, :])
    #     return S

    def encode(self, G):
        '''
        Translates a graph x into an SSP representation.


        Parameters:
        -----------
        x : np.ndarray
            A (s, l, d) numpy array specifying a graph G with particular structure
        '''
        G = np.atleast_2d(G.copy())
        S = np.zeros((G.shape[0], self.ssp_dim))

        for n in range(G.shape[0]):  # not vectorized
            S2 = self.identity.copy()#np.zeros((1,self.ssp_dim))#self.other_sp#self.identity
            for i in range(self.max_layers - 1):
                # S2 = self.identity
                layer_i = G[n,
                          int((self.max_layers - 1 + 0.5 * (1 - i)) * i):int((self.max_layers - 1 - 0.5 * i) * (i + 1))]
                if i == 0:
                    op_i = 0
                else:
                    op_i = int(G[n, self.x_dim + i])  # what operation is happening on this layer?
                _S = self.sp_space.bind(self.op_slot_sp, self.ops_sps[op_i][None, :])  # bind & bundle that in
                target_bundle = self.identity.copy()
                if np.sum(layer_i) > 0:
                    target_bundle = np.sum(self.layer_sps[1+i+np.where(layer_i > 0)[0], :], axis=0,
                                           keepdims=True).reshape(1,-1)  # target layers bundle
                    _S += self.sp_space.bind(self.target_slot_sp, target_bundle)
                _S = self.sp_space.bind(self.layer_sps[i][None, :], _S).flatten()


                S[n, :] += _S
                S2 = self.sp_space.bind(S2, self.layer_sps[i][None, :],
                                        target_bundle, self.ops_sps[op_i][None, :])
                # S2 = self.sp_space.bind(S2, _S)

            #     if np.sum(layer_i) > 0:
            #         S2 += self.sp_space.bind(self.sp_space.bind(self.layer_sps[i][None, :],
            #                                                                   self.ops_sps[op_i][None, :]),
            #                                                target_bundle)
            # # S[n, :] += 0.1 * S2.flatten()
            # S[n, :] += 0.1 * self.sp_space.bind(self.other_sp, S2).flatten()

                # S2 = self.sp_space.bind(S2, S[n, :])
                    # S2 = self.sp_space.bind(S2, S[n, :])
            S[n, :] += 0.5 * S2.flatten()
            # S[n, :] /= np.linalg.norm(S[n, :])
        return S
    
        
    def decode(self,ssp):
        ssp = np.atleast_2d(ssp.copy())
        n_conns = 0
        decoded_graphs = np.zeros((ssp.shape[0], self.x_dim + self.max_layers))
        for n in range(ssp.shape[0]):
            decoded_graph = np.zeros((self.max_layers, self.max_layers))
            decoded_ops = np.zeros(self.max_layers)
            for i in range(self.max_layers - 1):
                query_i = self.sp_space.bind(self.inverse_layer_sps[i][None, :], ssp[n,:].reshape(1,-1))
                op_query = self.sp_space.bind(self.inverse_op_slot_sp, query_i)
                op_i = 1 + np.argmax(np.sum(self.ops_sps[1:-1] * op_query, axis=-1))
                decoded_ops[i] = op_i
                # if op_i>4:
                #     print(op_i)
                target_query = self.sp_space.bind(self.inverse_target_slot_sp, query_i)
                sims = np.sum(self.layer_sps[i+1:] * target_query, axis=-1)
                target_bundle = [np.zeros(self.ssp_dim)]
                for j,sim in enumerate(sims):
                    if sim >= self.threshold:
                        n_conns += 1
                        decoded_graph[i,1+i+j] = 1
                        target_bundle.append(self.layer_sps[i+1+j])
                    if n_conns > self.max_conns:
                        break
                bound_term = self.sp_space.bind(self.op_slot_sp, self.ops_sps[op_i]) + self.sp_space.bind(self.target_slot_sp, np.sum(np.array(target_bundle), axis=0, keepdims=True))
                ssp[n,:] = ssp[n,:] - self.sp_space.bind(self.layer_sps[i][None,:], bound_term).flatten()
            decoded_ops[0] = 0
            decoded_ops[-1] = self.num_ops-1
            decoded_graphs[n,:] = np.concatenate([decoded_graph[i,i+1:] for i in range(decoded_graph.shape[0]-1)]
                                                 + [decoded_ops])
        return decoded_graphs
