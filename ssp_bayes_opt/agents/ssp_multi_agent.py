import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from .ssp_agent import SSPAgent

from .. import sspspace
from .. import blr


class SSPMultiAgent(SSPAgent):
    def __init__(self, init_xs, init_ys, **kwargs):
        super().__init__(init_xs, init_ys, None, **kwargs)


    def _set_ssp_space(self, n_agents,
                       x_dim, traj_len,
                       domain_bounds,
                       decoder_method,
                       ssp_x_spaces=None, ssp_t_space=None,
                       seed=0,
                       init_pos=None, **kwargs):
        assert n_agents > 0
        self.data_dim = x_dim * traj_len * n_agents
        self.n_agents = n_agents
        self.x_dim = x_dim
        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        self.traj_len = traj_len

        if ssp_x_spaces is None:
            ssp_x_spaces=[]
            for i in range(n_agents):
                ssp_x_spaces.append( sspspace.HexagonalSSPSpace(x_dim,ssp_dim=kwargs.get('ssp_dim', 100),
                                                                scale_min=0.1, scale_max=3,
                                                                domain_bounds=domain_bounds[i*x_dim:(i+1)*x_dim,:],
                                                                length_scale=1))

        if ssp_t_space is None:
            ssp_t_space = sspspace.RandomSSPSpace(1, ssp_dim=ssp_x_spaces[0].ssp_dim,
                                                  domain_bounds=np.array([[0, self.traj_len]]), length_scale=1)
        self.ssp_dim = ssp_x_spaces[0].ssp_dim
        self.ssp_x_spaces = ssp_x_spaces
        self.ssp_t_space = ssp_t_space

        if not 'length_scale' in kwargs or np.any(np.array(kwargs.get('length_scale')) < 0):
            optres = self._optimize_lengthscale(self.init_xs, self.init_ys)
            for i in range(n_agents):
                self.ssp_x_spaces[i].update_lengthscale(optres[i])
            self.ssp_t_space.update_lengthscale(optres[-1])
        else:
            for i in range(n_agents):
                self.ssp_x_spaces[i].update_lengthscale(kwargs.get('length_scale', 4))
            self.ssp_t_space.update_lengthscale(kwargs.get('time_length_scale', 10))
        self.timestep_ssps = self.ssp_t_space.encode(
            np.linspace(0,
                        self.traj_len,
                        self.traj_len
                        ).reshape(-1, 1)
        )
        self.agent_sps = sspspace.SPSpace(n_agents,ssp_x_spaces[0].ssp_dim, seed).vectors # changed on mar 14 2025

        if (decoder_method == 'network') | (decoder_method == 'network-optim'):
            for i in range(n_agents):
                self.ssp_x_spaces[i].train_decoder_net();
            self.init_samples = None
        else:
            init_samples = []
            for i in range(n_agents):
                init_samples.append(self.ssp_x_spaces[i].get_sample_pts_and_ssps(2 ** 17, method='length-scale'))
            self.init_samples = init_samples
        self.decoder_method = decoder_method

    def length_scale(self):
        return np.array([space.length_scale for space in self.ssp_x_spaces] + [self.ssp_t_space.length_scale])


    def _optimize_lengthscale(self, init_trajs, init_ys):
        ls_0 = 4 * np.ones((self.n_agents+1,1))

        def min_func(length_scale, xs=init_trajs, ys=init_ys,
                        ssp_x_spaces=self.ssp_x_spaces,ssp_t_space=self.ssp_t_space):
            errors = []
            kfold = KFold(n_splits=min(xs.shape[0], 50))
            for i in range(self.n_agents):
                ssp_x_spaces[i].update_lengthscale(length_scale[0])
            ssp_t_space.update_lengthscale(length_scale[1])
            # Encode timestamps
            timestep_ssps = ssp_t_space.encode(
                np.linspace(0,
                            self.traj_len,
                            self.traj_len
                            ).reshape(-1, 1)
            )
            for train_idx, test_idx in kfold.split(xs):
                train_x, test_x = xs[train_idx], xs[test_idx]
                train_y, test_y = ys[train_idx], ys[test_idx]

                train_phis = self.encode(train_x,timestep_ssps=timestep_ssps)
                test_phis = self.encode(test_x,timestep_ssps=timestep_ssps)

                b = blr.BayesianLinearRegression(ssp_x_spaces[0].ssp_dim)
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


    def encode(self,x,timestep_ssps=None):
        '''
        Translates a trajectory x into an SSP representation.
        

        Parameters:
        -----------
        x : np.ndarray
            A (s, l, d) numpy array specifying s trajectories
            of length l.
        '''
        if timestep_ssps is None:
            timestep_ssps=self.timestep_ssps
        enc_x = np.atleast_2d(x)
        S = np.zeros((x.shape[0], self.ssp_dim))
        
        enc_x = enc_x.reshape(-1,self.n_agents,self.traj_len,self.x_dim)
        for i in range(self.n_agents):
            Si = np.zeros((x.shape[0], self.ssp_dim))
            for j in range(self.traj_len):
                #print(enc_x.shape)
                #print(enc_x[:,i,j,:].shape)
                #print(self.ssp_x_spaces[i].encode(enc_x[:,i,j,:]).shape)
                Si += self.ssp_x_spaces[i].bind(timestep_ssps[j,:],
                                       self.ssp_x_spaces[i].encode(enc_x[:,i,j,:]))
            S += self.ssp_x_spaces[i].bind(self.agent_sps[i,:], Si)
        return S
    
        
    def decode(self,ssp,timestep_ssps=None):
        if timestep_ssps is None:
            timestep_ssps=self.timestep_ssps
        decoded_traj = np.zeros((self.n_agents, self.traj_len,self.x_dim))
        for i in range(self.n_agents):
            sspi = self.ssp_x_spaces[i].bind(self.ssp_x_spaces[i].invert(self.agent_sps[i,:]), ssp)
            queries = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(timestep_ssps) , sspi)
            decoded_traj[i,:,:] = self.ssp_x_spaces[i].decode(queries, 
                                                              method=self.decoder_method,
                                                              samples=self.init_samples[i])
            # for j in range(self.traj_len):
            #     query = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , sspi)
            #     decoded_traj[i,j,:] = self.ssp_x_spaces[i].decode(query, method=self.decoder_method,samples=self.init_samples[i])
        return decoded_traj.reshape(-1)
