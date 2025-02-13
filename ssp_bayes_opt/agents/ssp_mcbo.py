import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings

from .. import sspspace
from .. import blr

from .agent import Agent
from collections import defaultdict


# Made for use with MCBO benchmarks and framework
# https://github.com/huawei-noah/HEBO/tree/master/MCBO
class SSPMCBOAgent(Agent):
    def __init__(self, init_xs, init_ys,
                 search_space,
                 ssp_dim=151,
                 length_scale=1.0,
                 gamma_c=1.0,
                 beta_ucb=np.log(2 / 1e-6),
                 init_pos=None,
                 seed=None,
                 decoder_method='direct-optim',optim_ls=True,
                 conjunctive_w=0.1,
                 **kwargs):
        super().__init__()

        self.search_space = search_space
        self.param_names = np.array(list(search_space.params.keys()))
        self.n_params = len(search_space.params)
        self.n_ordinal = len(search_space.ordinal_names)
        domain_bounds = np.zeros((self.n_params, 2))
        domain_bounds[:, 0] = [search_space.params[p].transfo_lb for p in search_space.params]
        domain_bounds[:, 1] = [search_space.params[p].transfo_ub for p in search_space.params]
        self.domain_bounds = domain_bounds
        self.conjunctive_w = conjunctive_w

        if not isinstance(length_scale, (list, tuple, np.ndarray)):
            length_scale = [length_scale] * self.n_params
        length_scale = np.array(length_scale)
        self.length_scales = length_scale

        self.n_cont = len(search_space.cont_names)
        self.n_disc = len(search_space.disc_names)
        self.n_nomial = len(search_space.nominal_names)

        self.cont_idxs = []
        self.disc_idxs = []
        self.nominal_idxs = []
        self.ordinal_idxs = []
        self.spaces = defaultdict(None)
        if search_space.cont_names:  # continuous params

            self.cont_idxs = np.array([np.where(self.param_names == n)[0] for n in search_space.cont_names]).flatten()
            # ls = 1.0#(self.domain_bounds[self.cont_idxs, 1] - self.domain_bounds[self.cont_idxs, 0])# / 10.

            self.spaces['cont'] = sspspace.HexagonalSSPSpace(self.n_cont,
                                                             ssp_dim=ssp_dim,
                                                             scale_min=0.1, scale_max=3,
                                                             domain_bounds=domain_bounds[self.cont_idxs, :],
                                                             length_scale=length_scale[self.cont_idxs].reshape(-1,1))
            ssp_dim = self.spaces['cont'].ssp_dim

        if search_space.disc_names:  # discrete params (integers, not categories)
            self.disc_idxs = np.array([np.where(self.param_names == n)[0] for n in search_space.disc_names]).flatten()
            self.spaces['disc'] = sspspace.RandomSSPSpace(self.n_disc,
                                                          ssp_dim=ssp_dim,
                                                          domain_bounds=domain_bounds[self.disc_idxs, :],
                                                          length_scale=length_scale[self.disc_idxs].reshape(-1,1))

        if search_space.nominal_names:  #unordered categories
            self.nominal_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.nominal_names]).flatten()
            self.spaces['nominal_slots'] = sspspace.SPSpace(self.n_nomial, dim=ssp_dim)
            self.spaces['nominal_filler'] = defaultdict(None)
            self.nominal_names = search_space.nominal_names
            for i, p in enumerate(search_space.nominal_names):
                self.spaces['nominal_filler'][p] = sspspace.SPSpace(search_space.params[p].num_uniqs, dim=ssp_dim)

        if search_space.ordinal_names:  #ordered categories
            self.ordinal_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.ordinal_names]).flatten()
            self.spaces['ordinal_slots'] = sspspace.SPSpace(self.n_ordinal, dim=ssp_dim)
            self.spaces['ordinal_filler'] = defaultdict(None)
            self.ordinal_names = search_space.ordinal_names
            for i, p in enumerate(search_space.ordinal_names):
                self.spaces['ordinal_filler'][p] = sspspace.SPSpace(search_space.params[p].num_uniqs, dim=ssp_dim)

        if search_space.perm_names:  # permutations
            raise NotImplementedError

        self.spaces['slots'] = sspspace.SPSpace(4, dim=ssp_dim)
        self.identity = self.spaces['slots'].identity()[None,:]
        self.bind = self.spaces['slots'].bind
        self.ssp_dim = ssp_dim


        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        # domain_bounds = np.array([np.zeros(self.max_nodes*self.max_edges),
        #                      self.num_ops*np.ones(self.max_nodes*self.max_edges)]).T


        if optim_ls:
            ls = self._optimize_lengthscale(init_xs, init_ys)
            if (self.n_cont>0) and (self.n_disc>0): # TODO have less if-else
                self.spaces['cont'].update_lengthscale(ls[:self.n_cont])
                self.spaces['disc'].update_lengthscale(ls[-self.n_disc:])
            elif (self.n_cont>0):
                self.spaces['cont'].update_lengthscale(ls)
            elif (self.n_disc>0):
                self.spaces['disc'].update_lengthscale(ls)


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

        self.init_samples = defaultdict(None)
        if (decoder_method == 'network') | (decoder_method == 'network-optim'):
            if self.n_cont > 0:
                self.spaces['cont'].train_decoder_net();
                self.spaces['disc'].train_decoder_net();
        else:
            if self.n_cont > 0:
                self.init_samples['cont'] = self.spaces['cont'].get_sample_pts_and_ssps(2**10, method='length-scale')
            if self.n_disc > 0:
                self.init_samples['disc'] = self.spaces['disc'].get_sample_pts_and_ssps(int(np.max(domain_bounds[self.disc_idxs, 1])),
                                                                                        method='grid')
        self.decoder_method = decoder_method

    def length_scale(self):
        return self.length_scales

    def eval(self, xs):
        phis = self.encode(xs)
        mu, var = self.blr.predict(phis)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t))
        return mu, var, phi

    def _optimize_lengthscale(self, init_xs, init_ys):
        ls_0 = np.ones(self.n_cont + self.n_disc)
        ls_0[:self.n_cont] = (self.domain_bounds[self.cont_idxs, 1] - self.domain_bounds[self.cont_idxs, 0]) / 10.
        ls_0[self.n_cont:] = (self.domain_bounds[self.disc_idxs, 1] - self.domain_bounds[self.disc_idxs, 0]) / 10.

        if (self.n_cont > 0) and (self.n_disc > 0):

            def min_func(length_scale, xs=init_xs, ys=init_ys,
                         ssp_cont_space=self.spaces['cont'],
                         ssp_disc_space=self.spaces['disc']):
                errors = []
                kfold = KFold(n_splits=min(xs.shape[0], 50))
                ssp_cont_space.update_lengthscale(length_scale[:self.n_cont].reshape(-1,1))
                ssp_disc_space.update_lengthscale(length_scale[self.n_cont:].reshape(-1,1))
                for train_idx, test_idx in kfold.split(xs):
                    train_x, test_x = xs[train_idx], xs[test_idx]
                    train_y, test_y = ys[train_idx], ys[test_idx]

                    train_phis = self.encode(train_x)
                    test_phis = self.encode(test_x)

                    b = blr.BayesianLinearRegression(ssp_cont_space.ssp_dim)
                    b.update(train_phis, train_y)
                    mu, var = b.predict(test_phis)
                    diff = test_y.flatten() - mu.flatten()
                    loss = -0.5 * np.log(var) - np.divide(np.power(diff, 2), var)
                    errors.append(np.sum(-loss))
                ### end for
                return np.sum(errors)
        elif (self.n_cont > 0):
            def min_func(length_scale, xs=init_xs, ys=init_ys,
                         ssp_cont_space=self.spaces['cont']):
                errors = []
                kfold = KFold(n_splits=min(xs.shape[0], 50))
                ssp_cont_space.update_lengthscale(length_scale.reshape(-1,1))
                for train_idx, test_idx in kfold.split(xs):
                    train_x, test_x = xs[train_idx], xs[test_idx]
                    train_y, test_y = ys[train_idx], ys[test_idx]

                    train_phis = self.encode(train_x)
                    test_phis = self.encode(test_x)

                    b = blr.BayesianLinearRegression(ssp_cont_space.ssp_dim)
                    b.update(train_phis, train_y)
                    mu, var = b.predict(test_phis)
                    diff = test_y.flatten() - mu.flatten()
                    loss = -0.5 * np.log(var) - np.divide(np.power(diff, 2), var)
                    errors.append(np.sum(-loss))
                ### end for
                return np.sum(errors)
        elif (self.n_disc > 0):
                def min_func(length_scale, xs=init_xs, ys=init_ys,
                             ssp_disc_space=self.spaces['disc']):
                    errors = []
                    kfold = KFold(n_splits=min(xs.shape[0], 50))
                    ssp_disc_space.update_lengthscale(length_scale.reshape(-1,1))
                    for train_idx, test_idx in kfold.split(xs):
                        train_x, test_x = xs[train_idx], xs[test_idx]
                        train_y, test_y = ys[train_idx], ys[test_idx]

                        train_phis = self.encode(train_x)
                        test_phis = self.encode(test_x)

                        b = blr.BayesianLinearRegression(ssp_disc_space.ssp_dim)
                        b.update(train_phis, train_y)
                        mu, var = b.predict(test_phis)
                        diff = test_y.flatten() - mu.flatten()
                        loss = -0.5 * np.log(var) - np.divide(np.power(diff, 2), var)
                        errors.append(np.sum(-loss))
                    ### end for
                    return np.sum(errors)
        else:
            return None

        ### end min_func

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B',
                          bounds=[(1e-4, None)] * ls_0.shape[0],
                          options={'maxiter': 50})
        return np.abs(retval.x).reshape(-1,1)

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

        def min_func(phi, m=self.blr.m,  # + self.constraint_ssp,
                     sigma=self.blr.S,
                     gamma=self.gamma_t,
                     sqrt_alpha=self.sqrt_alpha,
                     beta_inv=1 / self.blr.beta,
                     norm_margin=self.phi_norm_bounds):
            phi_norm = np.linalg.norm(phi)
            phi_norm_scale = np.mean(norm_margin) / phi_norm
            phi = phi_norm_scale * phi

            val = phi.T @ m
            mi = sqrt_alpha * (np.sqrt(gamma + beta_inv + phi.T @ sigma @ phi) - np.sqrt(gamma))
            return -(val + mi).flatten()

        def gradient(phi, m=self.blr.m,  # + self.constraint_ssp,
                     sigma=self.blr.S,
                     gamma=self.gamma_t,
                     sqrt_alpha=self.sqrt_alpha,
                     beta_inv=1 / self.blr.beta,
                     norm_margin=self.phi_norm_bounds):
            phi_norm = np.linalg.norm(phi)
            phi_norm_scale = np.mean(norm_margin) / phi_norm
            phi = phi_norm_scale * phi

            sqr = (phi.T @ sigma @ phi)
            scale = np.sqrt(sqr + gamma + beta_inv)
            retval = -(m.flatten() + sqrt_alpha * sigma @ phi / scale)
            return retval

        return min_func, gradient

    def update(self, x_t: np.ndarray, y_t: np.ndarray, sigma_t: float, step_num=0):
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
            self.gamma_t = self.gamma_t + self.gamma_c * sigma_t
        elif callable(self.gamma_c):
            self.gamma_t = self.gamma_t + self.gamma_c(step_num) * sigma_t
        else:
            msg = f'unable to use {self.gamma_c}, expected number of callable'
            print(msg)
            raise RuntimeError(msg)

    def encode(self, x):
        x = np.atleast_2d(x)
        phi = np.zeros((x.shape[0], self.ssp_dim))
        phis = []
        if self.n_cont > 0:
            phis.append( self.spaces['cont'].encode(x[:, self.cont_idxs]))
            phi += self.bind(self.spaces['slots'].vectors[0], phis[-1])
        if self.n_disc > 0:
            phis.append( self.spaces['disc'].encode(x[:, self.disc_idxs]))
            phi += self.bind(self.spaces['slots'].vectors[1], phis[-1])

        if self.n_nomial>0:
            _phi = np.zeros((x.shape[0], self.ssp_dim))
            for i in range(self.n_nomial):
                phis.append(self.spaces['nominal_filler'][self.nominal_names[i]].encode(x[:, self.nominal_idxs[i]]))
                _phi += self.bind(self.spaces['nominal_slots'].vectors[i],
                                  phis[-1])
            phi += self.bind(self.spaces['slots'].encode(2),_phi)

        if self.n_ordinal>0:
            _phi = np.zeros((x.shape[0], self.ssp_dim))
            for i in range(self.n_ordinal):
                phis.append(self.spaces['ordinal_filler'][self.ordinal_names[i]].encode(x[:, self.ordinal_idxs[i]]))
                _phi += self.bind(self.spaces['ordinal_slots'].vectors[i],
                                  phis[-1])
            phi += self.bind(self.spaces['slots'].vectors[3],
                             _phi)
        _phi = self.identity
        for p in phis:
            _phi = self.bind(p, _phi)
        phi += self.conjunctive_w*_phi
        return phi

    def decode(self, ssp):
        ssp = np.atleast_2d(ssp)
        decoded_x = np.zeros((ssp.shape[0], self.n_params))
        if self.n_cont > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[0])
            decoded_x[:, self.cont_idxs] = self.spaces['cont'].decode(query,
                                                                      method=self.decoder_method,
                                                                      samples=self.init_samples['cont']
                                                                      )
        if self.n_disc > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[1])
            disc_x = self.spaces['disc'].decode(query,
                                                method=self.decoder_method,
                                                samples=self.init_samples['disc']
                                                )
        if self.n_nomial > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[2])
            for i in range(self.n_nomial):
                _query = self.bind(query, self.spaces['nominal_slots'].inverse_vectors[i])
                decoded_x[:, self.nominal_idxs[i]] = self.spaces['nominal_filler'][self.nominal_names[i]].decode(_query)
        if self.n_ordinal > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[3])
            for i in range(self.n_nomial):
                _query = self.bind(query, self.spaces['ordinal_slots'].inverse_vectors[i])
                decoded_x[:, self.ordinal_idxs[i]] = self.spaces['ordinal_filler'][self.ordinal_names[i]].decode(_query)
        return decoded_x
