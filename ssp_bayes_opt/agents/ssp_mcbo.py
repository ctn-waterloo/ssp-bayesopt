import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .. import sspspace
from .. import blr

from collections import defaultdict
from .ssp_agent import SSPAgent


# Made for use with MCBO benchmarks and framework
# https://github.com/huawei-noah/HEBO/tree/master/MCBO
class SSPMCBOAgent(SSPAgent):
    """SSP-based agent for mixed-variable Bayesian optimization.

    Handles continuous, discrete (integer), nominal (unordered categorical),
    and ordinal (ordered categorical) variables by encoding each type into a
    separate SSP space and binding the encodings together.

    Parameters (additional kwargs)
    --------------------------------
    search_space : mcbo.SearchSpace
        Defines variable names, types, and bounds.
    ssp_dim : int
    conjunctive_w : float
        Weight for the conjunctive (pairwise binding) term added to the encoding.
    same_nominal_space : bool
        If True, all nominal variables share a single filler space.
    seed : int
    """

    def __init__(self, init_xs, init_ys, search_space, **kwargs):
        super().__init__(init_xs, init_ys, None,
                         search_space=search_space, **kwargs)

    def _set_ssp_space(self, search_space, ssp_dim=151, length_scale=-1,
                       conjunctive_w=0.1, seed=0, same_nominal_space=False,
                       **kwargs):
        self.search_space = search_space
        self.param_names = np.array(list(search_space.params.keys()))
        self.n_params = len(search_space.params)
        self.n_ordinal = len(search_space.ordinal_names)
        domain_bounds = np.zeros((self.n_params, 2))
        domain_bounds[:, 0] = [search_space.params[p].transfo_lb
                                for p in search_space.params]
        domain_bounds[:, 1] = [search_space.params[p].transfo_ub
                                for p in search_space.params]
        self.domain_bounds = domain_bounds
        self.conjunctive_w = conjunctive_w
        make_quasi_ortho = False

        self.n_cont = len(search_space.cont_names)
        self.n_disc = len(search_space.disc_names)
        self.n_nomial = len(search_space.nominal_names)
        self.n_ordinal = len(search_space.ordinal_names)

        if 'length_scale' in kwargs:
            if not isinstance(length_scale, (list, tuple, np.ndarray)):
                length_scale = [length_scale] * self.n_params
            length_scale = np.array(length_scale)
        else:
            length_scale = np.ones(self.n_params)
        self.length_scales = length_scale

        self.cont_idxs = []
        self.disc_idxs = []
        self.nominal_idxs = []
        self.ordinal_idxs = []
        self.spaces = defaultdict(None)
        if search_space.cont_names:
            self.cont_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.cont_names]
            ).flatten()
            self.spaces['cont'] = sspspace.HexagonalSSPSpace(
                self.n_cont, ssp_dim=ssp_dim,
                domain_bounds=domain_bounds[self.cont_idxs, :],
                length_scale=length_scale[self.cont_idxs].reshape(-1, 1), rng=seed)
            ssp_dim = self.spaces['cont'].ssp_dim

        if search_space.disc_names:
            self.disc_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.disc_names]
            ).flatten()
            domain_bounds[self.disc_idxs, 0] = [
                search_space.params[p].lb for p in search_space.disc_names]
            domain_bounds[self.disc_idxs, 1] = [
                search_space.params[p].ub for p in search_space.disc_names]
            self.spaces['disc'] = sspspace.RandomSSPSpace(
                self.n_disc, ssp_dim=ssp_dim,
                domain_bounds=domain_bounds[self.disc_idxs, :],
                length_scale=length_scale[self.disc_idxs].reshape(-1, 1), rng=seed)

        if search_space.nominal_names:
            self.nominal_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.nominal_names]
            ).flatten()
            self.spaces['nominal_slots'] = sspspace.SPSpace(
                self.n_nomial, dim=ssp_dim, rng=seed, make_quasi_ortho=make_quasi_ortho)
            self.spaces['nominal_filler'] = defaultdict(None)
            self.nominal_names = search_space.nominal_names
            for i, p in enumerate(self.search_space.nominal_names):
                self.spaces['nominal_filler'][p] = sspspace.SPSpace(
                    search_space.params[p].num_uniqs, dim=ssp_dim,
                    rng=seed + i + 1, make_quasi_ortho=make_quasi_ortho)

        if search_space.ordinal_names:
            self.ordinal_idxs = np.array(
                [np.where(self.param_names == n)[0] for n in search_space.ordinal_names]
            ).flatten()
            self.spaces['ordinal_slots'] = sspspace.SPSpace(
                self.n_ordinal, dim=ssp_dim,
                rng=seed + 1 + self.n_nomial, make_quasi_ortho=make_quasi_ortho)
            self.spaces['ordinal_filler'] = defaultdict(None)
            self.ordinal_names = search_space.ordinal_names
            for i, p in enumerate(search_space.ordinal_names):
                self.spaces['ordinal_filler'][p] = sspspace.SPSpace(
                    search_space.params[p].num_uniqs, dim=ssp_dim,
                    rng=seed + i + 2 + self.n_nomial, make_quasi_ortho=make_quasi_ortho)

        if search_space.perm_names:
            raise NotImplementedError

        self.spaces['slots'] = sspspace.SPSpace(4, dim=ssp_dim, rng=seed)
        self.identity = self.spaces['slots'].identity()
        self.bind = self.spaces['slots'].bind
        self.ssp_dim = ssp_dim

        if 'length_scale' not in kwargs or kwargs.get('length_scale') < 0:
            ls = self._optimize_lengthscale(self.init_xs, self.init_ys)
            self.length_scales = ls
            if self.n_cont > 0 and self.n_disc > 0:
                self.spaces['cont'].update_lengthscale(ls[:self.n_cont])
                self.spaces['disc'].update_lengthscale(ls[-self.n_disc:])
            elif self.n_cont > 0:
                self.spaces['cont'].update_lengthscale(ls)
            elif self.n_disc > 0:
                self.spaces['disc'].update_lengthscale(ls)

    def _set_decoder(self):
        self.init_samples = defaultdict(None)
        if self.n_cont > 0:
            if self.decoder_method in ('network', 'network-optim'):
                self.spaces['cont'].train_decoder_net()
            else:
                self.init_samples['cont'] = self.get_init_samples(self.spaces['cont'])
        if self.n_disc > 0:
            _int_list = [
                np.arange(self.search_space.params[n].lb,
                          self.search_space.params[n].ub + 1)
                for n in self.search_space.disc_names
            ]
            int_pts = np.vstack(np.meshgrid(*_int_list)).T
            int_ssps = self.spaces['disc'].encode(int_pts)
            self.init_samples['disc'] = (int_ssps, int_pts)

    def length_scale(self):
        return self.length_scales

    def _optimize_lengthscale(self, init_xs, init_ys):
        """Cross-validation over continuous and discrete length scales."""
        ls_0 = np.ones(self.n_cont + self.n_disc)
        ls_0[:self.n_cont] = (
            (self.domain_bounds[self.cont_idxs, 1]
             - self.domain_bounds[self.cont_idxs, 0]) / 10.)
        ls_0[self.n_cont:] = (
            (self.domain_bounds[self.disc_idxs, 1]
             - self.domain_bounds[self.disc_idxs, 0]) / 10.)

        if self.n_cont > 0 and self.n_disc > 0:
            def min_func(length_scale, xs=init_xs, ys=init_ys,
                         ssp_cont_space=self.spaces['cont'],
                         ssp_disc_space=self.spaces['disc']):
                errors = []
                kfold = KFold(n_splits=min(xs.shape[0], 50))
                ssp_cont_space.update_lengthscale(
                    length_scale[:self.n_cont].reshape(-1, 1))
                ssp_disc_space.update_lengthscale(
                    length_scale[self.n_cont:].reshape(-1, 1))
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
                return np.sum(errors)
        elif self.n_cont > 0:
            def min_func(length_scale, xs=init_xs, ys=init_ys,
                         ssp_cont_space=self.spaces['cont']):
                errors = []
                kfold = KFold(n_splits=min(xs.shape[0], 50))
                ssp_cont_space.update_lengthscale(length_scale.reshape(-1, 1))
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
                return np.sum(errors)
        elif self.n_disc > 0:
            def min_func(length_scale, xs=init_xs, ys=init_ys,
                         ssp_disc_space=self.spaces['disc']):
                errors = []
                kfold = KFold(n_splits=min(xs.shape[0], 50))
                ssp_disc_space.update_lengthscale(length_scale.reshape(-1, 1))
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
                return np.sum(errors)
        else:
            return None

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B',
                          bounds=[(1e-4, None)] * ls_0.shape[0],
                          options={'maxiter': 50})
        return np.abs(retval.x).reshape(-1, 1)

    def encode(self, x):
        """Encode a mixed-variable input into an SSP.

        Each variable type is encoded separately and combined via binding:
            phi = slot_cont * phi_cont + slot_disc * phi_disc
                + slot_nom * (sum_i nom_slot_i * nom_filler_i(x_i))
                + slot_ord * (sum_i ord_slot_i * ord_filler_i(x_i))
                + conjunctive_w * (product bindings of all phi_i)
        """
        x = np.atleast_2d(x)
        phi = np.zeros((x.shape[0], self.ssp_dim))
        phis = []
        if self.n_cont > 0:
            phis.append(self.spaces['cont'].encode(x[:, self.cont_idxs]))
            phi += self.bind(self.spaces['slots'].vectors[0], phis[-1])
        if self.n_disc > 0:
            phis.append(self.spaces['disc'].encode(x[:, self.disc_idxs]))
            phi += self.bind(self.spaces['slots'].vectors[1], phis[-1])

        if self.n_nomial > 0:
            _phi = np.zeros((x.shape[0], self.ssp_dim))
            for i in range(self.n_nomial):
                phis.append(
                    self.spaces['nominal_filler'][self.nominal_names[i]].encode(
                        x[:, self.nominal_idxs[i]]))
                _phi += self.bind(self.spaces['nominal_slots'].vectors[i], phis[-1])
            phi += self.bind(self.spaces['slots'].encode(2), _phi)

        if self.n_ordinal > 0:
            _phi = np.zeros((x.shape[0], self.ssp_dim))
            for i in range(self.n_ordinal):
                phis.append(
                    self.spaces['ordinal_filler'][self.ordinal_names[i]].encode(
                        x[:, self.ordinal_idxs[i]]))
                _phi += self.bind(self.spaces['ordinal_slots'].vectors[i], phis[-1])
            phi += self.bind(self.spaces['slots'].vectors[3], _phi)

        # Conjunctive term: sequential binding of all encoded variables.
        # REVIEW: _phi is only defined if n_nomial>0 or n_ordinal>0; if neither
        # condition holds and phis is non-empty, the loop below would fail.
        # Initialise to identity to make it safe in all cases.
        _phi = self.identity
        for p in phis:
            _phi = self.bind(p, _phi)
        phi += self.conjunctive_w * _phi
        return phi

    def decode(self, ssp):
        """Decode an SSP back to a mixed-variable input."""
        ssp = np.atleast_2d(ssp)
        decoded_x = np.zeros((ssp.shape[0], self.n_params))
        if self.n_cont > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[0])
            decoded_x[:, self.cont_idxs] = self.spaces['cont'].decode(
                query, method=self.decoder_method,
                samples=self.init_samples['cont'])
        if self.n_disc > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[1])
            decoded_x[:, self.disc_idxs] = self.spaces['disc'].decode(
                query, method='from-set', samples=self.init_samples['disc'])
        if self.n_nomial > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[2])
            for i in range(self.n_nomial):
                _query = self.bind(query, self.spaces['nominal_slots'].inverse_vectors[i])
                decoded_x[:, self.nominal_idxs[i]] = (
                    self.spaces['nominal_filler'][self.nominal_names[i]].decode(_query))
        if self.n_ordinal > 0:
            query = self.bind(ssp, self.spaces['slots'].inverse_vectors[3])
            for i in range(self.n_ordinal):
                _query = self.bind(query, self.spaces['ordinal_slots'].inverse_vectors[i])
                decoded_x[:, self.ordinal_idxs[i]] = (
                    self.spaces['ordinal_filler'][self.ordinal_names[i]].decode(_query))
        return decoded_x
