import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold

from .. import sspspace
from .. import blr

from .ssp_agent import SSPAgent


class SSPTrajectoryAgent(SSPAgent):
    def __init__(self, init_xs, init_ys, **kwargs):
        super().__init__(init_xs, init_ys, None,
                         **kwargs)
        # Encode timestamps


        if not self.init_pos is None:
            constraint_val = self.ssp_x_space.bind(
                self.timestep_ssps[0, :],
                self.ssp_x_space.encode(self.init_pos)
            )
            # Transpose on constraint_val because the ssp_space expects
            # data to be organized with samples in rows 
            # but BLR expects samples in columns.
            self.constraint_ssp += constraint_val.T
        ### end if

    def _set_ssp_space(self, x_dim, traj_len,
                       domain_bounds,
                       decoder_method,
                       ssp_x_space=None, ssp_t_space=None,
                       init_pos=None, **kwargs):
        self.data_dim = x_dim * traj_len
        self.x_dim = x_dim
        self.init_pos = None if init_pos is None else np.atleast_2d(init_pos)
        self.traj_len = traj_len
        if domain_bounds is not None:
            domain_bounds = np.array([np.min(domain_bounds[:, 0]) * np.ones(x_dim),
                                      np.max(domain_bounds[:, 1]) * np.ones(x_dim)]).T
        if ssp_x_space is None:
            ssp_x_space = sspspace.HexagonalSSPSpace(x_dim, ssp_dim=kwargs.get('ssp_dim', 100),
                                                     scale_min=0.1, scale_max=3,
                                                     domain_bounds=domain_bounds, length_scale=1)
        if ssp_t_space is None:
            ssp_t_space = sspspace.RandomSSPSpace(1, ssp_dim=ssp_x_space.ssp_dim,
                                                  domain_bounds=np.array([[0, self.traj_len]]), length_scale=1)
        self.ssp_dim = ssp_x_space.ssp_dim
        self.ssp_x_space = ssp_x_space
        self.ssp_t_space = ssp_t_space

        if not 'length_scale' in kwargs or np.any(np.array(kwargs.get('length_scale')) < 0):
            optres = self._optimize_lengthscale(self.init_xs, self.init_ys)
            self.ssp_x_space.update_lengthscale(optres[0])
            self.ssp_t_space.update_lengthscale(optres[1])
        else:
            self.ssp_x_space.update_lengthscale(kwargs.get('length_scale', 4))
            self.ssp_t_space.update_lengthscale(kwargs.get('length_scale', 10))
        self.timestep_ssps = self.ssp_t_space.encode(
                np.linspace(0,
                            self.traj_len,
                            self.traj_len
                            ).reshape(-1, 1)
            )

        if (decoder_method == 'network') | (decoder_method == 'network-optim'):
            self.ssp_x_space.train_decoder_net();
            self.init_samples = None
        else:
            self.init_samples = self.ssp_x_space.get_sample_pts_and_ssps(10000, 'length-scale')
        self.decoder_method = decoder_method

    def length_scale(self):
        return np.array([self.ssp_x_space.length_scale, self.ssp_t_space.length_scale])

    def _optimize_lengthscale(self, init_trajs, init_ys):
        ls_0 = np.array([[4.], [10]])

        def min_func(length_scale, xs=init_trajs, ys=init_ys,
                     ssp_x_space=self.ssp_x_space, ssp_t_space=self.ssp_t_space):
            errors = []
            kfold = KFold(n_splits=min(xs.shape[0], 50))
            ssp_x_space.update_lengthscale(length_scale[0])
            ssp_t_space.update_lengthscale(length_scale[1])
            self.timestep_ssps = self.ssp_t_space.encode(
                np.linspace(0,
                            self.traj_len,
                            self.traj_len
                            ).reshape(-1, 1)
            )
            for train_idx, test_idx in kfold.split(xs):
                train_x, test_x = xs[train_idx], xs[test_idx]
                train_y, test_y = ys[train_idx], ys[test_idx]

                train_phis = self.encode(train_x)
                test_phis = self.encode(test_x)

                b = blr.BayesianLinearRegression(ssp_x_space.ssp_dim)
                b.update(train_phis, train_y)
                mu, var = b.predict(test_phis)
                diff = test_y.flatten() - mu.flatten()
                loss = -0.5 * np.log(var) - np.divide(np.power(diff, 2), var)
                errors.append(np.sum(-loss))
            ### end for
            return np.sum(errors)

        ### end min_func

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B',
                          bounds=[(1 / np.sqrt(init_trajs.shape[0]), None), (1 / np.sqrt(init_trajs.shape[0]), None)],
                          )
        return np.abs(retval.x)


    def encode(self, x):
        '''
        Translates a trajectory x into an SSP representation.
        HACK: This code depends on whether or not the init_pos
        has been specified in the constructor.  If it is, then
        x needs to be a trajectory of length l-1

        Parameters:
        -----------
        x : np.ndarray
            A (s, l, d) numpy array specifying s trajectories
            of length l.
        '''
        enc_x = np.atleast_2d(x)
        S = np.zeros((x.shape[0], self.ssp_x_space.ssp_dim))

        enc_x = enc_x.reshape(-1, self.traj_len, self.x_dim)
        for j in range(self.traj_len):
            S += self.ssp_x_space.bind(self.timestep_ssps[j, :],
                                       self.ssp_x_space.encode(enc_x[:, j, :]))
        return S

    def decode(self, ssp):
        quries = self.ssp_x_space.bind(self.ssp_t_space.invert(self.timestep_ssps), ssp)
        decoded_traj = self.ssp_x_space.decode(quries,
                                               method=self.decoder_method,
                                               samples=self.init_samples)
        return decoded_traj.reshape(-1)
