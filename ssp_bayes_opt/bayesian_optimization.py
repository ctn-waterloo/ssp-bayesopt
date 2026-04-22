import numpy as np
import time
import logging
import sys
from scipy.optimize import minimize, Bounds
from typing import Callable

from . import agents
from . import sspspace

try:
    from guppy import hpy as _hpy
except ImportError:
    _hpy = None


def _get_memory_usage(heap):
    return heap.heap().size / 1024 ** 2


class BayesianOptimization:
    def __init__(self, f: Callable[..., float] = None, bounds: np.ndarray = None,
                 log_and_plot_f: Callable[..., float] = None,
                 random_state: int = None, verbose: bool = False,
                 sampling_seed: int = None):
        """Initialize the Bayesian Optimization object.

        Parameters
        ----------
        f : Callable
            Target function to maximize. Expected signature: (x: np.ndarray, info: str) -> float,
            or (x: np.ndarray) -> float if it only takes one argument.
        bounds : np.ndarray, shape (n_dims, 2)
            Lower and upper bounds for each input dimension.
        random_state : int, optional
            Seed for numpy's global RNG.
        sampling_seed : int, optional
            Seed for the Sobol sampler used during agent initialization.
        log_and_plot_f : Callable, optional
            Hook called after each BO step with (xs, ys, times, trial, memory).
        verbose : bool
            If True, emit DEBUG-level log messages.
        """
        assert f is not None, 'Must specify a callable target function'
        assert bounds is not None, 'Must specify input bounds'
        assert bounds.shape[1] == 2, 'bounds must have shape (n_dims, 2)'

        if random_state is not None:
            np.random.seed(random_state)

        if hasattr(f, '__code__') and f.__code__.co_argcount == 1:
            self.target = lambda x, info=None: f(x)
        else:
            self.target = f
        self.sampling_seed = sampling_seed
        self.bounds = bounds
        self.log_and_plot_f = log_and_plot_f

        self.data_dim = self.bounds.shape[0]
        self.num_decoding = 10000

        self.xs = None
        self.ys = None

        self.logger = logging.getLogger(__name__)

    def initialize_agent(self, init_points: int = 10, agent_type='ssp-hex', **kwargs):
        """Sample init_points from the domain, evaluate the target, and construct an agent.

        Parameters
        ----------
        init_points : int or tuple
            Number of initial samples, or a (xs, ys) tuple of pre-evaluated points.
        agent_type : str
            One of: ssp-hex, ssp-rand, ssp-custom, gp, static-gp, gp-matern, gp-sinc,
            gp-ucb-matern, gp-ucb-sinc, disc-domain, rff, ssp-traj, ssp-multi,
            ssp-nas-graph, ssp-mcbo.

        Returns
        -------
        agt : Agent
        init_xs : np.ndarray
        init_ys : np.ndarray
        """
        num_perimeter_samples = kwargs.pop('num_perimeter_samples', 0)

        if 'traj' in agent_type:
            self.logger.info('Creating Trajectory Domain')
            domain = agents.domains.TrajectoryDomain(
                kwargs['traj_len'], kwargs['x_dim'], self.bounds)
        elif 'multi' in agent_type:
            self.logger.info('Creating Multi-Agent Trajectory Domain')
            domain = agents.domains.MultiTrajectoryDomain(
                kwargs['n_agents'], kwargs['traj_len'], kwargs['x_dim'],
                self.bounds, kwargs.pop('goals', None))
        elif ('nas' in agent_type) or ('mcbo' in agent_type):
            self.logger.info(f'Creating {agent_type} Domain')
            domain = agents.domains.TargetDefinedDomain(self.target)
        else:
            self.logger.info('Creating Rectangular Domain')
            domain = agents.domains.BoundedDomain(self.bounds)

        self.logger.info('Sampling from domain')
        if isinstance(init_points, int):
            init_xs = domain.sample(init_points)
            n_init_points = init_points
        else:
            init_xs = np.copy(init_points[0])
            n_init_points = init_xs.shape[0]

        if isinstance(init_points, tuple) and init_points[1] is not None:
            self.logger.info('Loading pre-evaluated samples')
            init_ys = init_points[1]
        else:
            self.logger.info('Evaluating target function on initial samples')
            init_ys = np.array(
                [self.target(np.atleast_2d(x), str(itr))
                 for itr, x in enumerate(init_xs)]
            ).reshape((n_init_points, -1))

        self.logger.info(f'Creating {agent_type} agent')
        if agent_type == 'ssp-hex':
            ssp_space = sspspace.HexagonalSSPSpace(self.data_dim, **kwargs)
            agt = agents.SSPAgent(init_xs, init_ys, ssp_space, **kwargs)
        elif agent_type == 'ssp-rand':
            ssp_space = sspspace.RandomSSPSpace(self.data_dim, **kwargs)
            agt = agents.SSPAgent(init_xs, init_ys, ssp_space, **kwargs)
        elif agent_type == 'ssp-custom':
            assert 'ssp_space' in kwargs
            agt = agents.SSPAgent(init_xs, init_ys, kwargs.get('ssp_space'))
        elif 'gp' in agent_type:
            if 'static' in agent_type:
                agt = agents.GPAgent(init_xs, init_ys, updating=False, **kwargs)
            elif 'ucb-matern' in agent_type:
                agt = agents.GPUCBAgent(init_xs, init_ys, kernel_type='matern',
                                        updating=False, **kwargs)
            elif 'ucb-sinc' in agent_type:
                agt = agents.GPUCBAgent(init_xs, init_ys, kernel_type='sinc',
                                        updating=False, **kwargs)
            elif 'matern' in agent_type:
                agt = agents.GPAgent(init_xs, init_ys, kernel_type='matern',
                                     updating=False, **kwargs)
            elif 'sinc' in agent_type:
                agt = agents.GPAgent(init_xs, init_ys, kernel_type='sinc',
                                     updating=False, **kwargs)
            else:
                agt = agents.GPAgent(init_xs, init_ys, **kwargs)
        elif 'disc-domain' in agent_type:
            agt = agents.DiscretizedDomainAgent(
                init_xs, init_ys, bounds=self.bounds, **kwargs)
        elif 'rff' in agent_type:
            agt = agents.RFFAgent(init_xs, init_ys, **kwargs)
        elif agent_type == 'ssp-traj':
            agt = agents.SSPTrajectoryAgent(init_xs, init_ys, **kwargs)
            init_xs = agt.init_xs
            init_ys = agt.init_ys
        elif agent_type == 'ssp-multi':
            agt = agents.SSPMultiAgent(init_xs, init_ys, **kwargs)
            init_xs = agt.init_xs
            init_ys = agt.init_ys
        elif agent_type == 'ssp-nas-graph':
            agt = agents.SSPNASGraphAgent(init_xs, init_ys, **kwargs)
            init_xs = agt.init_xs
            init_ys = agt.init_ys
        elif agent_type == 'ssp-mcbo':
            agt = agents.SSPMCBOAgent(
                init_xs, init_ys, self.target.search_space, **kwargs)
            init_xs = agt.init_xs
            init_ys = agt.init_ys
        else:
            raise NotImplementedError(f'{agent_type} agent not implemented')

        if num_perimeter_samples > 0:
            perimeter_xs = []
            print('Adding perimeter samples to prior')
            for i in range(num_perimeter_samples):
                idxs = np.random.choice(self.bounds.shape[1],
                                        size=(self.bounds.shape[0],))
                perimeter_xs.append([self.bounds[r, c] for r, c in enumerate(idxs)])
            perimeter_xs = np.squeeze(perimeter_xs)
            perimeter_ys = np.zeros((perimeter_xs.shape[0], 1))
            agt.update(perimeter_xs, perimeter_ys, 0, 0)

        self.logger.info(f'{type(agt).__name__} agent created')
        return agt, init_xs, init_ys

    def maximize(self, init_points: int = 10, n_iter: int = 100,
                 num_restarts: int = 5, agent_type='ssp-hex',
                 save_memory=True, use_jac=True, **kwargs):
        """Run the BO loop for n_iter steps.

        Parameters
        ----------
        init_points : int
            Initial samples used to fit agent hyperparameters.
        n_iter : int
            Number of BO acquisition steps.
        num_restarts : int
            Number of random restarts when optimizing the acquisition function.
        agent_type : str
            Agent type string (see initialize_agent for options).
        save_memory : bool
            Track heap memory usage (requires guppy; silently skipped if unavailable).
        use_jac : bool
            Use analytic gradients when available.
        """
        self.logger.info('Starting maximize')

        agt, init_xs, init_ys = self.initialize_agent(
            init_points, agent_type,
            domain_bounds=self.bounds,
            **kwargs
        )
        self.logger.info('Agent initialized')

        self.times = np.zeros((n_iter,))
        self.full_times = np.zeros((n_iter,))
        self.memory = np.zeros((n_iter, 1))
        self.xs = np.zeros((n_iter + init_xs.shape[0], init_xs.shape[1]))
        self.ys = np.zeros((n_iter + init_xs.shape[0],))

        for row_idx, (x, y) in enumerate(zip(init_xs, init_ys)):
            self.xs[row_idx] = x
            self.ys[row_idx] = y

        lbounds = self.bounds[:, 0]
        ubounds = self.bounds[:, 1]

        print('| iter\t | target\t | x\t |')
        print('-------------------------------')
        full_start = time.thread_time_ns()

        # REVIEW: best_val_idx = np.argmax(vals[:(t+1)]) below slices to only t+1
        # elements instead of all num_restarts. Should likely be np.argmax(vals).
        best_phi_score = np.ones((num_restarts,)) * -np.inf
        solns = np.zeros((num_restarts, init_xs.shape[1]))
        vals = np.zeros((num_restarts,))

        _use_memory = save_memory and _hpy is not None
        if _use_memory:
            heap = _hpy()
            heap.setref()

        for t in range(n_iter):
            if hasattr(time, 'thread_time_ns'):
                start = time.thread_time_ns()
            optim_func, jac_func = agt.acquisition_func()

            for restart_idx in range(num_restarts):
                start = time.thread_time_ns()
                if ('gp' in agent_type) or ('rff' in agent_type):
                    x_init = np.random.uniform(low=lbounds, high=ubounds,
                                               size=(len(ubounds),))
                    soln = minimize(optim_func, x_init,
                                    jac=jac_func if use_jac else None,
                                    method='L-BFGS-B',
                                    bounds=self.bounds)
                    self.times[t] = time.thread_time_ns() - start
                    solnx = np.copy(soln.x)
                    solnfun = soln.fun
                    if hasattr(time, 'thread_time_ns'):
                        self.times[t] = time.thread_time_ns() - start
                elif agent_type == 'disc-domain':
                    soln = agt.sample()
                    self.times[t] = time.thread_time_ns() - start
                    solnx = np.copy(soln.x)
                    solnfun = soln.fun
                    if hasattr(time, 'thread_time_ns'):
                        self.times[t] = time.thread_time_ns() - start
                elif 'ssp' in agent_type:
                    phi_init = np.ones(agt.ssp_dim) * (1 / agt.ssp_dim)
                    start = time.thread_time_ns()
                    soln = minimize(optim_func, phi_init.flatten(),
                                    jac=jac_func if use_jac else None,
                                    method='L-BFGS-B')
                    solnx = soln.x
                    solnfun = soln.fun
                    if hasattr(time, 'thread_time_ns'):
                        self.times[t] = time.thread_time_ns() - start
                    solnx = agt.decode(np.copy(np.atleast_2d(solnx)))
                    if hasattr(time, 'thread_time_ns'):
                        self.full_times[t] = time.thread_time_ns() - start
                else:
                    raise NotImplementedError(f'{agent_type} agent not implemented')

                vals[restart_idx] = -solnfun
                solns[restart_idx] = solnx

            optimization_status = f'{t + init_xs.shape[0]}'

            best_val_idx = np.argmax(vals[:(t + 1)])
            x_t = np.atleast_2d(solns[best_val_idx].flatten())
            y_t = np.atleast_2d(self.target(x_t, optimization_status))

            if y_t.flatten() >= best_phi_score.min():
                worst_idx = best_phi_score.argmin()
                best_phi_score[worst_idx] = y_t

            mu_t, var_t, phi_t = agt.eval(x_t)

            print(f'| step {t + init_xs.shape[0]}\t | {y_t}, {np.sqrt(var_t)}, {phi_t}\t ')
            update_start = time.thread_time_ns()
            agt.update(x_t, y_t, var_t, step_num=t + init_xs.shape[0])
            self.times[t] += time.thread_time_ns() - update_start
            self.full_times[t] += time.thread_time_ns() - update_start

            if _use_memory:
                self.memory[t, 0] = _get_memory_usage(heap)

            t_now = t + init_xs.shape[0]
            self.xs[t_now] = np.copy(x_t)
            self.ys[t_now] = np.copy(y_t)
            if self.log_and_plot_f is not None:
                self.log_and_plot_f(
                    np.vstack(self.xs[:t_now + 1]),
                    np.vstack(self.ys[:t_now + 1]),
                    times=self.times, trial=t_now, memory=self.memory
                )

            self.agt = agt

        self.total_time = time.thread_time_ns() - full_start

    @property
    def res(self):
        """List of {'target': float, 'params': np.ndarray} dicts for all evaluations."""
        return [{'target': t, 'params': p} for t, p in zip(self.ys, self.xs)]

    @property
    def max(self):
        """Best result found: {'target': float, 'params': np.ndarray}."""
        max_idx = np.argmax(self.ys)
        return {'target': self.ys[max_idx], 'params': self.xs[max_idx]}
