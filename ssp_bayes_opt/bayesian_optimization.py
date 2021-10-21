import numpy as np
import time

from . import agent

from scipy.stats import qmc
from typing import Callable

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, pbounds: dict =None, random_state: int =None, verbose: bool=False):
        assert not f is None, 'Must specify a callable target function'
        assert not pbounds is None, 'Must dictionary of input bounds'

        if not random_state is None:
            np.random.seed(random_state)

        self.target = f
        self.bounds = pbounds

        self.num_dims = len(self.bounds.keys())


    def maximize(self, init_points: int =10, n_iter: int =100) -> np.ndarray:


        init_xs = self._sample_domain(num_points=init_points)
        arg_names = self.bounds.keys()

        init_ys = [self.target(**dict(zip(arg_names, x))) for x in init_xs]

        # Initialize the agent
        agt = agent.SSPAgent(init_xs, init_ys) 

        # Select domain sample locations.
        sample_pts = self._sample_domain(num_points=100)

        for t in range(n_iter):
            start = time.thread_time_ns()

            # Use optimization to find a sample location
            x_t, mu, var, phi  = agt.select_optimal(samples=sample_pts)
            times[t] = time.thread_time_ns() - start

            # Log actions
            sample_locs[t,:] = np.copy(x_t)
            y_t = self.target(**dict(zip(arg_names, x_t)))
            agt.update(x_t, y_t, var)

        ### end for t in range(num_iters)

        pass

    def _sample_domain(self, num_points: int=10) -> np.ndarray:
        sampler = qmc.Sobol(d=self.num_dims) 

        lbounds, ubounds = zip(*[self.bounds[x] for x in self.bounds.keys()])
        u_sample_points = sampler.random(num_points)
        sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
        return sample_points

    @property 
    def res(self):
        raise NotImplementedError()

    @property
    def max(self):
        raise NotImplementedError()


