import numpy as np

# import ssp
# import blr

from scipy.stats import qmc
from typing import Callable

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, bounds: dict =None, random_state: int =None):
        assert not f is None, 'Must specify a callable target function'
        assert not bounds is None, 'Must dictionary of input bounds'

        if not random_state is None:
            np.random.seed(random_state)

        self.func = f
        self.bounds = bounds

        self.num_dims = len(self.bounds.keys())

        # TODO: Create the simplex.
        # TODO: Initialize BLR 

    def maximize(self, init_points: int =10, num_iters: int =100) -> np.ndarray:

        sampler = qmc.Sobol(d=self.num_dims) 

        lbounds, ubounds = zip(*[self.bounds[x] for x in self.bounds.keys()])
        u_sample_points = sample.random(init_points)
        sample_points = qmc.scale(u_sample_points, lbounds, ubounds)

        arg_names = self.bounds.keys()

        fs = [self.func(**dict(zip(arg_names, pt))) for pt in sample_points]




        for _ in range(num_iters):
            pass
        pass


