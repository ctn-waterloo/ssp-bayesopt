import numpy as np
import time

from . import agent

from scipy.stats import qmc
from typing import Callable

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, pbounds: dict =None, random_state: int =None, 
                 verbose: bool=False,agent_type: str='hex',ssp_dim: int=385):
        assert not f is None, 'Must specify a callable target function'
        assert not pbounds is None, 'Must dictionary of input bounds'

        if not random_state is None:
            np.random.seed(random_state)

        self.target = f
        self.bounds = pbounds

        self.num_dims = len(self.bounds.keys())
        self.num_decoding = 10000

        self.xs = None
        self.ys = None
        
        self.agent_type=agent_type
        self.ssp_dim=ssp_dim



    def maximize(self, init_points: int =10, n_iter: int =100) -> np.ndarray:


        init_xs = self._sample_domain(num_points=init_points)
        arg_names = self.bounds.keys()

        init_ys = np.array([self.target(**dict(zip(arg_names, x))) for x in init_xs]).reshape((init_points,-1))

        # Initialize the agent
        agt = agent.SSPAgent(init_xs, init_ys, axis_dim=int(self.ssp_dim), axis_type=self.agent_type) 


        # Determine decoding matrix
        ## TODO: how do we make sure that this stays within the bounds?
#         sample_xs = self._sample_domain(num_points=self.num_decoding)
        # Select domain sample locations.
        sample_xs = self._sample_domain(num_points=128 * 128) #self.num_decoding)
        sample_ssps = agt.encode(sample_xs)
        assert sample_ssps.shape[0] == sample_xs.shape[0]

        self.ssp_to_domain_mat = np.linalg.pinv(sample_ssps) @ sample_xs

        print(np.mean(np.linalg.norm(sample_xs - (sample_ssps @ self.ssp_to_domain_mat),axis=1)))


        self.times = np.zeros((n_iter,))
        self.xs = []
        self.ys = []

        for x,y in zip(init_xs, init_ys):
            self.xs.append(x)
            self.ys.append(y)


        print('| iter\t | target\t | x\t |')
        print('-------------------------------')
        for t in range(n_iter):

            # Use optimization to find a sample location
            start = time.thread_time_ns()
            x_t, var, phi  = agt.select_optimal([self.bounds[k] for k in self.bounds.keys()])
            self.times[t] = time.thread_time_ns() - start

            # Log actions
#             assert x_t_ssp.shape[0] == 1 
#             print(sample_ssps.shape, x_t_ssp.shape)
#             similarities = np.maximum(np.einsum('ij,kj->ik', sample_ssps, x_t_ssp/np.linalg.norm(x_t_ssp)),0)
#             x_t = sample_xs[np.argmax(similarities),:]
#             x_t = np.average(sample_xs, weights=similarities.flatten(), axis=0)

#             weights = similarities / np.sum(similarities)
#             print(similarities)
#             weights = np.exp(similarities) / np.sum(np.exp(similarities))
#             print(weights)
#             x_t = np.sum(sample_xs * weights, axis=0)
#             x_t = x_t_ssp @ self.ssp_to_domain_mat / np.linalg.norm(x_t_ssp)

#             sample_locs[t,:] = np.copy(x_t)

            query_point = dict(zip(arg_names, x_t.flatten()))
            y_t = np.array([[self.target(**query_point)]])

            print(f'| {t}\t | {y_t}\t | {query_point}\t |')
            agt.update(x_t, y_t, var)

            # Log actions
            self.xs.append(x_t)
            self.ys.append(y_t)

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
        return [{'target':t, 'params':p} for t,p in zip(self.ys, self.xs)]

    @property
    def max(self):
        max_idx = np.argmax(self.ys)
        return {'target':self.ys[max_idx], 'params':self.xs[max_idx]}


