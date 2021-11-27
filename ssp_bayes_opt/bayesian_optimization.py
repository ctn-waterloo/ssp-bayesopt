import numpy as np
import time

from . import agent

from scipy.stats import qmc
from scipy.optimize import minimize
from typing import Callable

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, pbounds: dict =None, 
                 random_state: int =None, verbose: bool=False):
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


    def maximize(self, init_points: int =10, n_iter: int =100, 
                 num_restarts: int = 5,
                 agent_type='ssp-mi') -> np.ndarray:


        init_xs = self._sample_domain(num_points=init_points)
        arg_names = self.bounds.keys()

        init_ys = np.array([self.target(**dict(zip(arg_names, x))) for x in init_xs]).reshape((init_points,-1))

        # Initialize the agent
        if agent_type == 'ssp-mi':
            agt = agent.SSPAgent(init_xs, init_ys) 
        elif agent_type == 'gp-mi':
            agt = agent.GPAgent(init_xs, init_ys)
        else:
            raise RuntimeWarning(f'Undefined agent type {agent_type}')

        self.times = np.zeros((n_iter,))
        self.xs = []
        self.ys = []

        for x,y in zip(init_xs, init_ys):
            self.xs.append(x)
            self.ys.append(y)


        # Extract the upper and lower bounds of domain for sampling.
        lbounds, ubounds = list(zip(*[self.bounds[k] for k in self.bounds.keys()]))

        print('| iter\t | target\t | x\t |')
        print('-------------------------------')
        for t in range(n_iter):
            solns = []
            vals = []

            ## Begin timing section
            start = time.thread_time_ns()
            # get the functions to optimize
            ### TODO fix jacobian so it returns dx in x space
            optim_func, jac_func = agt.acquisition_func()

            # Use optimization to find a sample location
            for res_idx in range(num_restarts):
               
                x_init = np.random.uniform(low=lbounds, high=ubounds, size=(2,))
#                 if res_idx == 0 and len(self.xs) > 0:
                if len(self.xs) > 0:
                    alpha = 0.9**t
                    x_init =  alpha * x_init + (1-alpha) * self.xs[np.argmax(self.ys)].flatten()


                # Do bounded optimization to ensure x stays in bound
                soln = minimize(optim_func, x_init,
                                jac=jac_func, 
                                method='L-BFGS-B', 
                                bounds=[self.bounds[k] for k in self.bounds.keys()])
                vals.append(-soln.fun)
                solns.append(np.copy(soln.x))
            self.times[t] = time.thread_time_ns() - start
            ## END timing section

            best_val_idx = np.argmax(vals)
            x_t = solns[best_val_idx].reshape((1,-1))
            mu_t, var_t, phi_t = agt.eval(x_t)

            # Log actions
            query_point = dict(zip(arg_names, x_t.flatten()))
            y_t = np.array([[self.target(**query_point)]])

            print(f'| {t}\t | {y_t}\t | {query_point}\t |')
            agt.update(x_t, y_t, var_t)

            # Log actions
            self.xs.append(x_t)
            self.ys.append(y_t)

        ### end for t in range(num_iters)

        pass

    def _sample_domain(self, num_points: int=10) -> np.ndarray:
        lbounds, ubounds = zip(*[self.bounds[x] for x in self.bounds.keys()])

        sampler = qmc.Sobol(d=self.num_dims) 
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


