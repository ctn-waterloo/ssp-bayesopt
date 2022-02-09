import numpy as np
import time

from . import agent
from . import sspspace

from scipy.stats import qmc
from scipy.optimize import minimize
from typing import Callable

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, bounds: np.ndarray=None, 
                 random_state: int =None, verbose: bool=False):
        assert not f is None, 'Must specify a callable target function'
        assert not bounds is None, 'Must dictionary of input bounds'
        assert bounds.shape[1] == 2, 'Must specify bounds of form [lower, upper] for each data dimension'

        if not random_state is None:
            np.random.seed(random_state)

        self.target = f
        self.bounds = bounds

        self.data_dim = self.bounds.shape[0]
        self.num_decoding = 10000

        self.xs = None
        self.ys = None

    def initialize_agent(self,init_points: int =10, 
                         agent_type='ssp-hex',
                         length_scale: int = None,
                         **kwargs):
        init_xs = self._sample_domain(num_points=init_points)
        init_ys = np.array([self.target(np.atleast_2d(x)) for x in init_xs]).reshape((init_points,-1))

        print(length_scale)

        # Initialize the agent
        if agent_type=='ssp-hex':
            ssp_space = sspspace.HexagonalSSPSpace(self.data_dim, **kwargs)
            agt = agent.SSPAgent(init_xs, init_ys,length_scale=length_scale, ssp_space=ssp_space) 
        elif agent_type=='ssp-rand':
            ssp_space = sspspace.RandomSSPSpace(self.data_dim, **kwargs)
#             agt = agent.SSPAgent(init_xs, init_ys,ssp_space) 
            agt = agent.SSPAgent(init_xs, init_ys,length_scale,ssp_space=ssp_space) 
        elif agent_type=='gp':
            agt = agent.GPAgent(init_xs, init_ys,**kwargs) 
        else:
            raise NotImplementedError()
        return agt, init_xs, init_ys


    def maximize(self, init_points: int =10, n_iter: int =100,
                 num_restarts: int = 5,
                 lenscale: int = None,
                 agent_type='ssp-hex',**kwargs) -> np.ndarray:
        # sample_xs = self._sample_domain(num_points=128 * 128) #self.num_decoding)
        # sample_ssps = self.agt.encode(sample_xs)
        # assert sample_ssps.shape[0] == sample_xs.shape[0]

        # self.ssp_to_domain_mat = np.linalg.pinv(sample_ssps) @ sample_xs

        # print(np.mean(np.linalg.norm(sample_xs - (sample_ssps @ self.ssp_to_domain_mat),axis=1)))

        agt, init_xs, init_ys = self.initialize_agent(init_points,
                                                      agent_type,
                                                      length_scale=lenscale,
                                                      domain_bounds=self.bounds,
                                                      **kwargs
                                                      )

#         self.lengthscale = agt.ssp_space.length_scale
        self.lengthscale = agt.get_lengthscale()

        self.times = np.zeros((n_iter,))
        self.xs = []
        self.ys = []

        for x,y in zip(init_xs, init_ys):
            self.xs.append(x)
            self.ys.append(y)


        # Extract the upper and lower bounds of domain for sampling.
        lbounds = self.bounds[:,0]
        ubounds = self.bounds[:,1]

        print('| iter\t | target\t | x\t |')
        print('-------------------------------')
        for t in range(n_iter):
            ## Begin timing section
            start = time.thread_time_ns()
            # get the functions to optimize
            ### TODO fix jacobian so it returns dx in x space
            optim_func, jac_func = agt.acquisition_func()

            # Use optimization to find a sample location
            solns = []
            vals = []
            for _ in range(num_restarts):
               
                x_init = np.random.uniform(low=lbounds, high=ubounds, size=(len(ubounds),))

                if agent_type=='gp':
                    # Do bounded optimization to ensure x stays in bound
                    soln = minimize(optim_func, x_init,
                                    jac=jac_func, 
                                    method='L-BFGS-B', 
                                    bounds=self.bounds)
                    solnx = np.copy(soln.x)
                else: ## ssp agent
#                     phi_init = agt.encode(x_init)
                    phi_init = agt.initial_guess()
                    soln = minimize(optim_func, phi_init,
                                    jac=jac_func, 
                                    method='L-BFGS-B')
                    solnx = agt.decode(np.copy(np.atleast_2d(soln.x)))
                vals.append(-soln.fun)
                solns.append(solnx)
            self.times[t] = time.thread_time_ns() - start
            ## END timing section

            best_val_idx = np.argmax(vals)
            x_t = np.atleast_2d(solns[best_val_idx].flatten())
            y_t = np.atleast_2d(self.target(x_t))
            
            mu_t, var_t, phi_t = agt.eval(x_t)

            print(f'| {t}\t | {y_t}, {phi_t}\t | {x_t}\t |')
            agt.update(x_t, y_t, var_t)

            # Log actions
            self.xs.append(np.copy(x_t))
            self.ys.append(np.copy(y_t))

    def _sample_domain(self, num_points: int=10) -> np.ndarray:
        sampler = qmc.Sobol(d=self.data_dim) 
        u_sample_points = sampler.random(num_points)
        sample_points = qmc.scale(u_sample_points, self.bounds[:,0], self.bounds[:,1])
        return sample_points

    @property 
    def res(self):
        return [{'target':t, 'params':p} for t,p in zip(self.ys, self.xs)]

    @property
    def max(self):
        max_idx = np.argmax(self.ys)
        return {'target':self.ys[max_idx], 'params':self.xs[max_idx]}
