import numpy as np
import time

import logging 
import sys
from scipy.stats import qmc


from . import agents
from . import sspspace

# from scipy.stats import qmc
from scipy.optimize import minimize, Bounds
from typing import Callable

from guppy import hpy

def get_memory_usage(h):
    return h.heap().size / 1024 ** 2
#     mem_info = psutil.Process().memory_info()
#     return (mem_info.rss / 1024.**2, mem_info.vms / 1024 ** 2)

class BayesianOptimization:
    def __init__(self, f: Callable[...,float] =None, bounds: np.ndarray=None,
                 log_and_plot_f: Callable[...,float] =None,
                 random_state: int =None, verbose: bool=False,
                 sampling_seed: int =None):
        '''
        Initializes the Bayesian Optimization object 

        Parameters:
        -----------

        f : Callable 
            The target function that is being optimized. Assumed to be a 
            of the form np.ndarray, str -> float. The second, string 
            argument will give status of the optimization.

        bounds : np.ndarray 
            A n X 2 numpy array. n is the dimensionality to the input of the 
            target function.

        random_state : int 
            A seed for the random number generator.

        sampling_seed : int
            A seed for the Sobol sampler for agent initialization.

        log_and_plot_f : Callable
            A function for logging and plotting.

        verbose : bool 
            Controls whether or not diagnostic information is printed.

        Returns:
        --------

            None
        '''

        assert not f is None, 'Must specify a callable target function'
        assert not bounds is None, 'Must dictionary of input bounds'
        assert bounds.shape[1] == 2, 'Must specify bounds of form [lower, upper] for each data dimension'

        if not random_state is None:
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

        self.logger = logging.getLogger()
        # self.logger.setLevel(logging.DEBUG)

        # handler = logging.StreamHandler(sys.stdout)
        # # handler.setLevel(logging.DEBUG)
        # self.logger.addHandler(handler)

    def initialize_agent(self,init_points: int =10,agent_type='ssp-hex',**kwargs):
        '''
        Creates the optimization agent from an initial sampling of the target
        function.

        Parameters:
        -----------
        
        init_points : int 
            The number of points sampled from the domain of the target 
            function for initial hyper parameter tuning.

        agent_type : str 
            One of (ssp-hex|ssp-rand|static-gp|gp).  'gp' re-optimizes the 
            lengthscale parameters after every new observation, the other 
            algorithms do not.  All agents use the Mutual Information 
            acquisition function of Contal et al. (2014).

        bounds : np.ndarray, optional 
            The bounds of the function domain, used with the ssp-* algorithms 
            when decoding sample points.

        Returns:
        --------

        agt : Agent 
            The agent used in optimizing the target function.

        init_xs : np.ndarray
            The sample points from the domain of the target function used to
            optimize the agent's hyper parameters

        init_ys : np.ndarray
            The values of the target function at init_xs
            
        '''

        #assert init_points > 1, f'Need to sample more than one point when initializing agents, got {init_points}'
   

        if 'traj' in agent_type:
            self.logger.info('Creating Trajectory Domain')
            domain = agents.domains.TrajectoryDomain(kwargs['traj_len'], 
                                                     kwargs['x_dim'],
                                                     self.bounds)
            
        elif 'multi' in agent_type:
            self.logger.info('Creating Multi-Agent Trajectory Domain')
            domain = agents.domains.MultiTrajectoryDomain(kwargs['n_agents'], 
                                                     kwargs['traj_len'], 
                                                     kwargs['x_dim'],
                                                     self.bounds,
                                                     kwargs.pop('goals',None))
        elif ('nas' in agent_type) or ('mcbo' in agent_type): # both just assume the target has a sample method
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

        self.logger.info('Evaluating Domain Samples')
        if isinstance(init_points, tuple) and init_points[1] is not None:
            self.logger.info('Loading Pre-evaluated Samples')
            init_ys = init_points[1]
        else:
            self.logger.info('Executing Target Function')
            init_ys = np.array(
                    [self.target(np.atleast_2d(x), str(itr))
                     for itr, x in enumerate(init_xs)]).reshape((n_init_points,-1))

        # Initialize the agent
        self.logger.info(f'Creating {agent_type} agent')
        if agent_type == 'ssp-hex':
            ssp_space = sspspace.HexagonalSSPSpace(self.data_dim, **kwargs)
            agt = agents.SSPAgent(init_xs, init_ys,ssp_space, **kwargs) 
        elif agent_type == 'ssp-rand':
            ssp_space = sspspace.RandomSSPSpace(self.data_dim, **kwargs)
            agt = agents.SSPAgent(init_xs, init_ys,ssp_space, **kwargs) 
        elif agent_type == 'ssp-custom':
            assert 'ssp_space' in kwargs
            agt = agents.SSPAgent(init_xs, init_ys,kwargs.get('ssp_space'))
        elif agent_type == 'gp':
            agt = agents.GPAgent(init_xs, init_ys,**kwargs) 
        elif agent_type == 'static-gp':
            agt = agents.GPAgent(init_xs, init_ys, updating=False, **kwargs) 
        elif agent_type == 'gp-matern':
            agt = agents.GPAgent(init_xs, init_ys, 
                                kernel_type='matern', 
                                updating=False, **kwargs) 
        elif agent_type == 'gp-sinc':
            agt = agents.GPAgent(init_xs, init_ys, 
                                kernel_type='sinc', 
                                updating=False, **kwargs) 
        elif agent_type == 'gp-ucb-matern':
            agt = agents.GPUCBAgent(init_xs, init_ys, 
                                kernel_type='matern', 
                                updating=False, **kwargs) 
        elif agent_type == 'gp-ucb-sinc':
            agt = agents.GPUCBAgent(init_xs, init_ys, 
                                kernel_type='sinc', 
                                updating=False, **kwargs)
        elif agent_type == 'rff':
            agt = agents.RFFAgent(init_xs, init_ys, **kwargs)
        elif agent_type == 'disc-domain':
            agt = agents.DiscretizedDomainAgent(init_xs, init_ys, bounds=self.bounds, **kwargs)
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
            agt = agents.SSPMCBOAgent(init_xs, init_ys, self.target.search_space, **kwargs)
            init_xs = agt.init_xs
            init_ys = agt.init_ys
        else:
            raise NotImplementedError(f'{agent_type} agent not implemented')
        self.logger.info(f'{type(agt).__name__} Agent created')
        return agt, init_xs, init_ys


    def maximize(self, init_points: int =10, n_iter: int =100,
                 num_restarts: int = 5, agent_type='ssp-hex',
                 save_memory=True,
                 **kwargs):

        '''
        Maximizes the target function.

        Parameters:
        -----------

        init_points : int
            The number of sample points used to initalize agent hyperparameters

        n_iter : int
            The total sampling budget for the optimization.

        num_restarts : int
            The number of restarts used when optimizing sample point selection.

        agent_type : str
            One of (ssp-hex|ssp-rand|static-gp|gp).  'gp' re-optimizes the 
            lengthscale parameters after every new observation, the other 
            algorithms do not.  All agents use the Mutual Information 
            acquisition function of Contal et al. (2014).

        Returns:
        --------

        None - Optimization results stored in property res
            
        '''

        # sample_xs = self._sample_domain(num_points=128 * 128) #self.num_decoding)
        # sample_ssps = self.agt.encode(sample_xs)
        # assert sample_ssps.shape[0] == sample_xs.shape[0]

        # self.ssp_to_domain_mat = np.linalg.pinv(sample_ssps) @ sample_xs

        # print(np.mean(np.linalg.norm(sample_xs - (sample_ssps @ self.ssp_to_domain_mat),axis=1)))

        self.logger.info('Maximizing')
       
        agt, init_xs, init_ys = self.initialize_agent(init_points,
                                                      agent_type,
                                                      domain_bounds=self.bounds,
                                                      **kwargs
                                                      )
        self.logger.info('Agent initialized')
        #self.length_scale = agt.length_scale()

        self.times = np.zeros((n_iter,))
        self.full_times = np.zeros((n_iter,))
        self.memory = np.zeros((n_iter,1)) 
        self.xs = np.zeros((n_iter + init_xs.shape[0], init_xs.shape[1]))
        self.ys = np.zeros((n_iter + init_xs.shape[0],))


        for row_idx, (x,y) in enumerate(zip(init_xs, init_ys)):
            self.xs[row_idx] = x
            self.ys[row_idx] = y



        # Extract the upper and lower bounds of domain for sampling.
        # Extract the upper and lower bounds of domain for sampling.
        lbounds = self.bounds[:,0]
        ubounds = self.bounds[:,1]

        print('| iter\t | target\t | x\t |')
        print('-------------------------------')
        full_start = time.thread_time_ns()
    
#         sorted_idxs = np.argsort(init_ys.flatten())[::-1]
#         best_phi = agt.encode(init_xs[sorted_idxs[:num_restarts],:])
#         best_phi_score = init_ys[sorted_idxs[:num_restarts]]
        best_phi_score = np.ones((num_restarts,)) * -np.inf
        solns = np.zeros((num_restarts,init_xs.shape[1]))
        vals = np.zeros((num_restarts,))

        gp_agent_types = ['gp-matern','gp-ucb-matern','gp-sinc','gp-ucb-sinc']
        if save_memory:
            heap = hpy()
            heap.setref()
        for t in range(n_iter):
#             heap.setref()
            ## Begin timing section
            if hasattr(time, 'thread_time_ns'):
                start = time.thread_time_ns()
            # get the functions to optimize
            optim_func, jac_func = agt.acquisition_func()
            # Use optimization to find a sample location
            for restart_idx in range(num_restarts):
                start = time.thread_time_ns()
                if (agent_type in gp_agent_types) or ('rff' in agent_type):
                    x_init = np.random.uniform(low=lbounds, high=ubounds, size=(len(ubounds),))
                    # Do bounded optimization to ensure x stays in bound
                    soln = minimize(optim_func, x_init,
                                    jac=jac_func, 
                                    method='L-BFGS-B',
                                    bounds=self.bounds)
                    self.times[t] = time.thread_time_ns() - start
                    solnx = np.copy(soln.x)
                elif agent_type=='disc-domain':
                    soln = agt.sample()
                    self.times[t] = time.thread_time_ns() - start
                    solnx = np.copy(soln.x)
                else: ## ssp agent
#                     phi_init = np.copy(best_phi[restart_idx,:])
                    phi_init = agt.initial_guess()
                    start = time.thread_time_ns()
                    soln = minimize(optim_func, phi_init.flatten(),
                                    jac=jac_func, 
                                    method='L-BFGS-B')
                    if hasattr(time, 'thread_time_ns'):
                        self.times[t] = time.thread_time_ns() - start
                    # TODO: move this outside the num_restarts loop
                    solnx = agt.decode(np.copy(np.atleast_2d(soln.x)))

                vals[restart_idx] = -soln.fun
                solns[restart_idx] = solnx
#             if hasattr(time, 'thread_time_ns'):
#                 self.times[t] = time.thread_time_ns() - start
            ## END timing section

            optimization_status = f'{t+init_xs.shape[0]}'

            best_val_idx = np.argmax(vals[:(t+1)])
            x_t = np.atleast_2d(solns[best_val_idx].flatten())
            y_t = np.atleast_2d(self.target(x_t, optimization_status))

            if y_t.flatten() >= best_phi_score.min():
                worst_idx = best_phi_score.argmin()
                best_phi_score[worst_idx] = y_t
            ### end if
            
            mu_t, var_t, phi_t = agt.eval(x_t)

            print(f'| step {t+init_xs.shape[0]}\t | {y_t}, {np.sqrt(var_t)}, {phi_t}\t ')#| {x_t}\t |')
            update_start = time.thread_time_ns()
            agt.update(x_t, y_t, var_t, step_num=t + init_xs.shape[0])
            self.times[t] += time.thread_time_ns() - update_start
            self.full_times[t] = time.thread_time_ns() - start

            if save_memory:
                self.memory[t,0] = get_memory_usage(heap)

            # Log actions
            t_now = t + init_xs.shape[0]
            self.xs[t_now] = np.copy(x_t)
            self.ys[t_now] = np.copy(y_t)
            if self.log_and_plot_f is not None:
                self.log_and_plot_f(np.vstack(self.xs[:t_now+1]), np.vstack(self.ys[:t_now+1]),times=self.times, trial=t_now, memory=self.memory)

            self.agt = agt
            
        self.total_time = time.thread_time_ns() - full_start

    def _sample_domain(self, num_points: int=10) -> np.ndarray:
        sampler = qmc.Sobol(d=self.data_dim, seed=self.sampling_seed) 
        u_sample_points = sampler.random(num_points)
        sample_points = qmc.scale(u_sample_points, 
                                  self.bounds[:,0],
                                  self.bounds[:,1])
        return sample_points

    @property 
    def res(self):
        '''
        The results of the optimization process
        '''
        return [{'target':t, 'params':p} for t,p in zip(self.ys, self.xs)]

    @property
    def max(self):
        '''
        The maximum identified by the optimization process
        '''
        max_idx = np.argmax(self.ys)
        return {'target':self.ys[max_idx], 'params':self.xs[max_idx]}
