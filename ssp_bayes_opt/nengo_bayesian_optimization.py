import numpy as np
import time

import logging
import sys, os


from . import network_solver
from .bayesian_optimization import BayesianOptimization

import nengo
import pickle


def get_soln(data, time_steps=500):
    return np.mean(data[-time_steps:,:], axis=0)

def save_checkpoint(state, filepath):
    """Save the current state to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(state, f)

def load_checkpoint(filepath):
    """Load state from a checkpoint file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

class NengoBayesianOptimization(BayesianOptimization):
    def __init__(self, random_state=0, **kwargs):
        self.seed = random_state
        super(NengoBayesianOptimization, self).__init__(**kwargs)

    def maximize(self, init_points: int =10, n_iter: int =100,
                 num_restarts: int = 5, agent_type='ssp-hex',
                 neurons_per_dim=8, sim_time=2.5, tau=0.05,
                 neuron_type=nengo.LIF(), sim_type=nengo.Simulator,
                 sim_args={},
                 checkpoint_path=None,
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
       

        self.logger.info('Agent initialized')
        #self.length_scale = agt.length_scale()


        self.memory = np.zeros((n_iter,1))

        if (checkpoint_path is not None) and os.path.exists(checkpoint_path):
            self.logger.info(f"Loading checkpoint from {checkpoint_path}")
            loaded_state = load_checkpoint(checkpoint_path)
            t = loaded_state['iteration']
            init_points = loaded_state['init_points']
            init_xs = loaded_state['xs'][:init_points]
            init_ys = loaded_state['ys'][:init_points]
            self.xs = loaded_state['xs']
            self.ys = loaded_state['ys']
            self.times = loaded_state['times']
            agt = loaded_state['agt']

        else:
            self.logger.info("No checkpoint found. Starting from scratch.")
            t = 0
            agt, init_xs, init_ys = self.initialize_agent(init_points,
                                                          agent_type,
                                                          domain_bounds=self.bounds,
                                                          **kwargs
                                                          )

            init_points = init_xs.shape[0]
            self.xs = np.zeros((n_iter + init_points, init_xs.shape[1]))
            self.ys = np.zeros((n_iter + init_points,))
            self.times = np.zeros((n_iter,))
            self.full_times = np.zeros((n_iter,))
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
        # heap = hpy()
        # heap.setref()
        while t < n_iter:
            if hasattr(time, 'thread_time_ns'):
                start = time.thread_time_ns()
            # get the functions to optimize
            ### TODO fix jacobian so it returns dx in x space
            # optim_func, jac_func = agt.acquisition_func()

            # Use optimization to find a sample location
            # for restart_idx in range(num_restarts):

            if (agent_type in gp_agent_types) or (agent_type=='disc-domain'):
                print('Agent type not implemented in nengo', agent_type)
                exit(1)
            else: ## ssp agent
                phi_init = agt.initial_guess()
                start = time.thread_time_ns()
                solver_net, soln_probe, stim_node = network_solver.make_network(
                    bo_soln_init=phi_init.flatten(),
                    m= agt.blr.m.flatten(),
                    sigma=agt.blr.S,
                    beta_inv=1 / agt.blr.beta,
                    gamma_t=agt.gamma_c,
                    var_weight=agt.var_weight,
                    neurons_per_dim=neurons_per_dim,
                    tau=tau,
                    seed=self.seed,
                    neuron_type=neuron_type,
                    rate=kwargs.get('rate',1.)
                )
                if 'spinnaker' in sim_type.__module__:
                    import nengo_spinnaker
                    nengo_spinnaker.add_spinnaker_params(solver_net.config)
                    solver_net.config[stim_node].function_of_time = True
                sim = sim_type(solver_net, **sim_args)
                with sim:
                    sim.run(sim_time)

                if hasattr(time, 'thread_time_ns'):
                    self.times[t] = time.thread_time_ns() - start
                # TODO: move this outside the num_restarts loop
                # print(sim.model.utilization_summary()) #if loihi: generally 7 blocks
                solnx = get_soln(sim.data[soln_probe])
                optim_func, _ = agt.acquisition_func()
                solnf = optim_func(solnx.flatten())
                solnx = agt.decode(np.copy(np.atleast_2d(solnx)))
                self.full_times[t] = time.thread_time_ns() - start


                vals = -solnf
                solns = solnx
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

            # self.memory[t,0] = get_memory_usage(heap)

            # Log actions
            t_now = t + init_xs.shape[0]
            self.xs[t_now] = np.copy(x_t)
            self.ys[t_now] = np.copy(y_t)
            if self.log_and_plot_f is not None:
                self.log_and_plot_f(np.vstack(self.xs[:t_now+1]), np.vstack(self.ys[:t_now+1]),times=self.times, trial=t_now, memory=self.memory)

            self.agt = agt

            if (t % 10 == 0) and (checkpoint_path is not None):
                save_state = {'iteration': t, 'xs': self.xs, 'ys': self.ys,
                              'agt': agt, 'times': self.times}
                save_checkpoint(save_state, checkpoint_path)
                self.logger.info(f"Checkpoint saved at iteration {t}")

            t += 1
            
        self.total_time = time.thread_time_ns() - full_start

