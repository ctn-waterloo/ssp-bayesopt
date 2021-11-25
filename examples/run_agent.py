import numpy as np
import pytry
import matplotlib.pyplot as plt
import time
import baseline_agents
import functions
from scipy.stats import qmc

from argparse import ArgumentParser
import os
import os.path
import random
import pickle


function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133,
    'styblinski-tang1': 39.16599,
    'styblinski-tang2': 39.16599*2,
    'styblinski-tang3': 39.16599*3 #1 dim ****
}

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

class SSPSamplingTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('agent_type', agent_type='ssp-mi')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('ssp dim', ssp_dim=385)
    
    def evaluate(self, p):
        target, pbounds, budget = functions.factory(p.function_name)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, pbounds=pbounds, 
                                                       verbose=p.verbose, 
                                                       agent_type=p.agent_type.split('_')[0],
                                                       ssp_dim=p.ssp_dim)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, n_iter=budget)
        elapsed_time = time.thread_time_ns() - start

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        regrets = function_maximum_value[p.function_name] - vals
        print(optimizer.max)
        
        return dict(
            regret=regrets,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            times = optimizer.times,
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )

# Code taken directly from gitlab
class GPSamplingTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
    
    def evaluate(self, p):
        target, pbounds, budget = functions.factory(p.function_name)
        num_dims = len(pbounds)
        
        def sample_domain(num_points: int=10) -> np.ndarray:
            sampler = qmc.Sobol(d=num_dims) 

            lbounds, ubounds = zip(*[pbounds[x] for x in pbounds.keys()])
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
            return sample_points
        
        agent_caller = baseline_agents.factory('gp-mi')
        init_xs = sample_domain(num_points=p.num_samples)
        arg_names = pbounds.keys()

        init_ys = np.array([target(**dict(zip(arg_names, x))) for x in init_xs]).reshape((p.num_samples,-1))

        init_idxs = np.random.randint(low=0, high=p.num_samples, size=(p.num_init_samples,))
        agt = agent_caller(init_xs[init_idxs,:], init_ys[init_idxs]) 
        sample_pts = init_xs
        vals = init_ys
        start = time.thread_time_ns()
        
        
        ##
        regret = np.zeros((budget,))
        times = np.zeros((budget,))

        sample_locs = np.zeros((budget, num_dims))
        val_mus = np.zeros((budget, p.num_samples ), dtype=np.float32)
        val_vars = np.zeros((budget, p.num_samples ), dtype=np.float32)
        val_acqs = np.zeros((budget, p.num_samples ), dtype=np.float32)
        for t in range(budget):

            start = time.thread_time_ns()

            mus, var, phis = agt.eval(sample_pts)
            sample_vals = mus+phis
            t_selected = np.argmax(sample_vals)

            times[t] = time.thread_time_ns() - start


            val_mus[t,:] = np.copy(mus).astype(np.float32).flatten()
            val_vars[t,:] = np.copy(var).astype(np.float32).flatten()
            val_acqs[t,:] = np.copy(sample_vals).astype(np.float32).flatten()

            x_t = sample_pts[t_selected,:]
#             sample_locs.append(x_t)
            sample_locs[t,:] = np.copy(x_t)
            sigmas = phis[t_selected]

            y_t = target(**dict(zip(arg_names, x_t)))
            agt.update(x_t, y_t, var[t_selected])
            regret[t] = function_maximum_value[p.function_name] - y_t 
            
        ##
        
        
        elapsed_time = time.thread_time_ns() - start

        
        return dict(
            regret=regret,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            times = times,
            budget=budget,
            vals=vals.astype(np.float32),
            mus=None,
            variances=None,
            acquisition=None,
        )

if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--func', dest='function_name', type=str, default='himmelblau')
    parser.add_argument('--agent', dest='agent_type', type=str, default='hex-mi')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=str, default=385)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=100)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=30)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='/home/ns2dumon/Documents/ssp-bayesopt/tests/data/')



    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir,args.function_name,args.agent_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'function_name':args.function_name,
                  'num_samples':args.num_samples,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'verbose':False,
                  'ssp_dim':args.ssp_dim
                  }
        if args.agent_type=='gp-mi':
            r = GPSamplingTrial().run(**params)
        else:
            r = SSPSamplingTrial().run(**params)
