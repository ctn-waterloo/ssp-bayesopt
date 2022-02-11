import numpy as np
import pytry
import matplotlib.pyplot as plt
import time
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

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('agent_type', agent_type='ssp-hex')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('ssp dim', ssp_dim=151)
    
    def evaluate(self, p):
        target, pbounds, budget = functions.factory(p.function_name)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=pbounds, 
                                                       verbose=p.verbose)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, n_iter=budget,
                           agent_type=p.agent_type,ssp_dim=p.ssp_dim)
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



if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--func', dest='function_name', type=str, default='himmelblau')
    parser.add_argument('--agent', dest='agent_type', type=str, default='ssp-hex')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=str, default=356)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=100)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='/home/ns2dumon/Documents/ssp-bayesopt/experiments/data/')



    
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
        r = SamplingTrial().run(**params)