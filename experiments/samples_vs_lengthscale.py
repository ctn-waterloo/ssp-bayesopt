import numpy as np
import pytry
import matplotlib.pyplot as plt
import time

import functions

function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133,
    'styblinski-tang3': 39.16599*3 # 1D function
}

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

class SamplesVsLenScaleTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('algorithm one of (ssp-hex|ssp-rand|gp-mi)', algorithm='ssp-hex')
        self.param('num initial samples', num_init_samples=10)
    
    def evaluate(self, p):
        target, bounds, budget = functions.factory(p.function_name)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=bounds, verbose=p.verbose)
        agt, _, _ = optimizer.initialize_agent(p.num_init_samples,
                                               p.algorithm,
                                               domain_bounds=bounds,
                                               )
        return dict(
#             length_scale=agt.ssp_space.length_scale,
            length_scale=agt.get_lengthscale(),
        )

num_trials = 30
init_samples = [10, 20, 50, 80, 100, 200, 500, 1000, 5000]

funcs = ['goldstein-price', 'colville', 'himmelblau', 'branin-hoo']
# alg = 'ssp-hex'
alg = 'gp'
import os.path

for func in funcs:
    data_dir = os.path.join('/run/media/furlong/Data/ssp-bayesopt/samples-vs-ls/', func, alg)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for init_points in init_samples:
        for trial in range(num_trials):
            r = SamplesVsLenScaleTrial().run(**{'data_format':'npz',
                                      'data_dir':data_dir,
                                      'function_name':func, 
                                      'algorithm':alg,
                                      'num_init_samples':init_points})
