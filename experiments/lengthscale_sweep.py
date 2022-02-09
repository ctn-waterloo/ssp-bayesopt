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

class SSPBayesOptTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('algorithm one of (ssp-hex|ssp-rand|gp-mi)', algorithm='ssp-hex')
        self.param('num initial samples', num_init_samples=10)
        self.param('num restarts', num_restarts=10)
        self.param('length_scale', length_scale=None)
    
    def evaluate(self, p):
        target, bounds, budget = functions.factory(p.function_name)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=bounds, verbose=p.verbose)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, 
                           n_iter=budget, 
                           agent_type=p.algorithm,
                           num_restarts=p.num_restarts,
                           lenscale=p.length_scale)
        elapsed_time = time.thread_time_ns() - start

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        regrets = function_maximum_value[p.function_name] - vals
        print(optimizer.max)
        
        return dict(
            lenscale=optimizer.lengthscale,
            regret=regrets,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )

cum_regrets = []
# num_trials = 30
num_trials = 5
# ls = [0.5, 1, 2, 5, 10, 20]
ls = [8, 12, 15, 18, 30]

# func = 'goldstein-price'
func = 'colville'
# func = 'himmelblau'
# func = 'branin-hoo'
alg = 'ssp-hex'
import os.path
data_dir = os.path.join('/run/media/furlong/Data/ssp-bayesopt/ls-sweep/', func, alg)
for l in ls:
    for trial in range(num_trials):
#     r = SSPBayesOptTrial().run(**{'function_name':'branin-hoo', 'algorithm':'ssp-mi'})
        r = SSPBayesOptTrial().run(**{'data_format':'npz',
                                      'data_dir':data_dir,
                                      'function_name':func, 
                                      'algorithm':alg,
                                      'num_restarts':10,
                                      'length_scale':l})
#     cum_reg = np.divide(np.cumsum(r['regret']), np.arange(1, len(r['regret'])+1))
#     cum_reg = r['regret']
#     cum_regrets.append(cum_reg)
# 
# mu_reg = np.mean(cum_regrets, axis=0)
# std_reg = np.std(cum_regrets, axis=0) / np.sqrt(num_trials)
# plt.plot(mu_reg)
# plt.plot(mu_reg - 1.96 * std_reg, ls='--')
# plt.plot(mu_reg + 1.96 * std_reg, ls='--')
# plt.show()
