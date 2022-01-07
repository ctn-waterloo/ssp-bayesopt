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
}

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

class SSPBayesOptTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('algorithm one of (ssp-mi|gp-mi)', algorithm='ssp-mi')
        self.param('num initial samples', num_init_samples=10)
        self.param('num restarts', num_restarts=10)
    
    def evaluate(self, p):
        target, pbounds, budget = functions.factory(p.function_name)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, pbounds=pbounds, verbose=p.verbose)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples,
                    n_iter=budget,
                    agent_type=p.algorithm,
                    num_restarts=p.num_restarts)
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
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )

cum_regrets = []
num_trials = 4
for trial in range(num_trials):
#     r = SSPBayesOptTrial().run(**{'function_name':'branin-hoo', 'algorithm':'ssp-mi'})
    r = SSPBayesOptTrial().run(**{'function_name':'himmelblau', 'algorithm':'ssp-mi'})
    cum_reg = np.divide(np.cumsum(r['regret']), np.arange(1, len(r['regret'])+1))
    cum_regrets.append(cum_reg)

mu_reg = np.mean(cum_regrets, axis=0)
std_reg = np.std(cum_regrets, axis=0) / np.sqrt(num_trials)
plt.plot(mu_reg)
plt.plot(mu_reg - 1.96 * std_reg, ls='--')
plt.plot(mu_reg + 1.96 * std_reg, ls='--')
plt.show()
