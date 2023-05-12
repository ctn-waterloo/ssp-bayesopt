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

class ContexualFunction:

    def __init__(self):
        self.data_dim = 2
        self.context_size = 2
        self.dim = self.data_dim +1
        self.bounds = np.array([[-5, 5],[-5, 5], [0,self.context_size]]) #[-5, 10],[-5, 15]
        self.budget = 500

    def reset(self):
        return np.array([[0]])
    
    def _func1(self, x):
        return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100
    
    def _func2(self, x):
        #res= -20*np.exp(-0.2*np.sqrt(0.5*np.sqrt(x[:,0]**2 + x[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[:,0]) + np.cos(2*np.pi*x[:,1]))) + np.exp(1) + 20
        #res = np.cos(x[:,0])*np.cos(x[:,1])*np.exp(-(x[:,0]-np.pi)**2 - (x[:,1]-np.pi)**2 )
        return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100
    
    def __call__(self, xc, info):
        x = xc[:,:self.data_dim]
        context = xc[:,-1]
       # return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100, np.array([[-1]])


        if context == 0:
            return self._func1(x), np.array([[1]])
        elif context==1:
            return self._func2(x), np.array([[0]])
 
    def call_vectorized(self, xc, info):
        x = xc[:,:self.data_dim]
        context = xc[:,-1].reshape(-1)
        # return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100, np.array([[-1]])
        res = np.zeros((x.shape[0]))
        res_c = np.zeros((x.shape[0]))
        res[context==0] = self._func1(x[context==0,:])
        res_c[context==0] = 1
        
        res[context==1] = self._func2(x[context==1,:])
        res_c[context==1] = 0
        return res, res_c

       
    
    def _maximum_value(self, xs):
        res = np.zeros(len(xs))
        context1 = np.array([x[0] == 0 for x in xs])
        res[context1]  = 0.07076226300682818
        res[~context1]  = 0.07076226300682818
        return res

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=200)
        self.param('ssp dim', ssp_dim=250)
        self.param('trial number', trial_num=None)
    
    def evaluate(self, p):        
        target = ContexualFunction()
        pbounds = target.bounds
        budget = target.budget
        #target, pbounds = functions.rescale(target,pbounds)
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                                       bounds=pbounds, 
                                                       verbose=p.verbose,
                                                       sampling_seed=p.seed)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, 
                           n_iter=budget,
                           num_restarts=1,
                           agent_type='ssp-discrete-context',
                           ssp_dim=p.ssp_dim,
                           decoder_method='from-set')
        elapsed_time = time.thread_time_ns() - start

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        regrets = target._maximum_value(optimizer.xs) - vals
        print(optimizer.max)
        
        return dict(
            regret=regrets,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            times = optimizer.times,#selected_len_scale = optimizer.length_scale,
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )



if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=151)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='/home/ns2dumon/Documents/ssp-bayesopt/experiments/data/')



    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir,'context')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'num_samples':args.num_samples,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'verbose':False,
                  'ssp_dim':args.ssp_dim
                  }
        r = SamplingTrial().run(**params)
        
        con1 = [l[-1] ==0 for l in r['sample_locs']]
        con2 = [l[-1] ==1 for l in r['sample_locs']]
        num_samples = np.arange(r['regret'].shape[0])
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(num_samples[con1], r['vals'][con1])
        plt.subplot(2,1,2)
        plt.plot(num_samples[con2], r['vals'][con2])
        
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(num_samples[con1], np.minimum.accumulate(r['regret'][con1]) )
        plt.subplot(2,1,2)
        plt.plot(num_samples[con2], np.minimum.accumulate(r['regret'][con2]) )
