import matplotlib.pyplot as plt
import seaborn as sns
import figure_utils as utils
import time
import numpy as np

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

# # Apply the default theme
# sns.set_theme()

def objective(x):
 return (1.4 - 3.0 * x) * np.sin(18.0 * x)

target = objective
pbounds = np.array([[0, 1.2 ]])
ls = 0.1
beta_ucb=5
gamma_c=0.1
optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                               bounds=pbounds, 
                                               verbose=True,
                                               sampling_seed=0)
start = time.thread_time_ns()
optimizer.maximize(init_points=1, 
                   n_iter=4,
                   num_restarts=1,
                   agent_type='ssp-hex',
                   ssp_dim=75,decoder_method='from-set', length_scale=ls,beta_ucb=beta_ucb,gamma_c=gamma_c)
elapsed_time = time.thread_time_ns() - start

vals = np.zeros((5,))
sample_locs = []

for i, res in enumerate(optimizer.res):
    vals[i] = res['target']
    sample_locs.append(res['params'])
    
fig, axs = plt.subplots(1,3, figsize=(7.1,2))
for i in range(3):
    axs[i].set_ylim([-3,2.5])
    axs[i].set_xlabel('$x$')

samples=[np.array(sample_locs).reshape(-1), np.array(vals)]
xs = np.linspace(pbounds[0,0],pbounds[0,1], 100).reshape(-1,1)
phis = optimizer.agt.encode(xs)

axs[0].plot(xs,target(xs), "-", color=utils.grays[2], label='$f(x)$')
kern = phis  @ optimizer.agt.encode(np.array(samples[0][0]).reshape(1,-1)).T
axs[0].plot(xs.reshape(-1), samples[1][0]*kern.reshape(-1), color=utils.blues[2], label='$f(x_0)[\phi(x_0) \cdot \phi(x)]$')
axs[0].plot(samples[0][0], samples[1][0], "o", color=utils.reds[1], label='Sample point $x_0$')
axs[0].legend()

mu, var = optimizer.agt.blr.predict(phis)
axs[1].plot(xs,target(xs), "-", color=utils.grays[2])
axs[1].plot(xs.reshape(-1), mu.reshape(-1), color=utils.blues[1], label='BLR mean $\mu$')
axs[1].fill_between(xs.reshape(-1),  mu.reshape(-1)-np.sqrt(var), 
                 mu.reshape(-1)+np.sqrt(var), alpha=0.5, edgecolor=None, facecolor=utils.blues[2], label='BLR sdev $\sigma$')
   
axs[1].plot(samples[0], samples[1], "o", color=utils.reds[1])
axs[1].legend()   

optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                               bounds=pbounds, 
                                               verbose=True,
                                               sampling_seed=0)
start = time.thread_time_ns()
optimizer.maximize(init_points=1, 
                   n_iter=19,
                   num_restarts=1,
                   agent_type='ssp-hex',
                   ssp_dim=75,decoder_method='from-set', length_scale=ls,beta_ucb=beta_ucb,gamma_c=gamma_c)
elapsed_time = time.thread_time_ns() - start

vals = np.zeros((20,))
sample_locs = []

for i, res in enumerate(optimizer.res):
    vals[i] = res['target']
    sample_locs.append(res['params'])
    
samples=[np.array(sample_locs).reshape(-1), np.array(vals)]
phis = optimizer.agt.encode(xs)
mu, var = optimizer.agt.blr.predict(phis)
axs[2].plot(xs,target(xs), "-", color=utils.grays[2])
axs[2].plot(xs.reshape(-1), mu.reshape(-1), color=utils.blues[1])
axs[2].fill_between(xs.reshape(-1),  mu.reshape(-1)-np.sqrt(var), 
                 mu.reshape(-1)+np.sqrt(var), alpha=0.5, edgecolor=None, facecolor=utils.blues[2])
   
axs[2].plot(samples[0], samples[1], "o", color=utils.reds[1])
        
