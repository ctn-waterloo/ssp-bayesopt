import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np

import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

# Apply the default theme
sns.set_theme()

def objective(x):
 return (1.4 - 3.0 * x) * np.sin(18.0 * x)

target = objective
pbounds = np.array([[0, 1.2 ]])
budget = 100

optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                               bounds=pbounds, 
                                               verbose=True,
                                               sampling_seed=0)
num_init_samples=20
start = time.thread_time_ns()
optimizer.maximize(init_points=num_init_samples, 
                   n_iter=budget,
                   num_restarts=1,
                   agent_type='ssp-hex',
                   ssp_dim=75,decoder_method='from-set')
elapsed_time = time.thread_time_ns() - start

vals = np.zeros((num_init_samples + budget,))
sample_locs = []

for i, res in enumerate(optimizer.res):
    vals[i] = res['target']
    sample_locs.append(res['params'])
    


def plot_1d_blr(agent, bounds, samples=None):
    xs = np.linspace(bounds[0,0],bounds[0,1], 100).reshape(-1,1)
    phis = agent.encode(xs)
    mu, var = agent.blr.predict(phis)
    plt.figure()
    plt.plot(xs,target(xs), "-", color="grey")
    plt.plot(xs.reshape(-1), mu.reshape(-1), color=sns.color_palette()[0])
    plt.fill_between(xs.reshape(-1),  mu.reshape(-1)-np.sqrt(var), 
                     mu.reshape(-1)+np.sqrt(var), alpha=0.5, edgecolor=sns.color_palette()[0], facecolor=sns.color_palette()[0])
   
    if samples is not None:
        plt.plot(samples[0], samples[1], "ro")
        
plot_1d_blr(optimizer.agt, pbounds,[np.array(sample_locs).reshape(-1), np.array(vals)] )