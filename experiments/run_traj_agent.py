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


import ssp_bayes_opt

import importlib

importlib.reload(ssp_bayes_opt)

budget=200
xstar = np.array([[2.5,-3],[5,2],[8,6]])
xstarfull = np.vstack([np.zeros(2),xstar])
traj_len = xstar.shape[0]
pt_dim = xstar.shape[1]
data_dim = traj_len*pt_dim
bounds = 10*np.stack([-np.ones(data_dim), np.ones(data_dim)]).T

def target(x):
    x = np.vstack([np.zeros(2),x.reshape(traj_len,pt_dim)])
    return -np.sum(np.sqrt(np.sum((x - xstarfull)**2, axis=1)))

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('ssp dim', ssp_dim=151)
    
    def evaluate(self, p):   
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=bounds, 
                                                       verbose=p.verbose)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, n_iter=budget,
                           agent_type='ssp-traj',ssp_dim=p.ssp_dim, x_dim=pt_dim, traj_len=traj_len,
                           length_scale=3.9)
        elapsed_time = time.thread_time_ns() - start

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        regrets = 0 - vals
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

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=str, default=151)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=100)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='/home/ns2dumon/Documents/ssp-bayesopt/experiments/data/')



    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(args.data_dir,'trajectory-agent')
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



from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = plt.axes(xlim=(-10, 10), ylim=(-10, 10))
ax.plot(xstarfull[:,0],xstarfull[:,1],'-g')
line, = ax.plot([], [], lw=3)

def init():
    line.set_data([], [])
    return line,
def animate(i):
    best_idx = np.argmax(r['vals'][:i+1])
    traj = r['sample_locs'][best_idx].reshape(traj_len, pt_dim)
    traj = np.vstack([np.zeros(2),traj])
    line.set_data(traj[:,0], traj[:,1])
    return line,

anim = FuncAnimation(fig, animate, init_func=init,
                               frames=budget-1, interval=20, blit=True)

anim.save('match_traj.gif', writer='imagemagick')

