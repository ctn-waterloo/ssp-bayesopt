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


n_agents = 4
traj_len = 5
pt_dim = 2
data_dim = traj_len*pt_dim*n_agents
bounds = np.stack([-np.ones(data_dim), np.ones(data_dim)]).T
space_size = bounds[0,1]-bounds[0,0]

def target(x, info=None):
    x = x.reshape(n_agents, traj_len, pt_dim)
    return -np.sum(np.linalg.norm(x[:,1:,:] - x[:,:-1,:], axis=-1))
    # return -np.sum(np.sqrt(np.sum((x - xstar)**2, axis=2)))

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=1)
        self.param('ssp dim', ssp_dim=271)
    
    def evaluate(self, p):   
        
        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=bounds, 
                                                       verbose=p.verbose)
        
        start = time.thread_time_ns()
        optimizer.maximize(init_points=p.num_init_samples, n_iter=p.num_samples,
                           agent_type='ssp-multi', ssp_dim=p.ssp_dim, x_dim=pt_dim, traj_len=traj_len, n_agents=n_agents,
                           decoder_method='from-set',
                           length_scale=[0.5,0.5],#[0.5,0.5,0.5],
                           time_length_scale=1,
                           gamma_c=0,
                           beta_ucb=1.,
                           )
        elapsed_time = time.thread_time_ns() - start

        init_xs = optimizer.agt.init_xs
        init_phis = optimizer.agt.encode(init_xs)
        decode_dists = np.zeros((init_phis.shape[0], traj_len*n_agents))
        for j in range(init_phis.shape[0]):
            hat_init_xs = optimizer.agt.decode(init_phis[j]).reshape(n_agents, traj_len, pt_dim)
            decode_dists[j,:] = np.linalg.norm(init_xs[j].reshape(n_agents, traj_len, pt_dim)-hat_init_xs,axis=-1).flatten()

        # hat_init_phis = optimizer.agt.encode(hat_init_xs.flatten()[None,:])
        # optimizer.agt.ssp_dim

        print(f"Avg. error of decoding one location a bundle: {np.mean(decode_dists)/space_size} (% of domain size)")
        print(f"Avg. max error of decoding one location a bundle: {np.mean(np.max(decode_dists,axis=-1))/space_size} (% of domain size)")

        vals = np.zeros((p.num_init_samples + p.num_samples,))
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
            budget=p.num_samples,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )



if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=str, default=601)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=50)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')


    
    args = parser.parse_args()

    random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    data_path = os.path.join(os.getcwd(),
                             args.data_dir, 'trajectory-agent-test')
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



# from matplotlib import pyplot as plt
# from matplotlib.animation import FuncAnimation
#
# fig, axs = plt.subplots(1,2)
# axs[0].set_xlim(-10, 10)
# axs[0].set_ylim(-10, 10)
# axs[1].set_xlim(-10, 10)
# axs[1].set_ylim(-10, 10)
# axs[0].plot(xstarfull[0,:,0],xstarfull[0,:,1],'-g')
# axs[1].plot(xstarfull[1,:,0],xstarfull[1,:,1],'-b')
# line1, = axs[0].plot([], [], lw=3)
# line2, = axs[1].plot([], [], lw=3)
#
# def init():
#     line1.set_data([], [])
#     line2.set_data([], [])
#     return line1,line2
# def animate(i):
#     best_idx = np.argmax(r['vals'][:i+1])
#     traj = r['sample_locs'][best_idx].reshape(n_agents,traj_len, pt_dim)
#     traj1 = np.vstack([np.zeros(2),traj[0]])
#     line1.set_data(traj1[:,0], traj1[:,1])
#     traj2 = np.vstack([np.zeros(2),traj[1]])
#     line2.set_data(traj2[:,0], traj2[:,1])
#     return line1,line2
#
# anim = FuncAnimation(fig, animate, init_func=init,
#                                 frames=budget-1, interval=20, blit=True)
#
# anim.save('match_traj.gif', writer='imagemagick')

