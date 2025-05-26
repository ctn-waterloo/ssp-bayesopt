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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import figure_utils as utils

import numpy.matlib as matlib
from matplotlib.gridspec import GridSpec
import sys, os
# sys.path.insert(1, os.path.dirname(os.getcwd()))
# os.chdir("..")
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib import rc
import sys
import zipfile
import pandas as pd
import os
from matplotlib import cm
import pickle
import re
plt.rcdefaults()

shift = 100


def get_mean_and_ci(raw_data, n=3000, p=0.95):
    """
    Gets the mean and 95% confidence intervals of data *see Note
    NOTE: data has to be grouped along rows, for example: having 5 sets of
    100 data points would be a list of shape (5,100)
    """
    sample = []
    upper_bound = []
    lower_bound = []
    sets = np.array(raw_data).shape[0]  # pylint: disable=E1136
    data_pts = np.array(raw_data).shape[1]  # pylint: disable=E1136
    print("Mean and CI calculation found %i sets of %i data points" % (sets, data_pts))
    raw_data = np.array(raw_data)
    for i in range(data_pts):
        data = raw_data[:, i]
        index = int(n * (1 - p) / 2)
        samples = np.random.choice(data, size=(n, len(data)))
        r = [np.mean(s) for s in samples]
        r.sort()
        ci = r[index], r[-index]
        sample.append(np.mean(data))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])

    data = {"mean": sample, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return data


def read(path):
    data = []
    if os.path.exists(path):
        for fn in os.listdir(path):
            fn = os.path.join(path, fn)
            if fn.endswith('.zip'):
                z = zipfile.ZipFile(fn)
                for name in z.namelist():
                    try:
                        if name.endswith('.txt'):
                            data.append(text(z.open(name)))
                        elif name.endswith('.npz'):
                            data.append(npz(z.open(name)))
                    except:
                        print('Error reading file "%s" in "%s"' % (name, fn))
                z.close()
            else:
                try:
                    if fn.endswith('.txt'):
                        data.append(text(fn))
                    elif fn.endswith('.npz'):
                        data.append(npz(fn))
                except:
                    print('Error reading file "%s"' % fn)
    return data


def text(fn):
    if not hasattr(fn, 'read'):
        with open(fn) as f:
            text = f.read()
    else:
        text = fn.read()
    d = {}
    try:
        import numpy
        d['array'] = numpy.array
        d['nan'] = numpy.nan
    except ImportError:
        numpy = None
    exec(text, d)
    del d['__builtins__']
    if numpy is not None:
        if d['array'] is numpy.array:
            del d['array']
        if d['nan'] is numpy.nan:
            del d['nan']
    return d


def npz(fn):
    import numpy as np
    d = {}
    f = np.load(fn, allow_pickle=True)
    for k in f.files:
        if k != 'ssp_space':
            d[k] = f[k]
            if d[k].shape == ():
                d[k] = d[k].item()
    return d


class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('agent', agent='ss-multi')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=1)
        self.param('ssp dim', ssp_dim=271)
        self.param('num agents', n_agents=2)
        self.param('traj len', traj_len=2)
        self.param('x dim', x_dim=2)

    
    def evaluate(self, p):   
        def target(x, info=None):
            x = x.reshape(p.traj_len, p.n_agents,  p.x_dim)
            x = np.concatenate((np.zeros(x[:1,:,:].shape), x), axis=0)
            return -np.sum(np.linalg.norm(x[1:,:, :] - x[:-1,:, :], axis=-1)) + shift

     

        data_dim = p.traj_len * p.x_dim * p.n_agents
        bounds = np.stack([-np.ones(data_dim), np.ones(data_dim)]).T

        optimizer = ssp_bayes_opt.BayesianOptimization(f=target, bounds=bounds,
                                                           verbose=p.verbose,
                                                           random_state=p.seed,
                                                           sampling_seed=p.seed)

        optimizer.maximize(init_points=p.num_init_samples, n_iter=p.num_samples,
                               agent_type=p.agent, #ssp-multi
                               ssp_dim=p.ssp_dim, x_dim=p.x_dim,
                               traj_len=p.traj_len, n_agents=p.n_agents,
                               decoder_method='direct-optim',
                               length_scale=1,#,#[0.5,0.5],#[0.5,0.5,0.5],
                               time_length_scale=0.5,
                               gamma_c=0,
                               beta_ucb=1.,#num_perimeter_samples=500,
                               rng=p.seed
                               )


        vals = np.zeros((p.num_init_samples + p.num_samples,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        regrets = 0 - vals
        print(optimizer.max)
        print(regrets[-1])
        #print(optimizer.ys)

        
        return dict(
            regret=regrets,
            sample_locs=sample_locs,
            times = optimizer.times,
            budget=p.num_samples,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
        )



if __name__=='__main__':
    parser = ArgumentParser()

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=str, default=271)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=50)
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=10)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data_mindist_startpt')
    
    args = parser.parse_args()

    random.seed(1)
    
    
    # n_agts_list = [1]
    # traj_len_list = [2]
    
    n_agts_list = [2,2,3,4,5,6,7]
    traj_len_list = [2,3,4,5,5,6,7]
    

    x_dim = 2
    for agent in ['ssp-multi', 'gp-matern-multi']: #'ssp-multi',
        for n_agents, traj_len in zip(n_agts_list,traj_len_list):
            nums=f'n_agents{n_agents}-traj_len{traj_len}'

            data_path = os.path.join(os.getcwd(),
                                 args.data_dir, nums, agent ) #+ '_no-perim'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            seeds = [random.randint(1, 100000) for _ in range(args.num_trials)]
            for seed in seeds:
                params = {'num_samples':args.num_samples,
                          'data_format':'npz',
                          'data_dir':data_path,
                          'seed':seed,
                          'verbose':False,
                          'ssp_dim':args.ssp_dim,
                          'agent':agent,
                          'x_dim':x_dim,
                          'traj_len':traj_len,
                          'n_agents':n_agents,
                          }
                r = SamplingTrial().run(**params)
                

    plt_data = {}
    agt_name_map = {'ssp-multi':'ssp-multi', 'ssp-multi_no-perim':'ssp-multi_no-perim', 'gp-matern-multi':'gp-matern'}
    budget = 60
    plt_agt_list = [ 'ssp-multi', 'gp-matern-multi']
    for agt in plt_agt_list: 
        plt_data[agt_name_map[agt]]={}
        for i, (n_agents, traj_len) in enumerate(zip(n_agts_list,traj_len_list)):
            plt_data[agt_name_map[agt]][i] = {'best': {}, 'cumreg': {}}
            try:
                data = pd.DataFrame(read(os.path.join(os.getcwd(),
                                         args.data_dir,f'n_agents{n_agents}-traj_len{traj_len}',agt)))

                vals = np.array([data['vals'][k][:budget] for k in range(len(data['vals'])) ]) 
                regrets = shift - vals
                num_trials = regrets.shape[0]
                best_vals = np.minimum.accumulate(regrets,axis=1)

                plt_data[agt_name_map[agt]][i]['best'] = get_mean_and_ci(best_vals).copy()
                
                cum_regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1, budget + 1), num_trials, 1))

                plt_data[agt_name_map[agt]][i]['cumreg'] = get_mean_and_ci(cum_regrets).copy()
            except Exception as e:
                print(e)
                plt_data[agt_name_map[agt]][i]['best'] = None
                plt_data[agt_name_map[agt]][i]['cumreg'] = None

    cols = {"ssp-multi": utils.blues[0], "ssp-multi_no-perim": utils.oranges[0],
            "gp-sinc": utils.greens[0], "gp-matern": utils.reds[0],
            "rff": utils.yellows[2],
            "ssp-hex_nengo-loihi-sim": utils.blues[1],
            "ssp-hex_nengo-spinnaker": utils.blues[2],
            "ssp-hex_nengo-loihi": utils.blues[1]}

    linestys = {"ssp-multi": '-', "ssp-multi_no-perim": '--',
                "gp-sinc": ':', "gp-matern": '-.',
                "rff": (0, (3, 1, 1, 1, 1, 1)),
                "ssp-hex_nengo-loihi-sim": '-.',
                "ssp-hex_nengo-spinnaker": ':',
                "ssp-hex_nengo-loihi": '--'}
    fig, axs = plt.subplots(len(n_agts_list),2,figsize=(10, 3*len(n_agts_list)), gridspec_kw={'hspace':0.8})
    # axs = np.array([[axs]])
    for agt in plt_agt_list:
        for i in range(len(n_agts_list)):
            axs[i,0].fill_between(np.arange(1, budget + 1), 
                                  plt_data[agt][i]['best']["upper_bound"],
                                  plt_data[agt][i]['best']["lower_bound"], alpha=.2,
                                color=cols[agt])
            axs[i,0].plot(np.arange(1, budget + 1), 
                          plt_data[agt][i]['best']["mean"],
                          color=cols[agt], label=agt,
                        linestyle=linestys[agt])
            if i==0:
                axs[i,0].set_title(f'Best score \n n_agents={n_agts_list[i]}, traj_len={traj_len_list[i]}')
            else:
                axs[i,0].set_title(f'n_agents={n_agts_list[i]}, traj_len={traj_len_list[i]}')

            
            axs[i,1].fill_between(np.arange(1, budget + 1), 
                                  plt_data[agt][i]['cumreg']["upper_bound"],
                                  plt_data[agt][i]['cumreg']["lower_bound"], alpha=.2,
                                color=cols[agt])
            axs[i,1].plot(np.arange(1, budget + 1), 
                          plt_data[agt][i]['cumreg']["mean"],
                          color=cols[agt], label=agt,
                        linestyle=linestys[agt])
            if i==0:
                axs[i,1].set_title(f'Cumulative regret \n n_agents={n_agts_list[i]}, traj_len={traj_len_list[i]}')
            else:
                axs[i,1].set_title(f'n_agents={n_agts_list[i]}, traj_len={traj_len_list[i]}')


    axs[0,1].legend()
    utils.save(fig, 'test-traj-zerofun-reg-ls1ts05.png')



