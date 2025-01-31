import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import qmc
import torch
from argparse import ArgumentParser
import os
import os.path
import random
import ssp_bayes_opt
from pandas import DataFrame
import numpy as np
from mcbo.utils.experiment_utils import get_task_from_id

# Requires installing MCBO: https://github.com/huawei-noah/HEBO/tree/master/MCBO
# and downloading nasbench_only108.tfrecord
# See https://github.com/google-research/nasbench
# Recommend using seperate env

# git clone git@github.com:huawei-noah/HEBO.git
# cd HEBO/MCBO
# conda create -n mcbo_env python=3.8
# conda activate mcbo_env
# pip install -e .
# chmod u+x ./bbox_setup.sh
# ./bbox_setup.sh
#
# Test:
# task_id = 'ackley-53' # try different ones, e.g
# # OR ackley aig_optimization antibody_design mig_optimization pest rna_inverse_fold ackley-53 xgboost_opt aig_optimization_hyp svm_opt
# task = get_task_from_id(task_id)
# search_space = task.get_search_space()
# print(search_space.params)
#
# I found that the aig_optimization and aig_optimization_hyp did not work. Work:
# Got to MCBO/mcbo/utils/experiment_utils.py
# Ctrl-f "aig_optimization", you should see
# task_kwargs = {'designs_group_id': "sin", "operator_space_id": "basic", "objective": "both",
#                        "seq_operators_pattern_id": "basic_w_post_map"}
# Change "sin" to "epfl_arithmetic"
# Same for "aig_optimization_hyp"


class MCBO_task:
    def __init__(self, task_id):
        """
        Initialize an objective function using a task from MCBO
        """
        self.task = get_task_from_id(task_id)
        self.search_space = self.task.get_search_space()
        self.param_names = self.search_space.params
        self.n_cont = len(self.search_space.cont_names)
        self.n_disc = len(self.search_space.disc_names)
        self.n_nominal = len(self.search_space.nominal_names)
        self.n_ordinal = len(self.search_space.ordinal_names)

        self.has_global_optimum = hasattr(self.task, 'global_optimum')
        if self.has_global_optimum:
            self.global_optimum = self.task.global_optimum
        else:
            self.global_optimum = None


        self.n_params = len(self.search_space.params)
        self.bounds = np.zeros((self.n_params, 2))
        self.bounds[:, 0] = [self.search_space.params[p].lb for p in self.param_names]
        self.bounds[:, 1] = [self.search_space.params[p].ub for p in self.param_names]


    def sample(self, num_points):
        df_samples = self.search_space.sample(num_points)
        for p in self.param_names:
            df_samples[p] = self.search_space.params[p].transform(df_samples[p]).cpu().numpy()
        samples = np.array(df_samples).astype(float)
        return samples

    def __call__(self, x, info=None):
        df = DataFrame(data=x, columns=self.param_names)
        for p in self.param_names:
            df[p] = self.search_space.params[p].inverse_transform(torch.tensor(df[p]))
        y = self.task(df)
        return -y  # minimization -> maximization


if __name__ == '__main__':
    parser = ArgumentParser()

    # combinatorial tasks used in paper:
    # ackley antibody_design mig_optimization pest rna_inverse_fold aig_optimization
    # mixed-variable tasks used in paper:
    # ackley-53 aig_optimization_hyp svm_opt xgboost_opt
    parser.add_argument('--task-id', dest='task_id', type=str, default='xgboost_opt')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=600)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--num-init-samples', dest='num_init_samples', type=int, default=20)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float, default=np.log(2/1e-6))#np.log(2/1e-6))
    parser.add_argument('--gamma-c', dest='gamma_c', type=float, default=1.0)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()


    num_init_samples = args.num_init_samples
    budget = args.num_samples
    target = MCBO_task(task_id=args.task_id)
    pbounds = target.bounds
    # samples = target.sample(10) #test
    # model_spec=target(samples) #test
    optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                                   bounds=pbounds,
                                                   verbose=args.verbose,
                                                   sampling_seed=args.seed)

    start = time.thread_time_ns()
    optimizer.maximize(init_points=num_init_samples,
                       n_iter=budget,
                       num_restarts=1,
                       agent_type='ssp-mcbo',
                       ssp_dim=args.ssp_dim,
                       beta_ucb=args.beta_ucb,
                       gamma_c=args.gamma_c
                       )
    elapsed_time = time.thread_time_ns() - start

    vals = np.zeros((num_init_samples + budget,))
    sample_locs = []

    for i, res in enumerate(optimizer.res):
        vals[i] = res['target']
        sample_locs.append(res['params'])

    print(optimizer.max)
    print(np.max(vals[num_init_samples:]))
    if target.has_global_optimum:
        regrets = target.global_optimum - vals
        cum_regrets = np.divide(np.cumsum(regrets), np.arange(1,regrets.shape[0]+1))
        print(target.global_optimum)

    # cum_regrets = np.divide(np.cumsum(regrets), matlib.repmat(range(1, regrets.shape[0] + 1), 1, 1))
    # plt.figure()
    # plt.plot(cum_regrets)
    # plt.show()