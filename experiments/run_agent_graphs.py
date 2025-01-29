from nasbench import api
import json
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import qmc
import numpy.matlib as matlib

from argparse import ArgumentParser
import os
import os.path
import random
import ssp_bayes_opt

import numpy as np


# Requries installing https://github.com/google-research/nasbench
# and downloading nasbench_only108.tfrecord
# See https://github.com/google-research/nasbench
# Recommend using seperate env

# git clone https://github.com/google-research/nasbench
# cd nasbench
# pip install -e .
# python nasbench/scripts/generate_graphs.py
# wget https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
# (Optional: move generated_graphs.json and nasbench_only108.tfrecord somewhere nice)
# Suppy the path to nasbench (or wherever you put generated_graphs.json and nasbench_only108.tfrecord)
# via --nas-data-dir

class NASBench:
    def __init__(self, data_dir='nasbench', results=None, graphs=None):
        """
        Initialize the NASBench
        """
        if results is None:
            self.results = api.NASBench(os.path.join(data_dir, 'nasbench_only108.tfrecord'))
        else:
            self.results = results
        if graphs is None:
            with open(os.path.join(data_dir, 'generated_graphs.json')) as f:
                self.graphs = json.load(f)
        else:
            self.graphs = graphs

        self.best_final_validation_accuracy = -1
        for ahash in self.results.computed_statistics.keys():
            nseeds = len(self.results.computed_statistics[ahash][108])
            final_validation_accuracy = 0
            for seed in range(nseeds):
                final_validation_accuracy += self.results.computed_statistics[ahash][108][seed][
                    'final_validation_accuracy']
            final_validation_accuracy /= nseeds
            if final_validation_accuracy > self.best_final_validation_accuracy:
                self.best_final_validation_accuracy = final_validation_accuracy

        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.CONV1X1 = 'conv1x1-bn-relu'
        self.CONV3X3 = 'conv3x3-bn-relu'
        self.MAXPOOL3X3 = 'maxpool3x3'
        self.operations = [self.CONV1X1, self.CONV3X3, self.MAXPOOL3X3]

        self.max_nodes = 9
        self.max_edges = 7
        self.num_ops = len(self.operations)
        self.n_graphs = len(self.graphs)
        self.graph_hashs = list(self.graphs.keys())
        self.rolled_len = int(0.5 * (self.max_edges - 1) * self.max_edges)

    def sample(self, num_points):
        samples = np.zeros((num_points, self.rolled_len))
        for i in range(num_points):
            retry = True
            while retry:
                idx = np.random.randint(0, self.n_graphs)
                _matrix = np.array(self.graphs[self.graph_hashs[idx]][0])
                if (_matrix.shape[0] == self.max_edges):
                    retry = False
            ops = np.array(self.graphs[self.graph_hashs[idx]][1])
            ops[ops > 0] = ops[ops > 0] + 1
            ops[ops == -1] = 1
            ops[ops == -2] = 0
            ops = ops.reshape(-1, 1)
            # matrix = np.zeros((self.max_edges, self.max_edges))
            # matrix[:_matrix.shape[0], :_matrix.shape[0]] = _matrix * ops
            matrix = _matrix * ops
            samples[i, :] = np.concatenate([matrix[i, i + 1:] for i in range(matrix.shape[0] - 1)])
        return samples

    def __call__(self, input_graphs, info=None):
        input_graphs = np.atleast_2d(input_graphs)
        outputs = np.zeros(input_graphs.shape[0])
        for n, input_graph in enumerate(input_graphs):

            matrix = np.zeros((self.max_edges, self.max_edges),dtype=int)
            operations = [None] * target.max_edges
            for i in range(self.max_edges - 1):
                layer_i = np.concatenate([np.zeros(i + 1),
                                          input_graph[int((self.max_edges - 1 + 0.5 * (1 - i)) * i):int(
                                              (self.max_edges - 1 - 0.5 * i) * (i + 1))]])
                if np.any(layer_i > 1):
                    operations[i] = self.operations[int(np.round(np.mean(layer_i[layer_i > 1]))) - 2]
                else:
                    operations[i] = self.operations[0]
                matrix[i, :] = layer_i > 0
            operations[0] = self.INPUT
            operations[-1] = self.OUTPUT

            model_spec = api.ModelSpec(matrix=matrix.copy(), ops=operations)
            print(model_spec)
            print(self.results.is_valid(model_spec))
            if self.results.is_valid(model_spec):
                outputs[n] = self.results.query(model_spec)['validation_accuracy']
            else:
                outputs[n] = 0
        return outputs


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nas-data-dir', dest='nas_data_dir', type=str, default='./experiments/nas_data')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=100)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=1000)
    parser.add_argument('--num-init-samples', dest='num_init_samples', type=int, default=10)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float, default=1.0)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.nas_data_dir[:2] == './':
        args.nas_data_dir = os.path.join(os.getcwd(), args.nas_data_dir[2:])

    nasbench = api.NASBench(os.path.join(args.nas_data_dir, 'nasbench_only108.tfrecord'))
    with open(os.path.join(args.nas_data_dir, 'generated_graphs.json')) as f:
        graphs = json.load(f)
    num_init_samples = args.num_init_samples
    budget = args.num_samples
    target = NASBench(data_dir=args.nas_data_dir, results=nasbench, graphs=graphs)
    pbounds = np.tile(np.array([0, target.num_ops + 2]), target.rolled_len).reshape(-1, 2)
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
                       agent_type='ssp-nas-graph',
                       ssp_dim=args.ssp_dim,
                       beta_ucb=args.beta_ucb,
                       )
    elapsed_time = time.thread_time_ns() - start

    vals = np.zeros((num_init_samples + budget,))
    sample_locs = []

    for i, res in enumerate(optimizer.res):
        vals[i] = res['target']
        sample_locs.append(res['params'])

    regrets = target.best_final_validation_accuracy - vals
    cum_regrets = np.divide(np.cumsum(regrets), np.arange(1,regrets.shape[0]+1))
    print(optimizer.max)
    print(target.best_final_validation_accuracy)

    # cum_regrets = np.divide(np.cumsum(regrets), matlib.repmat(range(1, regrets.shape[0] + 1), 1, 1))
    # plt.figure()
    # plt.plot(cum_regrets)
    # plt.show()