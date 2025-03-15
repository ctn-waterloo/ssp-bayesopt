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
import pytry
from run_agent import neuron_types, sim_types
# Requires installing NAS-bench: https://github.com/google-research/nasbench
# and downloading nasbench_only108.tfrecord
# See https://github.com/google-research/nasbench
# Recommend using separate env

# git clone git@github.com:google-research/nasbench.git
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
        Initialize the NASBench objective function
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

        self.best_final_accuracy = -1
        for ahash in self.results.computed_statistics.keys():
            nseeds = len(self.results.computed_statistics[ahash][108])
            final_accuracy = 0
            for seed in range(nseeds):
                final_accuracy += self.results.computed_statistics[ahash][108][seed][
                    'final_test_accuracy']
            final_accuracy /= nseeds
            if final_accuracy > self.best_final_accuracy:
                self.best_final_accuracy = final_accuracy

        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.CONV1X1 = 'conv1x1-bn-relu'
        self.CONV3X3 = 'conv3x3-bn-relu'
        self.MAXPOOL3X3 = 'maxpool3x3'
        self.operations = [self.INPUT, self.CONV1X1, self.CONV3X3, self.MAXPOOL3X3, self.OUTPUT]

        self.max_conns = 9
        self.max_layers = 7
        self.num_ops = len(self.operations)
        self.n_graphs = len(self.graphs)
        self.graph_hashs = list(self.graphs.keys())
        self.rolled_len = int(0.5 * (self.max_layers - 1) * self.max_layers)
        # self.invalid_calls = []

    def sample(self, num_points):
        samples = np.zeros((num_points, self.rolled_len + self.max_layers))
        for i in range(num_points):
            # retry = True
            # while retry:
            #     idx = np.random.randint(0, self.n_graphs)
            #     _matrix = np.array(self.graphs[self.graph_hashs[idx]][0])
            #     if (_matrix.shape[0] == self.max_layers):
            #         retry = False
            idx = np.random.randint(0, self.n_graphs)
            matrix = np.array(self.graphs[self.graph_hashs[idx]][0])
            ops = np.array(self.graphs[self.graph_hashs[idx]][1])
            ops[ops >= 0] = ops[ops >= 0] + 1
            ops[0] = 0
            ops[-1] = self.num_ops - 1

            n_layers = matrix.shape[0]
            if (n_layers < self.max_layers):
                matrix = np.pad(matrix, (0, self.max_layers - n_layers), )
                ops = np.pad(ops, (0, self.max_layers - n_layers),
                             constant_values=0)

            # ops = ops.reshape(-1, 1)
            # matrix = np.zeros((self.max_layers, self.max_layers))
            # matrix[:_matrix.shape[0], :_matrix.shape[0]] = _matrix * ops
            # matrix = matrix * ops
            samples[i, :] = np.concatenate([matrix[i, i + 1:] for i in range(matrix.shape[0] - 1)] + [ops])
        return samples

    def remove_disconnected(self, graph, operations):
        operations = np.array(operations)
        non_dead_nodes = [i for i in range(graph.shape[0]) if i == graph.shape[0] - 1 or np.any(graph[i, :])]
        graph = graph[np.ix_(non_dead_nodes, non_dead_nodes)]
        operations = operations[non_dead_nodes]

        # # Step 1: Remove disconnected subgraphs that don't contain the input node (index 0)
        def find_reachable_nodes(start_node, adj_matrix):
            visited = set()
            stack = [start_node]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    stack.extend(np.where(adj_matrix[node] > 0)[0])
            return visited

        # Keep only the reachable nodes
        reachable_from_input = find_reachable_nodes(0, graph)
        valid_nodes = sorted(reachable_from_input)
        # if len(valid_nodes) == 1: # nothing is reachable, just return smallest valid graph
        #     graph = np.array([[0,1],[0,0]])
        #     operations = [self.INPUT, self.OUTPUT]
        #     return graph, operations.tolist()

        if (graph.shape[0] - 1) not in reachable_from_input:
            # raise ValueError("The graph is entirely disconnected from the output node.")
            graph[valid_nodes[-1],-1] = 1
            valid_nodes.append(graph.shape[0] - 1)

        graph = graph[np.ix_(valid_nodes, valid_nodes)]
        operations = operations[valid_nodes]

        # Step 2: Remove dead-end nodes (rows with all zeros except last row)
        non_dead_nodes = [i for i in range(graph.shape[0]) if i == graph.shape[0] - 1 or np.any(graph[i, :])]
        graph = graph[np.ix_(non_dead_nodes, non_dead_nodes)]
        operations = operations[non_dead_nodes]
        return graph, operations.tolist()

    def transform_samples(self, input_graphs):
        input_graphs = np.atleast_2d(input_graphs)
        outputs = np.zeros(input_graphs.shape[0])

        matrices = []
        opts = []
        for n, input_graph in enumerate(input_graphs):

            matrix = np.zeros((self.max_layers, self.max_layers),dtype=int)
            op = [None] * target.max_layers
            for i in range(self.max_layers - 1):
                layer_i = np.concatenate([np.zeros(i + 1),
                                          input_graph[int((self.max_layers - 1 + 0.5 * (1 - i)) * i):int(
                                              (self.max_layers - 1 - 0.5 * i) * (i + 1))]])
                op[i] = int(input_graph[self.rolled_len+i])
                matrix[i, :] = layer_i > 0
            op[0] = 0
            op[-1] = 4

            matrix, operations = self.remove_disconnected(matrix,operations)
            matrices.append(matrix)
            opts.append(op)
        return outputs

    def __call__(self, input_graphs, info=None, score="validation_accuracy"):
        input_graphs = np.atleast_2d(input_graphs)
        outputs = np.zeros(input_graphs.shape[0])
        for n, input_graph in enumerate(input_graphs):

            matrix = np.zeros((self.max_layers, self.max_layers),dtype=int)
            operations = [None] * target.max_layers
            for i in range(self.max_layers - 1):
                layer_i = np.concatenate([np.zeros(i + 1),
                                          input_graph[int((self.max_layers - 1 + 0.5 * (1 - i)) * i):int(
                                              (self.max_layers - 1 - 0.5 * i) * (i + 1))]])
                operations[i] = self.operations[int(input_graph[self.rolled_len+i])]

                matrix[i, :] = layer_i > 0
            operations[0] = self.INPUT
            operations[-1] = self.OUTPUT

            matrix, operations = self.remove_disconnected(matrix,operations)
            model_spec = api.ModelSpec(matrix=matrix.copy(), ops=operations)
            # print(model_spec)
            # print(self.results.is_valid(model_spec))
            if self.results.is_valid(model_spec):
                outputs[n] = self.results.query(model_spec)[score]
            else:
                outputs[n] = 0
        return outputs


class SamplingTrial(pytry.Trial):
    def __init__(self, target):
        super(SamplingTrial, self).__init__()
        self.target = target

    def params(self):
        self.param('task_id', function_name='xgboost_opt')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('UCB Beta', beta_ucb=1.0)
        self.param('MI gamma_c', gamma=0.0)
        self.param('ssp dim', ssp_dim=151)
        self.param('trial number', trial_num=None)
        self.param('use nengo', nengo=False)
        self.param('nengo backend', backend='cpu')
        self.param('num neurons', num_neurons=8)
        self.param('sim time', sim_time=2.5)

    def evaluate(self, p):

        num_init_samples = p.num_init_samples
        budget = p.num_samples
        pbounds = np.tile(np.array([0, self.target.num_ops + 2]), self.target.rolled_len).reshape(-1, 2)
        # samples = target.sample(10) #test
        # model_spec=target(samples) #test
        if p.nengo:
            sim_time = p.sim_time
            neuron_type = neuron_types['loihilif'] if 'loihi' in p.backend else neuron_types['lif']
            sim_type, sim_args = sim_types[p.backend]

            optimizer = ssp_bayes_opt.NengoBayesianOptimization(f=self.target,
                                                           bounds=pbounds,
                                                           verbose=p.verbose,
                                                           sampling_seed=p.seed)
            start = time.thread_time_ns()
            optimizer.maximize(init_points=p.num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type='ssp-nas-graph',
                               ssp_dim=p.ssp_dim,
                               length_scale=1.0,
                               decoder_method='direct-optim',
                               gamma_c=p.gamma,
                               beta_ucb=p.beta_ucb,
                               neurons_per_dim=p.num_neurons,
                               neuron_type=neuron_type,
                               sim_type=sim_type, sim_args=sim_args,
                               sim_time=sim_time, tau=0.05
                               )
            elapsed_time = time.thread_time_ns() - start
        else:
            optimizer = ssp_bayes_opt.BayesianOptimization(f=self.target,
                                                   bounds=pbounds,
                                                   verbose=p.verbose,
                                                   sampling_seed=p.seed)

            start = time.thread_time_ns()
            optimizer.maximize(init_points=num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type='ssp-nas-graph',
                               ssp_dim=p.ssp_dim,
                               beta_ucb=p.beta_ucb,
                               gamma_c=p.gamma,
                               length_scale=1.0,
                               decoder_method='direct-optim',
                               )
            elapsed_time = time.thread_time_ns() - start

            # ex_graph = target.sample(1)
            # ex_ssp = optimizer.agt.encode(ex_graph)
            # ex_graph_hat = optimizer.agt.decode(ex_ssp)

            train_vals = np.zeros((num_init_samples + budget,))
            sample_locs = []

            for i, res in enumerate(optimizer.res):
                train_vals[i] = res['target']
                sample_locs.append(res['params'])
            test_vals = self.target(np.array(sample_locs), score="test_accuracy")

            regrets = self.target.best_final_accuracy - test_vals

            return dict(
                regret=regrets,
                sample_locs=sample_locs,
                elapsed_time=elapsed_time,
                times=optimizer.times,  # selected_len_scale = optimizer.length_scale,
                full_times=optimizer.full_times,
                memory=optimizer.memory,
                budget=budget,
                vals=test_vals,
                test_vals=test_vals,
                train_vals=train_vals,
                mus=None,
                variances=None,
                acquisition=None,
                total_time=optimizer.total_time
            )

            # best_vals = np.maximum.accumulate(test_vals)
            #
            # print(np.max(train_vals[args.num_init_samples:]))
            # print(np.max(test_vals[args.num_init_samples:]))

            # import json
            # import pickle
            # def save_graphs(self, optim_samples, n_samples=500):
            #     samples = self.sample(n_samples)
            #     samples = np.vstack([optim_samples,
            #                         samples])
            #     outs = []
            #     outs2 = []
            #     for n, input_graph in enumerate(samples):
            #         ssp_graph = optimizer.agt.encode(input_graph)
            #
            #         matrix = np.zeros((self.max_layers, self.max_layers),dtype=int)
            #         operations = np.zeros(self.max_layers)
            #         for i in range(self.max_layers - 1):
            #             layer_i = np.concatenate([np.zeros(i + 1),
            #                                       input_graph[int((self.max_layers - 1 + 0.5 * (1 - i)) * i):int(
            #                                           (self.max_layers - 1 - 0.5 * i) * (i + 1))]])
            #             operations[i] = int(input_graph[self.rolled_len+i]) + 1
            #
            #             matrix[i, :] = layer_i > 0
            #         operations[0] = 1
            #         operations[-1] = 0
            #         matrix, operations = self.remove_disconnected(matrix, operations)
            #
            #
            #         mat_str = " ".join(list(matrix.flatten().astype('str')))
            #         op_str = " ".join(np.array(operations).astype('int').astype('str'))
            #         outs.append((mat_str,op_str))
            #         outs2.append((matrix, operations, ssp_graph))
            #     with open("nas_graph_plot_data.json", "w") as f:
            #         json.dump(outs, f)
            #
            #     # with open("nas_full_plot_data.json", "w") as f:
            #     #     json.dump(outs2, f)
            #     with open("nas_full_plot_data.pkl", "wb") as f:
            #         pickle.dump(outs2, f)
            #     return outs
            # #
            # #
            # # # sample_locs = np.array(sample_locs)
            # # # outs = save_graphs(target, sample_locs[np.argsort(test_vals)[-50:]], n_samples=950)
            # # outs = save_graphs(target, sample_locs[np.argmax(test_vals)].reshape(1,-1), n_samples=500)
            #
            # # samples_ssps = np.array([optimizer.agt.encode(sam) for sam in sample_locs]).squeeze()
            # # sam_sims = samples_ssps @ samples_ssps[np.argmax(test_vals), :]
            # # sam_sims = sam_sims/sam_sims[np.argmax(test_vals)]
            # # sample_locs = np.array(sample_locs)
            # # outs = save_graphs(target, sample_locs[np.argsort(sam_sims)[-5:]], n_samples=1000)
            # outs = save_graphs(target, sample_locs[np.argmax(test_vals)].reshape(1, -1), n_samples=1000)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nas-data-dir', dest='nas_data_dir', type=str, default='./nas_data')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=201)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--num-init-samples', dest='num_init_samples', type=int, default=20)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float,
                        default=30.0)  # np.log(2/1e-6))#np.log(2/1e-6))#np.log(2/1e-6))
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.0)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')
    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--nengo', action='store_true')
    parser.add_argument('--backend', dest='backend', type=str, default="cpu")  # loihi-sim, loihi

    args = parser.parse_args()

    if args.nas_data_dir[:2] == './':
        args.nas_data_dir = os.path.join(os.getcwd(), args.nas_data_dir[2:])
    if args.data_dir[:2] == './':
        args.data_dir = os.path.join(os.getcwd(), args.data_dir[2:])
    if args.nengo:
        data_path = os.path.join(args.data_dir, 'ssp-nas-graph_nengo-' + args.backend)
    else:
        data_path = os.path.join(args.data_dir, 'ssp-nas-graph')

    nasbench = api.NASBench(os.path.join(args.nas_data_dir, 'nasbench_only108.tfrecord')) # loading this for all trials because it takes a bit of time
    with open(os.path.join(args.nas_data_dir, 'generated_graphs.json')) as f:
        graphs = json.load(f)
    target = NASBench(data_dir=args.nas_data_dir, results=nasbench, graphs=graphs)

    # random.seed(1)
    seeds = [random.randint(1, 100000) for _ in range(args.num_trials)]

    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'num_samples': args.num_samples,
                  'data_format': 'npz',
                  'data_dir': data_path,
                  'seed': seed,
                  'verbose': False,
                  'ssp_dim': args.ssp_dim,
                  'beta_ucb': args.beta_ucb,
                  'gamma': args.gamma,
                  'nengo': args.nengo,
                  'backend': args.backend,
                  }
        r = SamplingTrial(target).run(**params)
