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

    def remove_disconnected(self, graph, operations):
        operations = np.array(operations)
        non_dead_nodes = [i for i in range(graph.shape[0]) if i == graph.shape[0] - 1 or np.any(graph[i, :])]
        graph = graph[np.ix_(non_dead_nodes, non_dead_nodes)]
        operations = operations[non_dead_nodes]

        # # Step 1: Remove disconnected subgraphs that don't contain the input node (index 0)
        # def find_reachable_nodes(start_node, adj_matrix):
        #     visited = set()
        #     stack = [start_node]
        #     while stack:
        #         node = stack.pop()
        #         if node not in visited:
        #             visited.add(node)
        #             stack.extend(np.where(adj_matrix[node] > 0)[0])
        #     return visited
        #
        # # Keep only the reachable nodes
        # reachable_from_input = find_reachable_nodes(0, graph)
        # valid_nodes = sorted(reachable_from_input)
        # # if len(valid_nodes) == 1: # nothing is reachable, just return smallest valid graph
        # #     graph = np.array([[0,1],[0,0]])
        # #     operations = [self.INPUT, self.OUTPUT]
        # #     return graph, operations.tolist()
        #
        # if (graph.shape[0] - 1) not in reachable_from_input:
        #     # raise ValueError("The graph is entirely disconnected from the output node.")
        #     graph[valid_nodes[-1],-1] = 1
        #     valid_nodes.append(graph.shape[0] - 1)
        #
        # graph = graph[np.ix_(valid_nodes, valid_nodes)]
        # operations = operations[valid_nodes]
        #
        # # Step 2: Remove dead-end nodes (rows with all zeros except last row)
        # non_dead_nodes = [i for i in range(graph.shape[0]) if i == graph.shape[0] - 1 or np.any(graph[i, :])]
        # graph = graph[np.ix_(non_dead_nodes, non_dead_nodes)]
        # operations = operations[non_dead_nodes]
        return graph, operations.tolist()

    def __call__(self, input_graphs, info=None, score="validation_accuracy"):
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

            matrix, operations = self.remove_disconnected(matrix,operations)
            model_spec = api.ModelSpec(matrix=matrix.copy(), ops=operations)
            # print(model_spec)
            # print(self.results.is_valid(model_spec))
            if self.results.is_valid(model_spec):
                outputs[n] = self.results.query(model_spec)[score]
            else:
                outputs[n] = 0
        return outputs


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--nas-data-dir', dest='nas_data_dir', type=str, default='./nas_data')
    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=801)
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--num-init-samples', dest='num_init_samples', type=int, default=10)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float, default=1.)#14.5)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='./data/nasbench')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.nas_data_dir[:2] == './':
        args.nas_data_dir = os.path.join(os.getcwd(), args.nas_data_dir[2:])
    if args.data_dir[:2] == './':
        args.data_dir = os.path.join(os.getcwd(), args.data_dir[2:])

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
                       gamma_c=0.,
                       )
    elapsed_time = time.thread_time_ns() - start

    train_vals = np.zeros((num_init_samples + budget,))
    sample_locs = []

    for i, res in enumerate(optimizer.res):
        train_vals[i] = res['target']
        sample_locs.append(res['params'])
    test_vals = target(np.array(sample_locs), score="test_accuracy")

    regrets = target.best_final_accuracy - test_vals
    cum_regrets = np.divide(np.cumsum(regrets), np.arange(1,regrets.shape[0]+1))


    best_vals = np.maximum.accumulate(test_vals)
    print(optimizer.max)
    print(best_vals[-1])
    # for test_vals. want >94, ideal is 94.32; getting 93.78
    # for train_vals. want >94.6, ideal is 94.9; getting 94.1
    # fig=plt.figure();plt.plot(-best_vals);fig.savefig(f"{args.task_id}_sspbo.png");plt.show()


    np.savez(os.path.join(args.data_dir, f"nasbench_seed{args.seed}.npz"),
             test_vals=test_vals, train_vals=train_vals,
             sample_locs=sample_locs, best_vals=best_vals, regrets=regrets, cum_regrets=cum_regrets,
             times=optimizer.times, elapsed_time=elapsed_time,
             args=args)

    # cum_regrets = np.divide(np.cumsum(regrets), matlib.repmat(range(1, regrets.shape[0] + 1), 1, 1))
    # plt.figure()
    # plt.plot(cum_regrets)
    # plt.show()