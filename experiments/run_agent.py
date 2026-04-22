"""Run a single Bayesian optimization trial using pytry.

Supports SSP-hex, SSP-rand, GP, and RFF agents. Optional nengo/loihi/spinnaker
backends for neuromorphic execution. Results are saved as .npz files via pytry.

Usage:
    python run_agent.py --func himmelblau --agent ssp-hex --num-samples 200
"""
import numpy as np
import pytry
import time
import functions

from argparse import ArgumentParser
import os.path
import random
import nengo

import ssp_bayes_opt

try:
    import nengo_loihi
    loihi_lif = nengo_loihi.LoihiLIF()
    loihi_sim = nengo_loihi.Simulator
    loihi_allocator = nengo_loihi.hardware.allocators.Greedy()
except ImportError:
    print('Failed to import nengo_loihi')
    loihi_lif = None
    loihi_sim = None
    loihi_allocator = None
try:
    import nengo_spinnaker
    spinnaker_sim = nengo_spinnaker.Simulator
except ImportError:
    print('Failed to import nengo_spinnaker')
    spinnaker_sim = None

function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133
}

neuron_types = {
    'lif': nengo.LIF(),
    'lifrate': nengo.LIFRate(),
    'loihilif': loihi_lif,
    'direct': nengo.Direct()
}

#'progress_bar':False
sim_types = {
    'cpu': (nengo.Simulator, {'progress_bar':False}),#'gpu': (nengo_ocl.Simulator, {'progress_bar':False}),
    'loihi-sim': (loihi_sim, {'target':'sim','progress_bar':False}),
    'loihi': (loihi_sim,
               {'target':'loihi','precompute':True,
                 'hardware_options':{
                            "snip_max_spikes_per_step": 300,
                            "allocator": loihi_allocator,
                            "n_chips": 15
                        }}
               ),
    'spinnaker': (spinnaker_sim, {}),
}



def get_args():
    parser = ArgumentParser()

    parser.add_argument('--func', dest='function_name', type=str, default='branin-hoo')
    parser.add_argument('--agent', dest='agent_type', type=str, default='ssp-hex')
    parser.add_argument('--num-samples', dest='num_samples', type=int, default=200)
    parser.add_argument('--beta-ucb', dest='beta_ucb', type=float, default=1.)
    parser.add_argument('--gamma', dest='gamma', type=float, default=0.0)
    parser.add_argument('--decay', action='store_true')

    parser.add_argument('--ssp-dim', dest='ssp_dim', type=int, default=7)
    parser.add_argument('--n-scales', dest='n_scales', type=int, default=-1)
    parser.add_argument('--n-rotates', dest='n_rotates', type=int, default=-1)
    parser.add_argument('--len-scale', dest='len_scale', type=float, default=4)

    parser.add_argument('--num-trials', dest='num_trials', type=int, default=1)
    parser.add_argument('--data-dir', dest='data_dir', type=str, default='data')

    parser.add_argument('--nengo', action='store_true')
    parser.add_argument('--backend', dest='backend', type=str, default="cpu")  # loihi-sim, loihi
    parser.add_argument('--num-neurons', dest='num_neurons', type=int,
                        default=10)  # 7 is max for spinnaker with d=97, 8 for loihi

    return parser.parse_args()

class SamplingTrial(pytry.Trial):
    def params(self):
        self.param('function', function_name='himmelblau')
        self.param('agent_type', agent_type='ssp-hex')
        self.param('num initial samples', num_init_samples=10)
        self.param('number of sample points', num_samples=100)
        self.param('ssp length scale', len_scale=4)
        self.param('UCB Beta', beta_ucb=1.0)
        self.param('MI gamma_c', gamma=0.0)
        self.param('ssp dim', ssp_dim=151)
        self.param('n_scales', n_scales=4)
        self.param('n_rotates', n_rotates=4)
        self.param('trial number', trial_num=None)

        self.param('use nengo', nengo=False)
        self.param('nengo backend', backend='cpu')
        self.param('num neurons', num_neurons=8)
        self.param('sim time', sim_time=2.5)

        self.param('use beta decay', decay=False)
    
    def evaluate(self, p):        
        target, pbounds, _ = functions.factory(p.function_name)
        #target, pbounds = functions.rescale(target,pbounds)
        budget = p.num_samples
        if p.decay:
            var_decay = -p.beta_ucb / budget
        else:
            var_decay = 0
        # was 0 before

        if p.nengo:
            sim_time = p.sim_time
            neuron_type = neuron_types['loihilif'] if 'loihi' in p.backend else neuron_types['lif']
            sim_type, sim_args = sim_types[p.backend]

            optimizer = ssp_bayes_opt.NengoBayesianOptimization(f=target,
                                                           bounds=pbounds,
                                                           verbose=p.verbose,
                                                           sampling_seed=p.seed)
            start = time.thread_time_ns()
            optimizer.maximize(init_points=p.num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type=p.agent_type,
                               ssp_dim=p.ssp_dim,
                               n_scales=p.n_scales,
                               n_rotates=p.n_rotates,
                               length_scale=p.len_scale,
                               decoder_method='direct-optim',
                               gamma_c=p.gamma,
                               # decoder_method='network-optim',
                               beta_ucb=p.beta_ucb,
                               var_decay=var_decay,
                               neurons_per_dim=p.num_neurons,
                               neuron_type=neuron_type,
                               sim_type=sim_type, sim_args=sim_args,
                               sim_time=sim_time, tau=0.05,
                               save_memory=False,
                               partitions=None
                               )
            elapsed_time = time.thread_time_ns() - start
            sim_times = optimizer.sim_times
        else:
            optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                                       bounds=pbounds, 
                                                       verbose=p.verbose,

                                                       sampling_seed=p.seed)
            start = time.thread_time_ns()
            optimizer.maximize(init_points=p.num_init_samples,
                               n_iter=budget,
                               num_restarts=1,
                               agent_type=p.agent_type,
                               ssp_dim=p.ssp_dim,
                               n_scales=p.n_scales,
                               n_rotates=p.n_rotates,
                               length_scale=p.len_scale,
                               decoder_method='direct-optim',
                               gamma_c=p.gamma,
                               # decoder_method='network-optim',
                               beta_ucb=p.beta_ucb,
                               var_decay=var_decay,
                               rand_pad=False,use_jac=True
                               )
            elapsed_time = time.thread_time_ns() - start
            sim_times = None
        

        vals = np.zeros((p.num_init_samples + budget,))
        sample_locs = []
        
        for i, res in enumerate(optimizer.res):
            vals[i] = res['target']
            sample_locs.append(res['params'])
            
        if "styblinski-tang" in p.function_name:
            true_max_val = 39.16599*pbounds.shape[0]
        elif "rastrigin" in p.function_name:
            true_max_val = 0
        elif "rosenbrock " in p.function_name:
            true_max_val = 0
        else:
            true_max_val = function_maximum_value[p.function_name] 
        regrets = true_max_val - vals
        cum_reg = np.divide(np.cumsum(regrets), np.arange(1,p.num_init_samples+budget+1))
        print(optimizer.max)
        print(cum_reg[-1])
        print(np.mean(optimizer.times) * 1e-9)
        
        return dict(
            regret=regrets,
            sample_locs=sample_locs,
            elapsed_time=elapsed_time,
            times = optimizer.times,#selected_len_scale = optimizer.length_scale,
            full_times= optimizer.full_times,
            sim_times=sim_times,
            memory = optimizer.memory,
            budget=budget,
            vals=vals,
            mus=None,
            variances=None,
            acquisition=None,
            total_time=optimizer.total_time,
            params=p
        )



if __name__=='__main__':
    args = get_args()
    # args.nengo = True

    # random.seed(1)
    seeds = [random.randint(1,100000) for _ in range(args.num_trials)]

    if args.nengo:
        data_path = os.path.join(os.getcwd(),
                                 args.data_dir, args.function_name, args.agent_type + '_nengo-' + args.backend)
    else:
        data_path = os.path.join(os.getcwd(),
                             args.data_dir,args.function_name,args.agent_type)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    for seed in seeds:
        params = {'function_name':args.function_name,
                  'agent_type':args.agent_type,
                  'num_samples':args.num_samples,
                  'data_format':'npz',
                  'data_dir':data_path,
                  'seed':seed, 
                  'verbose':False,
                  'ssp_dim':args.ssp_dim,
                  'len_scale':args.len_scale,
                  'beta_ucb':args.beta_ucb,
                  'gamma':args.gamma,
                  'nengo':args.nengo,
                  'backend':args.backend,
                  'n_scales':args.n_scales,
                  'n_rotates':args.n_rotates,
                  'decay': args.decay,
                  'num_neurons': args.num_neurons
                  }
        r = SamplingTrial().run(**params)
