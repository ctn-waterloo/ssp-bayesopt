import numpy as np
import nengo
import nengo_loihi
from .vens import VirtualEnsemble
from .loihi_utils import *

#from lava.utils import weightutils as wu
#from lava.proc.dense.process import Dense
#from lava.proc.lif.process import LIF
#from lava.proc.graded.process import GradedReluVec
#
#from lava.proc import embedded_io as eio
#from lava.proc.io import sink, source
#
#from lava.magma.core.run_conditions import RunSteps, RunContinuous
#from lava.magma.core.run_configs import Loihi1SimCfg, Loihi2HwCfg


def make_network(bo_soln_init, m, sigma, beta_inv, gamma_t,
                 neurons_per_dim=50, seed=0, tau=0.05,
                 var_weight= 1.,
                 partition=None,
                 dt=0.001,
                 tau_probe=0.1,
                 pres_duration=0.1,
                 #neuron_type=nengo.RectifiedLinear(),
                 neuron_type=nengo.LIF(),
                 ):
    
    ssp_dim = bo_soln_init.size
    model = nengo.Network(seed=seed)
    model.config[nengo.Ensemble].neuron_type = neuron_type

    n_neurons = neurons_per_dim * ssp_dim

    with model:
        def stim_func(t,val=bo_soln_init.flatten()):
            if t < pres_duration:
                return val
            else:
                return np.zeros((ssp_dim,))

        def transform_func(x, sigma=sigma, mu=m, beta_inv=beta_inv, gamma_t=gamma_t, var_weight=var_weight):
            sqr = np.dot(x, np.dot(sigma, x))
            scale = np.sqrt(sqr + gamma_t + beta_inv)
            a_out = (mu + var_weight * sigma @ x / scale)
            return tau*a_out + x

        stim = nengo.Node(stim_func, label='stim')



        if partition is None:
            solution_neurons = nengo.Ensemble(
                    n_neurons, ssp_dim, 
                    label='solution_neurons',
                    #max_rates=40*np.ones((n_neurons,)),
                    )
            nengo.Connection(stim, solution_neurons, 
                    synapse=None, 
                    label='inp_conn',
                    )
            nengo.Connection(solution_neurons, solution_neurons, 
                    function=transform_func, 
                    synapse=tau,
                    label='fb_conn',
                    )  # , solver=nengo.solvers.LstsqDrop(weights=False,drop=0.25))
            solution_probe = nengo.Probe(solution_neurons, synapse=tau_probe,label='soln_probe')
            solution_node = nengo.Node(lambda t, x: x, 
                    size_in=ssp_dim,
                    label='soln_probe'
                    )
            nengo.Connection(solution_neurons,solution_node,
                    synapse=tau_probe, label='soln_conn')

        else:
            if n_neurons % partition != 0:
                raise ValueError("n_neurons (%s) must be divisible by partition (%s)" % (
                    n_neurons, partition))

            solution_neurons = VirtualEnsemble(n_ensembles=partition,
                                n_neurons_per_ensemble=n_neurons // partition,
                                dimensions=ssp_dim, label='solution_neurons')
            solution_neurons.add_input(stim, synapse=None)

            solution_neurons.add_input(solution_neurons.add_output(function=transform_func, dt=dt)[0],
                                       synapse=tau)

            # Copy the output so that the above is collapsed as a passthrough
            solution_node = nengo.Node(lambda t, x: x, 
                    size_in=ssp_dim,
                    )
            nengo.Connection(solution_neurons.add_output(dt=dt)[0],
                    solution_node,
                    synapse=tau_probe, label='soln_probe')
            #solution_probe = nengo.Probe(solution_neurons.add_output(dt=dt)[0], synapse=tau_probe, label='soln_probe')

    return model, solution_probe

if __name__ == '__main__':

    run_duration = 0.2 # sec
    pres_duration = 0.01 # sec
    

    ssp_dim = 32
    sigma = np.eye(ssp_dim)
    mu = np.zeros((ssp_dim,))
    step_size = 0.1
    beta_inv = 2
    gamma_t = 0
    init_guess = 2*(np.random.random(size=(ssp_dim,)) - 0.5 )
    # init_guess = np.ones((ssp_dim,))
    # init_guess = np.random.random(size=(ssp_dim,))

    nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

    use_relu = True
    model, solution_probe = make_network(
            bo_soln_init=init_guess,
            m=mu,
            sigma=sigma,
            beta_inv=beta_inv,
            gamma_t=gamma_t,
            neurons_per_dim=4,
            pres_duration=pres_duration,
            partition=None,
            tau_probe=1e-3,
            tau=1 if use_relu else 0.05,
            neuron_type=nengo.RectifiedLinear() if use_relu else nengo.LIF(),
        )
    sim = nengo.Simulator(model)
    model_params = extract_params(model, sim)

    num_steps = np.ceil(run_duration/sim.dt).astype(np.int32)
    num_pres = np.ceil(pres_duration/sim.dt).astype(np.int32)
    
    lava_model = LavaBoModel(inp_exp=20, w_exp=7)

    lava_model.convert(model_params,
            init_guess, 
            num_steps, 
            num_pres, 
            use_relu=use_relu)
    retval = lava_model.run().T
    print(retval.shape)

    # Run Nengo model 
    sim.run(run_duration)
    data = sim.data[solution_probe]

    print(np.mean(data[-10:,:], axis=0) - np.mean(retval[-10:,:], axis=0))
