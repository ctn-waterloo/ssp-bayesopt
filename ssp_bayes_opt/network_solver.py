import numpy as np
import nengo
import nengo_loihi
from .vens import VirtualEnsemble

def make_network(bo_soln_init, m, sigma, beta_inv, gamma_t,
                 neurons_per_dim=50, seed=0, tau=0.05,
                 var_weight= 1.,
                 partition=None,
                 dt=0.001,
                 tau_probe=0.1,
                 neuron_type=nengo.LIF()):
    
    ssp_dim = bo_soln_init.size
    model = nengo.Network(seed=seed)
    model.config[nengo.Ensemble].neuron_type = neuron_type

    n_neurons = neurons_per_dim * ssp_dim

    with model:
        def stim_func(t,val=bo_soln_init.flatten()):
            if t < 0.1:
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
            solution_neurons = nengo.Ensemble(n_neurons, ssp_dim, label='solution_neurons')
            nengo.Connection(stim, solution_neurons, synapse=None)
            nengo.Connection(solution_neurons, solution_neurons, function=transform_func,
                             synapse=tau)  # , solver=nengo.solvers.LstsqDrop(weights=False,drop=0.25))
            solution_probe = nengo.Probe(solution_neurons, synapse=tau_probe)

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
            solution_probe = nengo.Probe(solution_neurons.add_output(dt=dt)[0], synapse=tau_probe)

    return model, solution_probe

if __name__ == '__main__':

    ssp_dim = 151

    sigma = np.eye(ssp_dim)
    mu = np.zeros((ssp_dim,))
    step_size = 0.1
    beta_inv = 2
    gamma_t = 0
    init_guess = 2*(np.random.random(size=(ssp_dim,)) - 0.5 )
    # init_guess = np.ones((ssp_dim,))
    # init_guess = np.random.random(size=(ssp_dim,))

    nengo.rc['progress']['progress_bar'] = 'nengo.utils.progress.TerminalProgressBar'

    model, solution_probe = make_network(
            bo_soln_init=init_guess,
            m=mu,
            sigma=sigma,
            beta_inv=beta_inv,
            gamma_t=gamma_t,
            neurons_per_dim=4,
            partition=None,
        )
    sim = nengo_loihi.Simulator(model)
    with sim:
        sim.run(2.5)
    raw_data = sim.data[solution_probe]
    print("Done no partition")

    model, solution_probe = make_network(
        bo_soln_init=init_guess,
        m=mu,
        sigma=sigma,
        beta_inv=beta_inv,
        gamma_t=gamma_t,
        neurons_per_dim=8,
        partition=2,
    )
    sim = nengo_loihi.Simulator(model)
    with sim:
        sim.run(2.5)
    raw_data_v2 = sim.data[solution_probe]
    print("Done with partition")

    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(raw_data, alpha=0.8, color='blue')
    plt.plot(raw_data_v2, '--', color='red')
    plt.show()
