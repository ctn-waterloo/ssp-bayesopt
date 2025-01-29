import numpy as np
import nengo
import nengo_loihi


def make_network(bo_soln_init, m, sigma, beta_inv, gamma_t,
                 neurons_per_dim=50, seed=0, tau = 0.05,
                 neuron_type=nengo.LIF()):
    
    ssp_dim = bo_soln_init.size
    model = nengo.Network(seed=seed)
    model.config[nengo.Ensemble].neuron_type = neuron_type

    with model:
        def stim_func(t,val=bo_soln_init.flatten()):
            if t < 0.1:
                return val
            else:
                return np.zeros((ssp_dim,))

        def transform_func(x, sigma=sigma, mu=m, beta_inv=beta_inv, gamma_t=gamma_t):
            sqr = np.dot(x, np.dot(sigma, x))
            scale = np.sqrt(sqr + gamma_t + beta_inv)
            a_out = (mu + sigma @ x / scale)
            return tau*a_out + x

        stim = nengo.Node(stim_func, label='stim')
        solution_neurons = nengo.Ensemble(neurons_per_dim * ssp_dim, ssp_dim, label='solution_neurons')
        nengo.Connection(stim, solution_neurons, synapse=None)
        nengo.Connection(solution_neurons, solution_neurons, function=transform_func,
                         synapse=tau)#, solver=nengo.solvers.LstsqDrop(weights=False,drop=0.25))
        solution_probe = nengo.Probe(solution_neurons)
    return model, solution_probe

if __name__ == '__main__':

    ssp_dim = 16

    sigma = np.eye(ssp_dim)
    mu = np.zeros((ssp_dim,))
    step_size = 0.1
    neurons_per_dim=100
    beta_inv = 2
    gamma_t = 0
    init_guess = 2*(np.random.random(size=(ssp_dim,)) - 0.5 )
    # init_guess = np.ones((ssp_dim,))
    # init_guess = np.random.random(size=(ssp_dim,))

    model, solution_probe = make_solver_network(
            bo_soln_init=init_guess, 
            m=mu, 
            sigma=sigma, 
            beta_inv=beta_inv,
            gamma_t=gamma_t,
        )
    sim = nengo_loihi.Simulator(model)
    with sim:
        sim.run(5)

    import matplotlib.pyplot as plt


    raw_data = sim.data[solution_probe]
    n = np.arange(1,raw_data.shape[0]+1)
    smoothed_data = (np.cumsum(raw_data, axis=0).T / n).T

    plt.plot(smoothed_data)
    plt.show()
