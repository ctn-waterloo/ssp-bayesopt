#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# Adpated from https://github.com/AndrzejKucik/SNN4Space/blob/c759e48b02508b3596dbecfbd3e27306fddc6a49/energy_estimations.py
# Original code was authored by Andrzej S. Kucik, Gabriele Meoni,2022-01-28, copyright European Space Agency
# DEVICES is from the original code
# energy_estimation hsa been changed considerable for handling nengo network and using spike counts from a sim, but
# the overall logic behind the calculation comes from the above file


import numpy as np
import nengo
from run_agent import get_args, neuron_types, sim_types
import functions
import pandas as pd
import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)



# Device parameters per energy estimation
DEVICES = {
    # https://ieeexplore.ieee.org/abstract/document/7054508
    'cpu': dict(spiking=False, energy_per_synop=8.6e-9, energy_per_neuron=8.6e-9),
    'gpu': dict(spiking=False, energy_per_synop=0.3e-9, energy_per_neuron=0.3e-9),
    'arm': dict(spiking=False, energy_per_synop=0.9e-9, energy_per_neuron=0.9e-9),
    'myriad2': dict(spiking=False, energy_per_synop=1.9918386085474344e-10,
                    energy_per_neuron=1.9918386085474344e-10),

    # Value estimated by considering the energy for a MAC operation. Such energy (E_per_mac) is obtained through a
    # maximum-likelihood estimation: E_inf = E_per_mac * N_ops by, E_inf and N_ops values come from our previous
    # work: https://ieeexplore.ieee.org/abstract/document/8644728

    # https://www.researchgate.net/publication/322548911_Loihi_A_Neuromorphic_Manycore_Processor_with_On-Chip_Learning
    # 'loihi': dict(spiking=True, energy_per_synop=(23.6 + 3.5) * 1e-12, energy_per_neuron=81e-12), # why 3.5? that's the time per spike???
    'loihi1': dict(spiking=True, energy_per_synop=23.6 * 1e-12, energy_per_neuron=81e-12),
    # why 3.5? that's the time per spike???

    # https://arxiv.org/abs/1903.08941
    'spinnaker': dict(spiking=True, energy_per_synop=13.3e-9, energy_per_neuron=26e-9),
    'spinnaker2': dict(spiking=True, energy_per_synop=450e-12, energy_per_neuron=2.19e-9),
}
#The energy_per_neuron is Energy per neuron update (active / inactive)=81 pJ / 52 pJ. The energy_per_synop is Energy per synaptic spike op (min) = 23.6 pJ
# Note "energy per synaptic operation", E_sop is reported differently in some sources,
# eg https://www.mdpi.com/2079-9292/13/16/3203 (specific conditions)
# Loihi 1 E_sop > 23.6 pJ, SpiNNaker E_sop > 26.6 nJ
# OR https://pmc.ncbi.nlm.nih.gov/articles/PMC11060491/
# Loihi 1 E_sop = 23.6 pJ. SpiNNaker E_sop = 11.3 nJ

def energy_estimation(model,
                      sim,
                      spiking_model: bool = True,
                      device_list: list = None,
                      verbose: bool = False):
    """
    Estimate the energy spent for synaptic operations and neurons update required for an inference for an Artificial or
    Spiking layer on a target hardware list. Energy is estimated by multiplying the number of synaptic operations and
    neuron updates times the energy values for a single synaptic operation and neuron update. The number of synaptic
    operations is obtained using the sim object (a test run of the model).  Energy values for
    Myriad 2 devices is obtained by mean square error interpolation of the values provided in: Benelli, Gionata,
    Gabriele Meoni, and Luca Fanucci. "A low power keyword spotting algorithm for memory constrained embedded systems."
    2018 IFIP/IEEE International Conference on Very Large Scale Integration (VLSI-SoC). IEEE, 2018.

    Parameters
    ----------
    model : nengo.Network()
        Nengo model object
    sim:
        Simulation to extract number of spikes for a spiking model. It can be None for Artificial
        Neural Networks.
    spiking_model: bool (default=True)
        Flag to indicate if the model is a spiking model or not.
    device_list: list (default=['loihi'])
        List containing the name of the target hardware devices. Supported `cpu` (Intel i7-4960X), `gpu` (Nvidia GTX
        Titan Black), `arm` (ARM Cortex-A), `loihi`, `spinnaker`, `spinnaker2`, `myriad2`.

    verbose: bool (default=False)
        If `True`, energy contributions are shown for every single layer and additional log info is provided.

    Returns
    -------
    synop_energy_dict : dict
        Dictionary including the energy contribution for synaptic operations for each target device.
    neuron_energy_dict : dict
         Dictionary including the energy contribution for neuron updates for each target device.
    total_energy_dict : dict
        Dictionary including the total energy per device.
    """

    if device_list is None:
        device_list = list(DEVICES.keys())

    dt = sim.dt
    n_timesteps = len(sim.trange())

    all_ensembles = model.all_ensembles
    all_connections = model.all_connections
    all_probes = model.all_probes

    # Energy estimation
    # - Initialize it with the total number of neurons
    neuron_active_energy = 0
    neuron_inactive_energy = 0
    neuron_energy = model.n_neurons

    probes_dict = {}
    # - Spiking model
    if spiking_model:
        # -- In spiking model the neuron energy must be multiplied by the number of timesteps
        neuron_energy *= n_timesteps
        ## Not currently using this, though it should be more accurate because I'm not 100% sore what active/inactive #s meant and don't have those for the other backends
        for ens in all_ensembles:
            probe = all_probes[np.where([ens == p.target.ensemble for p in all_probes if
                                         (p.target.__module__ == 'nengo.ensemble') and (p.attr == 'output')])[0][0]]
            probes_dict[ens] = probe
            spikes = sim.data[probe]  # shape: (timesteps, n_neurons)
            total_spikes_per_neuron = np.sum(spikes, axis=0)

            # Energy per neuron based on whether it spiked or not
            neuron_active_energy += np.count_nonzero(total_spikes_per_neuron)
            neuron_inactive_energy += (len(total_spikes_per_neuron) - np.count_nonzero(total_spikes_per_neuron))

        synop_energy = 0
        for conn in all_connections:
            if isinstance(conn.pre_obj, nengo.Ensemble):
                # Get presynaptic spike activity
                spikes = sim.data[probes_dict[conn.pre_obj]]  # shape: (timesteps, pre_neurons)
                total_spikes = np.sum(spikes)  # scalar

                if isinstance(conn.post_obj, nengo.Ensemble):
                    n_post = conn.post_obj.n_neurons
                elif isinstance(conn.post_obj, nengo.Node):
                    n_post = conn.post_obj.size_in
                else:
                    continue  # Unknown type

                synop_energy += total_spikes * n_post

            elif isinstance(conn.pre_obj, nengo.Node):
                # If a Node is the source, assume continuous signal sampled at each timestep
                n_pre = conn.pre_obj.size_out
                if isinstance(conn.post_obj, nengo.Ensemble):
                    n_post = conn.post_obj.n_neurons
                else:
                    continue

                # Assume 1 activation per timestep per dimension
                total_activations = n_timesteps * n_pre
                synop_energy += total_activations * n_post

    # - Non-spiking model: assuming an algebraic version doing math in vector space, not neural at all
    # also assuming/not counting any functions inside nodes
    else:
        synop_energy = 0
        for conn in all_connections:
            n_pre = 0
            n_post = 0
            if isinstance(conn.pre_obj, nengo.Ensemble):
                n_pre = conn.pre_obj.n_neurons
            elif isinstance(conn.pre_obj, nengo.Node):
                n_pre = conn.pre_obj.size_out
            if isinstance(conn.post_obj, nengo.Ensemble):
                n_post = conn.post_obj.n_neurons
            elif isinstance(conn.post_obj, nengo.Node):
                n_post = conn.post_obj.size_in

            synop_energy += n_timesteps * n_pre * n_post


    # - Placeholders for energy readings
    synop_energy_dict = {}
    neuron_energy_dict = {}
    total_energy_dict = {}

    # - Loop over the devices
    for device in device_list:
        energy_dict = DEVICES[device]

        # -- Multiply the energy units by the energy consumption specific for a device
        synop_energy_dict[device] = synop_energy * energy_dict['energy_per_synop']
        neuron_energy_dict[device] = neuron_energy * energy_dict['energy_per_neuron']
        total_energy_dict[device] = synop_energy_dict[device] + neuron_energy_dict[device]

        # -- Print out the results if necessary
        if verbose:
            print(f'Estimated energy on {device}')
            print('\tSynop energy: ', synop_energy_dict[device], 'J')
            print('\tNeuron energy: ', neuron_energy_dict[device], 'J')
            print('\tTotal energy:', total_energy_dict[device], 'J\n\n')

    return synop_energy_dict, neuron_energy_dict, total_energy_dict

# For SSP-BO
if __name__ == '__main__':
    args = get_args()
    args.nengo = True
    args.backend = 'cpu' # don't really need to actually run on hardware to get info for energy estimates
    args.num_neurons = 8 # 7
    sim_time = 2.5
    target, pbounds, _ = functions.factory(args.function_name)
    budget = 50 # don't need to fully run it, just want reasonable mu and Sigma
    init_points = 10
    tau = 0.05
    neuron_type = neuron_types['lif']
    sim_type, sim_args = sim_types[args.backend]

    optimizer = ssp_bayes_opt.NengoBayesianOptimization(f=target,
                                                        bounds=pbounds,
                                                        verbose=False,
                                                        sampling_seed=np.random.randint(1000))
    optimizer.maximize(init_points=init_points,
                       n_iter=budget,
                       num_restarts=1,
                       agent_type=args.agent_type,
                       ssp_dim=args.ssp_dim,
                       n_scales=args.n_scales,
                       n_rotates=args.n_rotates,
                       length_scale=args.len_scale,
                       decoder_method='direct-optim',
                       gamma_c=args.gamma,
                       beta_ucb=args.beta_ucb,
                       var_decay=0.,
                       neurons_per_dim=args.num_neurons,
                       neuron_type=neuron_type,
                       sim_type=sim_type, sim_args=sim_args,
                       sim_time=sim_time, tau=tau,
                       save_memory=False,
                       partitions=None
                       )
    phi_init = optimizer.agt.initial_guess()
    solver_net, soln_probe, stim_node = ssp_bayes_opt.make_network(
        bo_soln_init=phi_init.flatten(),
        m=optimizer.agt.blr.m.flatten(),
        sigma=optimizer.agt.blr.S,
        beta_inv=1 / optimizer.agt.blr.beta,
        gamma_t=optimizer.agt.gamma_c,
        var_weight=optimizer.agt.var_weight,
        neurons_per_dim=args.num_neurons,
        tau=tau,
        seed=optimizer.seed,
        neuron_type=neuron_type,
        rate=1.
    )
    with solver_net:
        for ens in solver_net.all_ensembles:
            nengo.Probe(ens.neurons, synapse=None)

    sim = sim_type(solver_net, **sim_args)
    with sim:
        sim.run(sim_time)

    # --- Get the energy usage
    snn_synop_energy_dict, snn_neuron_energy_dict, snn_total_energy_dict = energy_estimation(solver_net,
                                                                  sim, spiking_model = True,
                                                                  device_list=['loihi1', 'spinnaker'],
                                                                  verbose=True)
    nn_synop_energy_dict, nn_neuron_energy_dict, nn_total_energy_dict = energy_estimation(solver_net,
                                                                                             sim, spiking_model=False,
                                                                                             device_list=['cpu',
                                                                                                          'gpu'],
                                                                                             verbose=True)
    nn_synop_energy_dict.update(snn_synop_energy_dict)
    nn_neuron_energy_dict.update(snn_neuron_energy_dict)
    nn_total_energy_dict.update(snn_total_energy_dict)
    data_dict = {'Synop energy (J)': nn_synop_energy_dict,
                'Neuron energy (J)': nn_neuron_energy_dict,
                'Total energy (J)': nn_total_energy_dict}
    df = pd.concat({k: pd.DataFrame.from_dict(v, 'index') for k, v in data_dict.items()}, axis=0)

