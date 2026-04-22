"""Plot a 1-D BLR/SSP surrogate demo on a simple periodic function.

Runs a short BO loop on (1.4 - 3x)·sin(18x) and plots the SSP kernel
basis function, the BLR posterior mean/variance, and the sample points.

Usage:
    python plot_1d_blr.py
"""
import matplotlib.pyplot as plt
import figure_utils as utils
import time
import numpy as np

import ssp_bayes_opt


def objective(x):
    return (1.4 - 3.0 * x) * np.sin(18.0 * x)


if __name__ == '__main__':
    target = objective
    pbounds = np.array([[0, 1.2]])
    ls = 0.5
    beta_ucb = 10.
    gamma_c = 0.
    optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                                   bounds=pbounds,
                                                   verbose=True,
                                                   sampling_seed=0)

    n_iter = 20
    init_points = 1
    start = time.thread_time_ns()
    optimizer.maximize(init_points=init_points,
                       n_iter=n_iter - init_points,
                       num_restarts=1,
                       agent_type='ssp-hex',
                       ssp_dim=75, decoder_method='from-set',
                       length_scale=ls, beta_ucb=beta_ucb, gamma_c=gamma_c)
    elapsed_time = time.thread_time_ns() - start

    vals = np.zeros((len(optimizer.res),))
    sample_locs = []

    for i, res in enumerate(optimizer.res):
        vals[i] = res['target']
        sample_locs.append(res['params'])

    fig, axs = plt.subplots(1, 3, figsize=(7.1, 2))
    for i in range(3):
        axs[i].set_ylim([-3, 2.5])
        axs[i].set_xlabel('$x$')

    samples = [np.array(sample_locs).reshape(-1), np.array(vals)]
    xs = np.linspace(pbounds[0, 0], pbounds[0, 1], 100).reshape(-1, 1)
    phis = optimizer.agt.encode(xs)

    axs[0].plot(xs, target(xs), "-", color=utils.grays[2], label='$f(x)$')
    kern = phis @ optimizer.agt.encode(np.array(samples[0][0]).reshape(1, -1)).T
    axs[0].plot(xs.reshape(-1), samples[1][0] * kern.reshape(-1),
                color=utils.blues[2], label='$f(x_0)[\\phi(x_0) \\cdot \\phi(x)]$')
    axs[0].plot(samples[0][0], samples[1][0], "o", color=utils.reds[1],
                label='Sample point $x_0$')
    axs[0].legend()

    mu, var = optimizer.agt.blr.predict(phis)
    axs[1].plot(xs, target(xs), "-", color=utils.grays[2])
    axs[1].plot(xs.reshape(-1), mu.reshape(-1), color=utils.blues[1], label='BLR mean $\\mu$')
    axs[1].fill_between(xs.reshape(-1), mu.reshape(-1) - np.sqrt(var),
                        mu.reshape(-1) + np.sqrt(var), alpha=0.5,
                        edgecolor=None, facecolor=utils.blues[2], label='BLR sdev $\\sigma$')
    axs[1].plot(samples[0], samples[1], "o", color=utils.reds[1])
    axs[1].legend()

    plt.tight_layout()
    plt.show()
