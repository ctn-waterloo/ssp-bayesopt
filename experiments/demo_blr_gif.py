#!/usr/bin/env python3
"""BLR surrogate development GIF with smooth inter-frame interpolation.

UCB acquisition drives sample selection; both surrogate and acquisition
are animated as samples are gathered.

Usage:
    conda run -n bonengo python experiments/demo_blr_gif.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import minimize

import figure_utils as utils
import ssp_bayes_opt
from ssp_bayes_opt.blr import BayesianLinearRegression

# ── Config ────────────────────────────────────────────────────────────────────
N_INIT        = 1      # points used to initialise the agent (sets length scale)
N_STEPS       = 20     # UCB-guided samples added one at a time
N_INTERP      = 8      # extra interpolated frames between each pair of keyframes
SSP_DIM       = 75
LENGTH_SCALE  = 0.1
BLR_ALPHA     = 0.1    # prior precision; prior std ∝ ‖φ‖/√α ≈ 3.16
BLR_BETA      = 30.0   # observation precision (1/σ²); high → mean passes through data
BETA_UCB      = 100.
GAMMA_C       = 0.
AGENT_TYPE    = 'ssp-hex'
SEED          = 0
MS_KEYFRAME   = 500    # ms to hold each keyframe
MS_INTERP     = 60     # ms per interpolated frame
OUT_GIF       = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'blr_demo.gif')

# ── Target function ───────────────────────────────────────────────────────────
NOISE_SCALE = 0.8
NOISE_ZERO  = 0.8
pbounds = np.array([[0., 1.2]])


def target_clean(x):
    return (1.4 - 6.0 * x) * np.sin(18.0 * x)


def noise_std_func(x):
    """Spatially-varying noise std: 0 at NOISE_ZERO, max at domain bounds."""
    x = np.asarray(x).reshape(-1)
    left  = NOISE_SCALE * (NOISE_ZERO - x)  / (NOISE_ZERO - pbounds[0, 0])
    right = NOISE_SCALE * (x - NOISE_ZERO)  / (pbounds[0, 1] - NOISE_ZERO)
    return np.where(x <= NOISE_ZERO, left, right)


# ── Dense grid for plotting ───────────────────────────────────────────────────
xs         = np.linspace(pbounds[0, 0], pbounds[0, 1], 200).reshape(-1, 1)
ys_true    = target_clean(xs).reshape(-1)
noise_band = noise_std_func(xs.reshape(-1))
_pad  = 0.15 * (ys_true.max() - ys_true.min())
Y_LIM = (ys_true.min() - _pad, ys_true.max() + _pad)


def make_frame(label, sxs, sys_, mu, std, acq):
    fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
    ax.fill_between(
        xs.reshape(-1), ys_true - noise_band, ys_true + noise_band,
        alpha=0.35, edgecolor=None, facecolor=utils.grays[3],
        label=r'$\pm1\sigma$ noise',
    )
    ax.plot(xs, ys_true, '-', color=utils.grays[2], lw=1.5, label=r'$f(x)$')
    for k, a in zip([3, 2, 1], [0.12, 0.22, 0.38]):
        ax.fill_between(xs.reshape(-1), mu - k * std, mu + k * std,
                        alpha=a, edgecolor=None, facecolor=utils.blues[2])
    ax.plot(xs.reshape(-1), mu, color=utils.blues[1], lw=2, label=r'BLR mean $\mu$')
    acq_min, acq_max = acq.min(), acq.max()
    denom = acq_max - acq_min if acq_max > acq_min else 1.0
    acq_scaled = Y_LIM[0] + (acq - acq_min) / denom * (Y_LIM[1] - Y_LIM[0])
    ax.plot(xs.reshape(-1), acq_scaled, '--', color='darkorange', lw=1.5, label='UCB (scaled)')
    ax.plot(sxs, sys_, 'o', color=utils.reds[1], ms=5, zorder=5, label='Samples')
    ax.set_xlim(pbounds[0])
    ax.set_ylim(Y_LIM)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$f(x)$')
    ax.legend(fontsize=8, loc='upper right')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


if __name__ == '__main__':
    target = target_clean
    np.random.seed(SEED)

    optimizer = ssp_bayes_opt.BayesianOptimization(
        f=target, bounds=pbounds, verbose=False, sampling_seed=SEED
    )

    agt, init_xs, init_ys = optimizer.initialize_agent(
        N_INIT, AGENT_TYPE,
        domain_bounds=pbounds,
        ssp_dim=SSP_DIM,
        decoder_method='from-set',
        length_scale=LENGTH_SCALE,
        beta_ucb=BETA_UCB,
        gamma_c=GAMMA_C,
    )
    optimizer.agt = agt

    init_phis = agt.encode(init_xs)
    agt.blr   = BayesianLinearRegression(agt.ssp_dim, alpha=BLR_ALPHA, beta=BLR_BETA)
    agt.blr.update(init_phis, init_ys)

    noise_rng = np.random.default_rng(SEED + 2)

    def eval_surrogate(agt):
        phis    = agt.encode(xs)
        mu, var = agt.blr.predict(phis)
        return mu.reshape(-1), np.sqrt(np.maximum(var, 0.0))

    def eval_acq(agt):
        phis    = agt.encode(xs)
        mu, var = agt.blr.predict(phis)
        mu = mu.reshape(-1)
        acq = mu + agt.var_weight * np.sqrt(np.maximum(agt.gamma_t + var, 0.0))
        if agt.gamma_t > 0:
            acq -= np.sqrt(agt.gamma_t)
        return acq

    # ── Pre-compute all keyframe states ──────────────────────────────────────
    print('Pre-computing keyframe states...')

    sample_xs = list(init_xs[:, 0])
    sample_ys = list(init_ys[:, 0])

    states = []
    mu0, std0 = eval_surrogate(agt)
    acq0      = eval_acq(agt)
    states.append(dict(mu=mu0, std=std0, acq=acq0,
                       xs=list(sample_xs), ys=list(sample_ys),
                       label=f'Init ({N_INIT} sample)'))

    for step in range(N_STEPS):
        optim_func, jac_func = agt.acquisition_func()
        phi_init = np.ones(agt.ssp_dim) * (1.0 / agt.ssp_dim)
        soln  = minimize(optim_func, phi_init.flatten(), jac=jac_func, method='L-BFGS-B')
        x_t   = agt.decode(np.copy(np.atleast_2d(soln.x)))
        x_val = float(x_t.flatten()[0])

        x_t_2d = np.atleast_2d([[x_val]])
        noise  = noise_rng.normal(0.0, noise_std_func(np.array([x_val]))[0])
        y_t    = np.atleast_2d(target_clean(x_t_2d) + noise)

        mu_t, var_t, phi_t = agt.eval(x_t_2d)
        agt.update(x_t_2d, y_t, var_t, step_num=step + init_xs.shape[0])

        sample_xs.append(x_val)
        sample_ys.append(y_t.item())

        mu_new, std_new = eval_surrogate(agt)
        acq_new         = eval_acq(agt)
        states.append(dict(mu=mu_new, std=std_new, acq=acq_new,
                           xs=list(sample_xs), ys=list(sample_ys),
                           label=f'Step {step+1}/{N_STEPS}  ({len(sample_xs)} samples)'))
        print(f'  State {step+1}: x={x_val:.3f}  f(x)={y_t.item():.3f}')

    # ── Build frame list with interpolation ──────────────────────────────────
    print('\nRendering frames...')
    frames    = []
    durations = []

    s0 = states[0]
    frames.append(make_frame(s0['label'], s0['xs'], s0['ys'], s0['mu'], s0['std'], s0['acq']))
    durations.append(MS_KEYFRAME)
    print('  Keyframe 0 (init)')

    for i in range(len(states) - 1):
        s_prev, s_next = states[i], states[i + 1]
        for k in range(1, N_INTERP + 1):
            alpha = k / (N_INTERP + 1)
            mu_i  = (1 - alpha) * s_prev['mu']  + alpha * s_next['mu']
            std_i = (1 - alpha) * s_prev['std'] + alpha * s_next['std']
            acq_i = (1 - alpha) * s_prev['acq'] + alpha * s_next['acq']
            frames.append(make_frame(s_next['label'], s_next['xs'], s_next['ys'], mu_i, std_i, acq_i))
            durations.append(MS_INTERP)
        frames.append(make_frame(s_next['label'], s_next['xs'], s_next['ys'],
                                 s_next['mu'], s_next['std'], s_next['acq']))
        durations.append(MS_KEYFRAME)
        print(f'  Keyframe {i+1} (step {i+1})')

    # ── Save GIF ──────────────────────────────────────────────────────────────
    total = len(frames)
    print(f'\nSaving {total}-frame GIF → {OUT_GIF}')
    frames[0].save(
        OUT_GIF,
        save_all      = True,
        append_images = frames[1:],
        loop          = 0,
        duration      = durations,
        optimize      = False,
    )
    print('Done!')
