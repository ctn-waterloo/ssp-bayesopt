"""Plot memory usage over BO steps for GP-UCB and SSP-BO agents.

Reads .npz result files produced by run_agent.py (requires guppy memory tracking).
Generates a multi-panel figure with delta-memory curves per benchmark function.

Usage:
    python memory_figure.py --data-dir /path/to/memory-test/
"""
import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt
import figure_utils as utils
from argparse import ArgumentParser


def get_memory_data(filenames):
    data = []
    for f in filenames:
        mem = np.load(f)['memory'][:, 0]
        data.append(mem - mem[0])
    return np.vstack(data)


def get_stats(data):
    mean = np.mean(data, axis=0)
    ste = np.std(data, axis=0) / np.sqrt(data.shape[0])
    mean_sum = np.mean(np.sum(data, axis=1))
    ste_sum = np.std(np.sum(data, axis=1)) / np.sqrt(data.shape[0])
    return mean, ste, mean_sum, ste_sum


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default='/run/media/furlong/Data/ssp-bayesopt/memory-test/')
    args = parser.parse_args()

    folder = args.data_dir
    funcs = ["himmelblau", "goldstein-price", "branin-hoo"]
    fig_let = ['A', 'B', 'C']
    let_pos = [0, 0.32, 0.635]
    agts = ["gp-ucb-matern", "gp-ucb-sinc", "ssp-hex", "ssp-rand"]
    labels = ['GP-UCB-Matern', 'GP-UCB-Sinc', 'SSP-BO-Hex', 'SSP-BO-Rand']
    cols = [utils.reds[0], utils.greens[0], utils.blues[0], utils.oranges[0]]
    linestys = ['-.', ':', '-', '--']

    plt.figure(figsize=(6.5, 2.75))
    for f_idx, mem_func in enumerate(funcs):
        plt.subplot(1, len(funcs), 1 + f_idx)
        for j, agt in enumerate(agts):
            memory_data_agt = get_memory_data(glob.glob(f'{folder}/{mem_func}/{agt}/*npz'))
            agt_mu, agt_ste, _, _ = get_stats(memory_data_agt)
            steps = np.arange(1, agt_mu.shape[0] + 1)
            plt.plot(steps, agt_mu, label=labels[j], ls=linestys[j], color=cols[j], lw=2)
            plt.fill_between(steps, agt_mu - 1.96 * agt_ste,
                             agt_mu + 1.96 * agt_ste, alpha=0.4, color=cols[j])

            if f_idx == 0:
                plt.ylabel(r'$\Delta$ Memory (MB)')
                plt.legend()
            elif f_idx == 2:
                plt.gca().annotate(r'GP-MI',
                                   xy=(175, 0.3), xycoords='data',
                                   xytext=(20, 0.5), textcoords='data',
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3', shrinkA=0.05))
                plt.gca().annotate(r'SSP-BO',
                                   xy=(175, 0.05), xycoords='data',
                                   xytext=(20, 0.2), textcoords='data',
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3', shrinkA=0.05))
            plt.gcf().text(0.06 + let_pos[f_idx], 0.95,
                           f'\\textbf{{ {fig_let[f_idx]} \; \; }}', size=11, va="baseline", ha="left")
            plt.xlabel('Sample number ($n$)')
        plt.title(mem_func.title())
        plt.tight_layout()

    utils.save(plt.gcf(), 'gp_memory_usage.pdf')
