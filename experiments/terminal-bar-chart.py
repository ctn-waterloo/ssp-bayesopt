"""Violin plot of terminal average regret with Wilcoxon significance bars.

Reads .npz result files saved by run_agent.py and produces per-function
violin plots comparing all agents. Data folder defaults to
experiments/data/d151_v2; override with --data-dir.

Usage:
    python terminal-bar-chart.py --data-dir experiments/data/d151_v2
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import figure_utils as utils
import numpy.matlib as matlib
from matplotlib.markers import TICKDOWN
from scipy.stats import sem
import pandas as pd
import os
import zipfile
from scipy.stats import wilcoxon
from argparse import ArgumentParser


def read(path):
    data = []
    if os.path.exists(path):
        for fn in os.listdir(path):
            fn = os.path.join(path, fn)
            if fn.endswith('.zip'):
                z = zipfile.ZipFile(fn)
                for name in z.namelist():
                    try:
                        if name.endswith('.txt'):
                            data.append(_read_text(z.open(name)))
                        elif name.endswith('.npz'):
                            data.append(_read_npz(z.open(name)))
                    except Exception:
                        print('Error reading file "%s" in "%s"' % (name, fn))
                z.close()
            else:
                try:
                    if fn.endswith('.txt'):
                        data.append(_read_text(fn))
                    elif fn.endswith('.npz'):
                        data.append(_read_npz(fn))
                except Exception:
                    print('Error reading file "%s"' % fn)
    return data


def _read_text(fn):
    if not hasattr(fn, 'read'):
        with open(fn) as f:
            text = f.read()
    else:
        text = fn.read()
    d = {}
    try:
        import numpy
        d['array'] = numpy.array
        d['nan'] = numpy.nan
    except ImportError:
        numpy = None
    exec(text, d)
    del d['__builtins__']
    if numpy is not None:
        if d.get('array') is numpy.array:
            del d['array']
        if d.get('nan') is numpy.nan:
            del d['nan']
    return d


def _read_npz(fn):
    d = {}
    f = np.load(fn, allow_pickle=True)
    for k in f.files:
        if k != 'ssp_space':
            d[k] = f[k]
            if d[k].shape == ():
                d[k] = d[k].item()
    return d


def significance_bar(ax, start, end, height, displaystring,
                     linewidth=1.1, markersize=6, boxpad=0.2, fontsize=11, color='k'):
    ax.plot([start, end], [height] * 2, '-', color=color,
            lw=linewidth, marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    ax.text(0.5 * (start + end), height, displaystring,
            ha='center', va='center',
            bbox=dict(facecolor='1.', edgecolor='none', boxstyle='Square,pad=' + str(boxpad)),
            size=fontsize)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default=os.path.join(os.getcwd(), 'experiments', 'data', 'd151_v2'))
    args = parser.parse_args()

    funcs = ["himmelblau", "goldstein-price", "branin-hoo"]
    agts = ["ssp-hex", "ssp-rand", "gp-sinc", "gp-matern", 'rff']
    labels = {"ssp-hex": 'SSP-BO-Hex', "ssp-rand": 'SSP-BO-Rand',
              "gp-sinc": 'GP (sinc)', "gp-matern": 'GP (Mat\u00e9rn)',
              "rff": 'RFF-BO',
              "ssp-hex_nengo-loihi-sim": "SSP-BO-Hex (Loihi 1$^*$)",
              "ssp-hex_nengo-spinnaker": "SSP-BO-Hex (Spinnaker)",
              "ssp-hex_nengo-loihi": "SSP-BO-Hex (Loihi 2)"}
    cols = {"ssp-hex": utils.blues[0], "ssp-rand": utils.oranges[0],
            "gp-sinc": utils.greens[0], "gp-matern": utils.reds[0],
            "rff": utils.purples[0],
            "ssp-hex_nengo-loihi-sim": utils.blues[1],
            "ssp-hex_nengo-spinnaker": utils.blues[2],
            "ssp-hex_nengo-loihi": utils.blues[1]}

    def do_plot(ax, regrets):
        ordered_regrets = {k: v for k, v in sorted(regrets.items(), key=lambda item: np.mean(item[1]))}
        ordered_regret_keys = list(ordered_regrets.keys())
        ordered_regrets_arr = np.array(list(ordered_regrets.values())).T
        plots = ax.violinplot(ordered_regrets_arr)
        for pc, k in zip(plots['bodies'], ordered_regret_keys):
            pc.set_facecolor(cols[k])
        plots['cbars'].set_colors([cols[k] for k in ordered_regret_keys])
        plots['cmaxes'].set_colors([cols[k] for k in ordered_regret_keys])
        plots['cmins'].set_colors([cols[k] for k in ordered_regret_keys])
        return np.max(ordered_regrets_arr, axis=0), [labels[k] for k in ordered_regret_keys]

    letters = ["\\textbf{a}", "\\textbf{b}", "\\textbf{c}", "\\textbf{d}"]
    letters = [l + " $\\quad$ " for l in letters]
    fontsize = 9
    starsym = '$\\ast$'
    fig, axs = plt.subplots(1, 3, figsize=(utils.doublecolwidth, 3.))
    axs = axs.reshape(-1)
    shift = [0.2, 0.05, 5]
    for i, func in enumerate(funcs):
        all_regrets = {}
        axs[i].set_title(letters[i] + func.title(), fontsize=fontsize)
        for j, agt in enumerate(agts):
            data = pd.DataFrame(read(os.path.join(args.data_dir, func, agt)))
            regrets = np.array([data['regret'][k] for k in range(len(data['regret']))])
            budget = regrets.shape[1]
            num_trials = regrets.shape[0]
            regrets = np.divide(np.cumsum(regrets, axis=1),
                                matlib.repmat(range(1, budget + 1), num_trials, 1))[:, -1]
            all_regrets[agt] = regrets
        max_vals, ordered_regrets = do_plot(axs[i], all_regrets)
        axs[i].set_xticks(range(1, len(ordered_regrets) + 1), ordered_regrets, rotation=45, fontsize=fontsize)
        drawn_maxs = []
        for j in range(1, len(agts)):
            mval = np.max([max_vals[ordered_regrets.index(labels['ssp-hex'])], max_vals[j]] + drawn_maxs)
            pvalue = wilcoxon(all_regrets['ssp-hex'], all_regrets[agts[j]], alternative='less').pvalue
            starstr = (starsym * 3 if pvalue < 0.0001 else starsym * 2 if pvalue < 0.001
                       else starsym if pvalue < 0.01 else 'n.s' if pvalue >= 0.05 else None)
            if starstr is None:
                continue
            drawn_maxs.append(mval + shift[i])
            significance_bar(axs[i], ordered_regrets.index(labels['ssp-hex']) + 1, j + 1,
                             mval + shift[i], starstr,
                             fontsize=fontsize - 2 if pvalue >= 0.01 else 11)

    axs[0].set_ylabel('$\\leftarrow$ Terminal Regret', fontsize=fontsize)
    fig.tight_layout()
    utils.save(fig, 'test-func-terminal.pdf')
