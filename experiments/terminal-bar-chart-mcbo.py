"""Violin plot of terminal objective value for MCBO mixed-variable tasks.

Compares SSP-BO against MCBO baselines (Casmopolitan, BOiLS, COMBO, BODi, BOCS, BOSS)
on Pest Control and RNA Inverse Folding tasks. Requires the mcbo package
(https://github.com/huawei-noah/HEBO/tree/master/MCBO) and a pre-generated
all_res_comb.csv results file pointed to by RESULTS_DIR.

Usage:
    python terminal-bar-chart-mcbo.py
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

from mcbo import RESULTS_DIR
from mcbo.optimizers.bo_builder import BO_ALGOS
from mcbo.utils.experiment_utils import get_task_from_id, get_opt


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
    data_save_name = "data_mcbo"
    ALL_RES_COMB_PATH = os.path.join(RESULTS_DIR, "data", data_save_name, "all_res_comb.csv")
    task_ids = ['pest', 'rna_inverse_fold']
    task_dict = {
        'pest': 'Pest Control',
        'rna_inverse_fold': 'RNA Inverse Folding'
    }
    model_ids = ["gp_o", "gp_to", "gp_hed", "gp_ssk", "gp_diff", "lr_sparse_hs"]
    acq_opt_ids = ["is", "sa", "ga", "ls"]
    tr_ids = [None, "basic"]
    non_bo_short_ids = ["ga", "rs", "sa", "hc", "mab"]
    seeds = np.arange(42, 52)

    COMB_BO_ALGO = ["Casmopolitan", "BOiLS", "COMBO", "BODi", "BOCS", "BOSS"]
    BO_ALGO_FULL_NAME_TO_ALIAS = {}
    BO_ALGO_ALIAS_TO_FULL_NAME = {}

    task = get_task_from_id(task_id=task_ids[0])
    search_space = task.get_search_space()

    for alias in COMB_BO_ALGO:
        bo_algo_builder = BO_ALGOS[alias]
        opt = bo_algo_builder.build_bo(
            search_space=search_space,
            n_init=20,
            input_constraints=task.input_constraints,
        )
        BO_ALGO_FULL_NAME_TO_ALIAS[opt.name] = alias
        BO_ALGO_ALIAS_TO_FULL_NAME[alias] = opt.name

    NON_BO_ALGO_NAMES = []
    for non_bo_short_id in non_bo_short_ids:
        opt = get_opt(task=task, short_opt_id=non_bo_short_id)
        NON_BO_ALGO_NAMES.append(opt.name)

    all_res = pd.read_csv(ALL_RES_COMB_PATH, index_col=0)

    def data_key_to_label(k: str):
        if k in BO_ALGO_FULL_NAME_TO_ALIAS:
            k = BO_ALGO_FULL_NAME_TO_ALIAS[k]
        if k == "IS":
            return "HC"
        k = k.replace("LR (sparse_horseshoe)", "LSH")
        k = k.replace("HED-mat52", "HED")
        k = k.replace("Diffusion", "Diff.")
        k = k.replace("basic", "w/ TR")
        k = k.replace("no-tr", "w/o TR")
        k = k.replace("Tr-based GA acq optim", "GA w/ TR")
        k = k.replace("IS acq optim", "HC acq optim")
        k = k.replace("GP (SSK) - GA w/ TR", "GP (SSK)")
        k = k.replace("GP (TO) - GA w/ TR", "GP (TO)")
        return k

    num_evals = np.arange(20, 201)
    group_col = "Optimizer"
    final_num_eval = 199
    sub_regret_df = all_res.copy()
    sub_regret_df["rank"] = sub_regret_df.groupby(["Task", "Eval Num", "Seed"])[["f(x*)"]].rank()
    algo_name_to_avg_rank = (sub_regret_df[sub_regret_df["Eval Num"] == final_num_eval]
                             .groupby(["Optimizer"])["rank"].mean().to_dict())

    best_unbranded_opt_names = sorted(
        [k for k in algo_name_to_avg_rank if (k not in BO_ALGO_FULL_NAME_TO_ALIAS and k not in NON_BO_ALGO_NAMES)],
        key=lambda k: algo_name_to_avg_rank[k])
    selected_algo_names = (best_unbranded_opt_names[:2] +
                           [k for k in BO_ALGO_FULL_NAME_TO_ALIAS.keys() if k in algo_name_to_avg_rank])

    folder = os.path.join(os.getcwd(), "data_copy")
    agts = ["ssp-mcbo"]
    labels = dict(zip(selected_algo_names, [data_key_to_label(s) for s in selected_algo_names]))
    labels['ssp-mcbo'] = 'SSP-BO'

    COLORS = utils.cols + [utils.yellows[0], utils.grays[0], utils.reds[0], utils.greens[0], utils.grays[1]]
    cols = {data_key: COLORS[i] for i, data_key in enumerate(selected_algo_names)}
    cols['ssp-mcbo'] = utils.blues[0]

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

    inv_label = dict(zip([labels[k] for k in selected_algo_names], selected_algo_names))
    inv_label[labels['ssp-mcbo']] = 'ssp-mcbo'
    n_seeds = 10
    letters = ["\\textbf{d}", "\\textbf{e}", "\\textbf{f}", "\\textbf{g}"]
    letters = [l + " $\\quad$ " for l in letters]
    fontsize = 9
    starsym = '$\\ast$'
    shifts = [0.3, 1]
    fig, axs = plt.subplots(1, len(task_ids), figsize=(utils.doublecolwidth, 3.))
    axs = axs.reshape(-1)
    for i, func in enumerate(task_ids):
        all_regrets = {}
        axs[i].set_title(letters[i] + task_dict[func], fontsize=fontsize)
        for j, agt in enumerate(agts):
            data = pd.DataFrame(read(os.path.join(folder, func, agt)))
            regrets = np.min(-np.array([data['vals'][k] for k in range(len(data['vals']))]), axis=-1)[:n_seeds]
            all_regrets[agt] = regrets.flatten()
        for agt in selected_algo_names:
            regrets = all_res["f(x*)"][(all_res["Optimizer"] == agt) & (all_res["Task"] == task_dict[func])
                                       & (all_res["Eval Num"] == final_num_eval)]
            all_regrets[agt] = np.array(regrets).flatten()
        max_vals, ordered_regrets = do_plot(axs[i], all_regrets)
        axs[i].set_xticks(range(1, len(ordered_regrets) + 1), ordered_regrets, rotation=45, fontsize=fontsize)
        drawn_maxs = []
        ssp_ind = ordered_regrets.index(labels['ssp-mcbo'])
        if ssp_ind < len(ordered_regrets) - 1:
            for j in range(ssp_ind + 1, len(ordered_regrets)):
                agt = ordered_regrets[j]
                mval = np.max([max_vals[ssp_ind], max_vals[j]] + drawn_maxs)
                pvalue = wilcoxon(all_regrets['ssp-mcbo'], all_regrets[inv_label[agt]], alternative='less').pvalue
                starstr = (starsym * 3 if pvalue < 0.0001 else starsym * 2 if pvalue < 0.001
                           else starsym if pvalue < 0.01 else 'n.s' if pvalue >= 0.05 else None)
                if starstr is None:
                    continue
                drawn_maxs.append(mval + shifts[i])
                significance_bar(axs[i], ssp_ind + 1, j + 1, mval + shifts[i], starstr,
                                 fontsize=fontsize - 2 if pvalue >= 0.01 else 11)
        drawn_maxs = []
        if ssp_ind > 0:
            for j, agt in enumerate(ordered_regrets[:ssp_ind]):
                mval = np.max([max_vals[j], max_vals[ssp_ind]] + drawn_maxs)
                pvalue = wilcoxon(all_regrets[inv_label[agt]], all_regrets['ssp-mcbo'], alternative='less').pvalue
                starstr = (starsym * 3 if pvalue < 0.0001 else starsym * 2 if pvalue < 0.001
                           else starsym if pvalue < 0.01 else 'n.s' if pvalue >= 0.05 else None)
                if starstr is None:
                    continue
                drawn_maxs.append(mval + shifts[i])
                significance_bar(axs[i], j + 1, ssp_ind + 1, mval + shifts[i], starstr,
                                 fontsize=fontsize - 2 if pvalue >= 0.01 else 11)

    axs[0].set_ylabel('$\\leftarrow$ Best Objective Value', fontsize=fontsize)
    fig.tight_layout()
    utils.save(fig, 'mcbo-terminal.pdf')
