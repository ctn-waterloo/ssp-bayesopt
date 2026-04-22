"""Plot average regret and sample-selection time for standard test functions.

Reads .npz result files saved by run_agent.py. Data folder defaults to
experiments/data/d151_v2; override with --data-dir.

Usage:
    python plot_test_funcs.py --data-dir experiments/data/d151_v2
"""
import numpy as np
import matplotlib.pyplot as plt
import figure_utils as utils
import numpy.matlib as matlib
from matplotlib.gridspec import GridSpec
import os
import zipfile
import pandas as pd
import pickle
import re
from argparse import ArgumentParser


function_maximum_value = {
    'himmelblau': 0.07076226300682818,
    'branin-hoo': -0.397887,
    'goldstein-price': -3/1e5,
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133,
    'styblinski-tang1': 39.16599,
    'styblinski-tang2': 39.16599 * 2,
    'styblinski-tang3': 39.16599 * 3,
}


def get_mean_and_ci(raw_data, n=3000, p=0.95):
    """Bootstrap 95% CI over rows of raw_data."""
    sample = []
    upper_bound = []
    lower_bound = []
    raw_data = np.array(raw_data)
    data_pts = raw_data.shape[1]
    for i in range(data_pts):
        col = raw_data[:, i]
        index = int(n * (1 - p) / 2)
        samples = np.random.choice(col, size=(n, len(col)))
        r = sorted([np.mean(s) for s in samples])
        ci = r[index], r[-index]
        sample.append(np.mean(col))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])
    return {"mean": sample, "lower_bound": lower_bound, "upper_bound": upper_bound}


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


def read_nengo(folder_path, function_name="", agent_type=""):
    filename_pattern = re.compile(r'sspd97_N8_tau0\.05_seed(\d+)\.pkl')
    data = []
    for filename in os.listdir(folder_path):
        match = filename_pattern.match(filename)
        if match:
            seed = int(match.group(1))
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                regrets = function_maximum_value[function_name] - loaded_data['f_vals']
                times = loaded_data['times']
            data.append({'seed': seed, 'function_name': function_name,
                         'agent_type': agent_type, 'regret': regrets, 'times': times})
    return data


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-dir', dest='data_dir', type=str,
                        default=os.path.join(os.getcwd(), 'experiments', 'data', 'd151_v2'))
    args = parser.parse_args()

    funcs = ["himmelblau", "goldstein-price", "branin-hoo"]
    folders = [args.data_dir]
    agts = ["ssp-hex", "ssp-rand", "gp-sinc", "gp-matern", "rff"]
    labels = {"ssp-hex": 'SSP-BO-Hex', "ssp-rand": 'SSP-BO-Rand',
              "gp-sinc": 'GP (sinc)', "gp-matern": 'GP (Mater\u00e9n)',
              "rff": 'RFF-BO'}
    cols = {"ssp-hex": utils.blues[0], "ssp-rand": utils.oranges[0],
            "gp-sinc": utils.greens[0], "gp-matern": utils.reds[0],
            "rff": utils.purples[0]}
    linestys = {"ssp-hex": '-', "ssp-rand": '--',
                "gp-sinc": ':', "gp-matern": '-.',
                "rff": (0, (3, 1, 1, 1, 1, 1))}

    max_num_trials = 60
    num_init = 10
    plt_datas = []
    for i, func in enumerate(funcs):
        plt_datass = []
        true_max_val = 0 if ("rastrigin" in func or "rosenbrock" in func) else function_maximum_value[func]

        for j, agt in enumerate(agts):
            folder = folders[0] if len(folders) == 1 else folders[j]
            data = pd.DataFrame(read(os.path.join(folder, func, agt)))
            good_runs = np.where([data['params'][k] is not np.nan for k in range(len(data['params']))])[0]

            if 'regret' in data.keys():
                regrets = np.array([data['regret'][k] for k in range(len(data['regret'])) if k in good_runs])
            elif 'vals' in data.keys():
                vals = np.array([data['vals'][k] for k in range(len(data['vals'])) if k in good_runs])
                regrets = true_max_val - vals
            else:
                print(data.keys())
                continue
            if regrets.shape[0] > max_num_trials:
                regrets = regrets[:max_num_trials, :]

            budget = regrets.shape[1]
            num_trials = regrets.shape[0]
            regrets = np.divide(np.cumsum(regrets, axis=1),
                                matlib.repmat(range(1, budget + 1), num_trials, 1))[:max_num_trials, num_init:]
            plt_datass.append(get_mean_and_ci(regrets).copy())
        plt_datas.append(plt_datass)

    mem_data = {}
    t_plt_datas = {}
    t_totals = {}
    for i, func in enumerate(funcs):
        t_plt_datas[func] = []
        t_totals[func] = {}
        mem_data[func] = {}
        for j, agt in enumerate(agts):
            folder = folders[0] if len(folders) == 1 else folders[j]
            data = pd.DataFrame(read(os.path.join(folder, func, agt)))
            good_runs = np.where([data['params'][k] is not np.nan for k in range(len(data['params']))])[0]

            if 'ssp' in agt:
                times = np.array([data['full_times'][k] for k in range(len(data['full_times'])) if k in good_runs]) * 1e-9
            else:
                times = np.array([data['times'][k] for k in range(len(data['times'])) if k in good_runs]) * 1e-9
            times = times[:max_num_trials, :]
            total_time = np.sum(times, axis=-1)
            t_plt_datas[func].append(get_mean_and_ci(times))
            t_totals[func][agt] = (np.mean(total_time), np.std(total_time))
            mem = np.array([data['memory'][k] for k in range(len(data['memory'])) if k in good_runs]).squeeze()[:max_num_trials, :]
            mem = mem[:, num_init:] - mem[:, num_init].reshape(-1, 1)
            mem_data[func][agt] = get_mean_and_ci(mem)

    letters = ["\\textbf{a}", "\\textbf{b}", "\\textbf{c}", "\\textbf{d}"]
    letters = [l + " $\\quad$ " for l in letters]
    fontsize = 9

    fig = plt.figure(figsize=(8.5, 3.5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.6, wspace=0.2)
    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for i, func in enumerate(funcs):
        axs[i].set_title(letters[i] + 'Average Regret: ' + func.title(), fontsize=fontsize)
        for j, agt in enumerate(agts):
            budget = len(plt_datas[i][j]["mean"])
            plt_data = plt_datas[i][j].copy()
            axs[i].fill_between(np.arange(num_init, budget + num_init),
                                plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[agt])
            axs[i].plot(np.arange(num_init, budget + num_init), plt_data["mean"],
                        color=cols[agt], label=labels[agt], linestyle=linestys[agt])
        axs[i].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
        axs[i].set_ylabel("$\\leftarrow$ Average Regret", fontsize=fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
    axs[1].legend(fontsize=fontsize - 2)

    time_func = "branin-hoo"
    axs[-1].set_title(letters[-1] + 'Sample Selection Time: ' + time_func.title(), fontsize=fontsize)
    for j, agt in enumerate(agts):
        plt_data = t_plt_datas[time_func][j]
        budget = len(plt_data["mean"])
        t = np.arange(num_init, budget + num_init)
        axs[-1].fill_between(t, plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[agt])
        axs[-1].plot(t, plt_data["mean"], color=cols[agt], label=labels[agt], linestyle=linestys[agt])
    axs[-1].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
    axs[-1].set_ylabel("Sample Selection Time (sec)", fontsize=fontsize)
    axs[-1].tick_params(axis='both', which='major', labelsize=fontsize)

    utils.save(fig, 'test-func_regret.pdf')

    for i, func in enumerate(funcs):
        print(f'{func}, SSP-BO, {t_totals[func]["ssp-hex"][0]:.2f}, {t_totals[func]["ssp-hex"][1]:.2f}')
        print(f'{func}, GP, {t_totals[func]["gp-matern"][0]:.2f}, {t_totals[func]["gp-matern"][1]:.2f}')
        print(f'Improvement, {t_totals[func]["gp-matern"][0] / t_totals[func]["ssp-hex"][0]:.2f}x')

    fig_mem = plt.figure(figsize=(8.5, 1.5))
    gs_mem = GridSpec(1, 3, figure=fig_mem, hspace=0.6, wspace=0.2)
    axs_m = [fig_mem.add_subplot(gs_mem[0])]
    axs_m.append(fig_mem.add_subplot(gs_mem[1], sharey=axs_m[0]))
    axs_m.append(fig_mem.add_subplot(gs_mem[2], sharey=axs_m[0]))
    for i, func in enumerate(funcs):
        axs_m[i].set_title(letters[i] + 'Memory: ' + func.title(), fontsize=fontsize)
        for j, agt in enumerate(agts):
            plt_data = mem_data[func][agt]
            budget = len(plt_data["mean"])
            axs_m[i].fill_between(np.arange(num_init, budget + num_init),
                                  plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[agt])
            axs_m[i].plot(np.arange(num_init, budget + num_init), plt_data["mean"],
                          color=cols[agt], label=labels[agt], linestyle=linestys[agt])
        axs_m[i].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
        axs_m[i].tick_params(axis='both', which='major', labelsize=fontsize)
    axs_m[0].legend(fontsize=fontsize - 2)
    axs_m[0].set_ylabel("$\\Delta$ Memory (MB)", fontsize=fontsize)
    utils.save(fig_mem, 'test-func_memory.pdf')
