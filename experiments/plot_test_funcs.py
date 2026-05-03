"""Plot average regret and sample-selection time for standard test functions.

Reads .npz result files saved by run_agent.py. Pass --data-dir to point at the
results folder and --fig-dir to control where the PDFs are saved. Missing
(function, agent) combinations are skipped with a printed notice; only
combinations that have data are plotted.

Usage:
    python plot_test_funcs.py --data-dir experiments/data/d97 --fig-dir experiments/figs
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
                        default=os.path.join(os.getcwd(), 'experiments', 'data', 'd97'))
    parser.add_argument('--fig-dir', dest='fig_dir', type=str,
                        default=os.getcwd(),
                        help='Directory to save the generated PDF figures into.')
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)

    funcs = ["himmelblau", "goldstein-price", "branin-hoo"]
    folders = [args.data_dir]
    agts = ["ssp-hex", "ssp-rand", "gp-sinc", "gp-matern", "rff"]
    labels = {"ssp-hex": 'SSP-BO-Hex', "ssp-rand": 'SSP-BO-Rand',
              "gp-sinc": 'GP (sinc)', "gp-matern": 'GP (Materén)',
              "rff": 'RFF-BO'}
    cols = {"ssp-hex": utils.blues[0], "ssp-rand": utils.oranges[0],
            "gp-sinc": utils.greens[0], "gp-matern": utils.reds[0],
            "rff": utils.purples[0]}
    linestys = {"ssp-hex": '-', "ssp-rand": '--',
                "gp-sinc": ':', "gp-matern": '-.',
                "rff": (0, (3, 1, 1, 1, 1, 1))}

    max_num_trials = 60
    num_init = 10

    # Per-function maps of agent -> bootstrap CI dict; missing combinations
    # are simply omitted so downstream plotting can skip them.
    regret_data = {func: {} for func in funcs}
    time_data = {func: {} for func in funcs}
    mem_data = {func: {} for func in funcs}
    t_totals = {func: {} for func in funcs}

    for func in funcs:
        true_max_val = 0 if ("rastrigin" in func or "rosenbrock" in func) else function_maximum_value[func]
        for agt in agts:
            folder = folders[0] if len(folders) == 1 else folders[agts.index(agt)]
            agt_path = os.path.join(folder, func, agt)
            try:
                if not os.path.isdir(agt_path):
                    print(f'[skip] {func}/{agt}: folder not found ({agt_path})')
                    continue

                raw = read(agt_path)
                if len(raw) == 0:
                    print(f'[skip] {func}/{agt}: no result files in {agt_path}')
                    continue

                data = pd.DataFrame(raw)
                good_runs = np.where([data['params'][k] is not np.nan
                                      for k in range(len(data['params']))])[0]
                if len(good_runs) == 0:
                    print(f'[skip] {func}/{agt}: no valid trials')
                    continue

                if 'regret' in data.keys():
                    regrets = np.array([data['regret'][k] for k in range(len(data['regret']))
                                        if k in good_runs])
                elif 'vals' in data.keys():
                    vals = np.array([data['vals'][k] for k in range(len(data['vals']))
                                     if k in good_runs])
                    regrets = true_max_val - vals
                else:
                    print(f'[skip] {func}/{agt}: no regret or vals key found')
                    continue

                if regrets.shape[0] > max_num_trials:
                    regrets = regrets[:max_num_trials, :]

                budget = regrets.shape[1]
                num_trials = regrets.shape[0]
                cum_regret = np.divide(np.cumsum(regrets, axis=1),
                                       matlib.repmat(range(1, budget + 1), num_trials, 1)
                                       )[:max_num_trials, num_init:]
                regret_data[func][agt] = get_mean_and_ci(cum_regret).copy()

                # Sample-selection times.
                try:
                    if 'ssp' in agt:
                        times = np.array([data['full_times'][k]
                                          for k in range(len(data['full_times']))
                                          if k in good_runs]) * 1e-9
                    else:
                        times = np.array([data['times'][k]
                                          for k in range(len(data['times']))
                                          if k in good_runs]) * 1e-9
                    times = times[:max_num_trials, :]
                    total_time = np.sum(times, axis=-1)
                    time_data[func][agt] = get_mean_and_ci(times)
                    t_totals[func][agt] = (np.mean(total_time), np.std(total_time))
                except Exception as e:
                    print(f'[warn] {func}/{agt}: timing data unavailable ({e})')

                # Memory.
                try:
                    mem = np.array([data['memory'][k]
                                    for k in range(len(data['memory']))
                                    if k in good_runs]).squeeze()
                    if mem.ndim == 1:
                        mem = mem.reshape(1, -1)
                    mem = mem[:max_num_trials, :]
                    mem = mem[:, num_init:] - mem[:, num_init].reshape(-1, 1)
                    mem_data[func][agt] = get_mean_and_ci(mem)
                except Exception as e:
                    print(f'[warn] {func}/{agt}: memory data unavailable ({e})')

            except Exception as e:
                print(f'[skip] {func}/{agt}: error reading data ({e})')
                continue

    funcs_with_regret = [f for f in funcs if len(regret_data[f]) > 0]
    if not funcs_with_regret:
        print('No regret data found in any function folder; nothing to plot.')
        raise SystemExit(0)

    letters = ["\\textbf{a}", "\\textbf{b}", "\\textbf{c}", "\\textbf{d}"]
    letters = [l + " $\\quad$ " for l in letters]
    fontsize = 9

    n_regret_panels = len(funcs_with_regret)
    # Pick a function for the timing panel (prefer branin-hoo if present).
    time_func = 'branin-hoo' if 'branin-hoo' in funcs_with_regret and len(time_data['branin-hoo']) > 0 else None
    if time_func is None:
        for f in funcs_with_regret:
            if len(time_data[f]) > 0:
                time_func = f
                break

    n_panels = n_regret_panels + (1 if time_func is not None else 0)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig = plt.figure(figsize=(8.5, 1.75 * n_rows + 0.25))
    gs = GridSpec(n_rows, n_cols, figure=fig, hspace=0.6, wspace=0.2)
    axs = [fig.add_subplot(gs[idx // n_cols, idx % n_cols]) for idx in range(n_panels)]

    for i, func in enumerate(funcs_with_regret):
        title_letter = letters[i] if i < len(letters) else ''
        axs[i].set_title(title_letter + 'Average Regret: ' + func.title(), fontsize=fontsize)
        for agt in agts:
            if agt not in regret_data[func]:
                continue
            plt_data = regret_data[func][agt]
            budget = len(plt_data["mean"])
            axs[i].fill_between(np.arange(num_init, budget + num_init),
                                plt_data["upper_bound"], plt_data["lower_bound"],
                                alpha=.2, color=cols[agt])
            axs[i].plot(np.arange(num_init, budget + num_init), plt_data["mean"],
                        color=cols[agt], label=labels[agt], linestyle=linestys[agt])
        axs[i].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
        axs[i].set_ylabel("$\\leftarrow$ Average Regret", fontsize=fontsize)
        axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
        axs[i].legend(fontsize=fontsize - 2)

    if time_func is not None:
        time_letter = letters[n_regret_panels] if n_regret_panels < len(letters) else ''
        axs[-1].set_title(time_letter + 'Sample Selection Time: ' + time_func.title(),
                          fontsize=fontsize)
        for agt in agts:
            if agt not in time_data[time_func]:
                continue
            plt_data = time_data[time_func][agt]
            budget = len(plt_data["mean"])
            t = np.arange(num_init, budget + num_init)
            axs[-1].fill_between(t, plt_data["upper_bound"], plt_data["lower_bound"],
                                 alpha=.2, color=cols[agt])
            axs[-1].plot(t, plt_data["mean"], color=cols[agt],
                         label=labels[agt], linestyle=linestys[agt])
        axs[-1].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
        axs[-1].set_ylabel("Sample Selection Time (sec)", fontsize=fontsize)
        axs[-1].tick_params(axis='both', which='major', labelsize=fontsize)

    cwd = os.getcwd()
    try:
        os.chdir(args.fig_dir)
        utils.save(fig, 'test-func_regret.pdf')
    finally:
        os.chdir(cwd)

    for func in funcs_with_regret:
        if 'ssp-hex' in t_totals[func] and 'gp-matern' in t_totals[func]:
            print(f'{func}, SSP-BO, {t_totals[func]["ssp-hex"][0]:.2f}, {t_totals[func]["ssp-hex"][1]:.2f}')
            print(f'{func}, GP, {t_totals[func]["gp-matern"][0]:.2f}, {t_totals[func]["gp-matern"][1]:.2f}')
            print(f'Improvement, {t_totals[func]["gp-matern"][0] / t_totals[func]["ssp-hex"][0]:.2f}x')

    funcs_with_mem = [f for f in funcs_with_regret if len(mem_data[f]) > 0]
    if funcs_with_mem:
        n_mem = len(funcs_with_mem)
        fig_mem = plt.figure(figsize=(2.85 * n_mem, 1.5))
        gs_mem = GridSpec(1, n_mem, figure=fig_mem, hspace=0.6, wspace=0.2)
        axs_m = [fig_mem.add_subplot(gs_mem[0])]
        for k in range(1, n_mem):
            axs_m.append(fig_mem.add_subplot(gs_mem[k], sharey=axs_m[0]))
        for i, func in enumerate(funcs_with_mem):
            title_letter = letters[i] if i < len(letters) else ''
            axs_m[i].set_title(title_letter + 'Memory: ' + func.title(), fontsize=fontsize)
            for agt in agts:
                if agt not in mem_data[func]:
                    continue
                plt_data = mem_data[func][agt]
                budget = len(plt_data["mean"])
                axs_m[i].fill_between(np.arange(num_init, budget + num_init),
                                      plt_data["upper_bound"], plt_data["lower_bound"],
                                      alpha=.2, color=cols[agt])
                axs_m[i].plot(np.arange(num_init, budget + num_init), plt_data["mean"],
                              color=cols[agt], label=labels[agt], linestyle=linestys[agt])
            axs_m[i].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
            axs_m[i].tick_params(axis='both', which='major', labelsize=fontsize)
        axs_m[0].legend(fontsize=fontsize - 2)
        axs_m[0].set_ylabel("$\\Delta$ Memory (MB)", fontsize=fontsize)

        try:
            os.chdir(args.fig_dir)
            utils.save(fig_mem, 'test-func_memory.pdf')
        finally:
            os.chdir(cwd)
    else:
        print('No memory data available; skipping memory plot.')
