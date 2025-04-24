
## Warning: this script is a complete mess and not for the faint of heart
import numpy as np
import sys, os, shutil
import matplotlib
from matplotlib import colors, colorbar
from matplotlib import cm
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec

import figure_utils as utils

from matplotlib.ticker import FixedLocator
from mcbo.utils.general_plot_utils import plot_curves_with_ranked_legends, COLORS, MARKERS, plot_task_regrets
from mcbo.utils.plot_resource_utils import get_color

from mcbo import RESULTS_DIR
from mcbo.utils.general_utils import plot_mean_std
from mcbo.optimizers.bo_builder import BO_ALGOS
from mcbo.utils.experiment_utils import get_opt, get_task_from_id


from typing import Optional, Dict, Any, Tuple, Union, Callable, List

from matplotlib.axes import Axes
from matplotlib.ticker import FixedLocator
from scipy.stats import t

from mcbo.utils.general_utils import plot_mean_std
from mcbo.utils.plot_resource_utils import COLORS, MARKERS

from mcbo import task_factory
from mcbo.utils.experiment_utils import run_experiment, get_task_from_id, get_opt


data_save_name = "data_mcbo"

ALL_RES_COMB_PATH = os.path.join(RESULTS_DIR, "data", data_save_name, "all_res_comb.csv")

# TASK_ID_TO_NAME = {
#     'Ackley Function': 'Ackley-20D',
#     'EDA Sequence Optimization - Design sin - Ops basic - Pattern basic - Obj both': 'AIG flow tuning',
#     '2DD8_S Antibody Design': 'Antibody design',
#     'MIG Sequence Optimisation - sqrt - both': 'MIG flow tuning',
# }
TASK_ID_TO_NAME = {
    'Ackley Function': 'Ackley-20D',
    'EDA Sequence Optimization - Design sin - Ops basic - Pattern basic - Obj both': 'AIG flow tuning',
    '2DD8_S Antibody Design': 'Antibody design',
    'MIG Sequence Optimisation - sqrt - both': 'MIG flow tuning',
}

task_ids = ['ackley', 'aig_optimization', 'antibody_design', 'mig_optimization', 'pest', 'rna_inverse_fold']
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

model_name_map={'ssp-mcbo': 'SSP-BO'}
    # 'ssp-mcbo_nengo-cpu': 'SSP-BO (SNN)'}


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
                            data.append(text(z.open(name)))
                        elif name.endswith('.npz'):
                            data.append(npz(z.open(name)))
                    except:
                        print('Error reading file "%s" in "%s"' % (name, fn))
                z.close()
            else:
                try:
                    if fn.endswith('.txt'):
                        data.append(text(fn))
                    elif fn.endswith('.npz'):
                        data.append(npz(fn))
                except:
                    print('Error reading file "%s"' % fn)
    return data


def text(fn):
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
        if d['array'] is numpy.array:
            del d['array']
        if d['nan'] is numpy.nan:
            del d['nan']
    return d


def npz(fn):
    import numpy as np
    d = {}
    f = np.load(fn, allow_pickle=True)
    for k in f.files:
        if k != 'ssp_space':
            d[k] = f[k]
            if d[k].shape == ():
                d[k] = d[k].item()
    return d


def get_ssp_data(task_id_map):
    # Collect new rows to add
    new_rows = []
    for task_id in task_id_map:
        data_folder = os.path.join(os.getcwd(),'data/ucb_beta10', task_id)
        for model_type in model_name_map:
            data = read(os.path.join(data_folder, model_type))
            for seed in range(len(data)):
                eval_nums = np.arange(len(data[seed]['vals']))
                f_x = -data[seed]['vals']
                f_x_star = np.minimum.accumulate(f_x)
                times = 1e-9 * np.cumsum(data[seed]['times'])  # need Elapsed Time like the other data
                for eval_num, fx, fx_star, time in zip(eval_nums, f_x, f_x_star, times):
                    new_row = {
                        'Task': task_id_map[task_id],
                        'Optimizer': model_name_map[model_type],
                        'Model': None,
                        'Acq opt': None,
                        'Acq func': None,
                        'TR': 'no-tr',
                        'Seed': seed,
                        'Eval Num': eval_num,
                        'f(x)': fx,
                        'f(x*)': fx_star,
                        'Elapsed Time': time
                    }
                    new_rows.append(new_row)

    # Append new rows to the existing DataFrame
    new_df = pd.DataFrame(new_rows)
    return new_df


ssp_tasks_cmbo = {
    'pest': 'Pest Control',
    'rna_inverse_fold': 'RNA Inverse Folding'
}
new_df = get_ssp_data(ssp_tasks_cmbo)

all_res = pd.concat([all_res, new_df], ignore_index=True)


def data_key_to_label(k: str):
    if k in BO_ALGO_FULL_NAME_TO_ALIAS:
        k = BO_ALGO_FULL_NAME_TO_ALIAS[k]
    else:
        k = k
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


x_fontsize = 15
y_fontsize = 15
y_label_fontsize = 22
x_label_fontsize = 17

xticks_labels_size = 14
yticks_labels_size = 14

legend_fontsize = 18
ax_title_fontsize = 20

final_num_eval = 200
x_evals = np.arange(0, 200)

ci_level = .95

all_opt_to_rank_dicts = {}
sub_regret_df = all_res.copy()

sub_regret_df["rank"] = sub_regret_df.groupby(["Task", "Eval Num", "Seed"])[["f(x*)"]].rank()
sub_regret_df.head()
algo_name_to_avg_rank = sub_regret_df[sub_regret_df["Eval Num"] == final_num_eval].groupby(["Optimizer"])["rank"].mean().to_dict()

best_unbranded_opt_names = sorted([k for k in algo_name_to_avg_rank if (k not in BO_ALGO_FULL_NAME_TO_ALIAS and k not in NON_BO_ALGO_NAMES)], key=lambda k: algo_name_to_avg_rank[k])
selected_algo_names = best_unbranded_opt_names[:2] + [k for k in BO_ALGO_FULL_NAME_TO_ALIAS.keys() if k in algo_name_to_avg_rank] + NON_BO_ALGO_NAMES
for name in model_name_map.values():
    selected_algo_names.append(name)
for nonboname in NON_BO_ALGO_NAMES:
    selected_algo_names.remove(nonboname)


num_evals = np.arange(20, 201)
group_col = "Optimizer"

opt_to_rank_dict = {}

sub_regret_df = sub_regret_df[sub_regret_df.Optimizer.isin(selected_algo_names)]

for num_eval in num_evals:
    for opt_name, rank_data in sub_regret_df[sub_regret_df["Eval Num"] == num_eval].groupby(group_col)["rank"]:
        if opt_name not in opt_to_rank_dict:
            opt_to_rank_dict[opt_name] = []
        opt_to_rank_dict[opt_name].append(rank_data.values)

for opt_name in opt_to_rank_dict:
    opt_to_rank_dict[opt_name] = np.array(opt_to_rank_dict[opt_name]).T
all_opt_to_rank_dicts["None"] = opt_to_rank_dict.copy()


def task_id_to_name(task_id: str) -> str:
    return TASK_ID_TO_NAME.get(task_id, task_id)


def plot_task_regrets(
        ax: Axes,
        data_y: Dict[str, np.ndarray],
        data_x: Union[np.ndarray, Dict[str, np.ndarray]],
        data_lb: Optional[Union[Dict[str, np.ndarray], np.ndarray, float]] = None,
        data_ub: Optional[Union[Dict[str, np.ndarray], np.ndarray, float]] = None,
        data_key_to_label: Optional[Union[Dict[str, str], Callable[[str], str]]] = None,
        data_marker: Optional[Dict[str, str]] = None,
        data_color: Optional[Dict[str, str]] = None,
        data_linestyle: Optional[Dict[str, str]] = None,
        alpha: float = .3,
        n_std: float = 1,
        linewidth: int = 3,
        marker_kwargs: Optional[Dict[str, Any]] = None,
        ci_level: Optional[float] = None,
        show_std_error: Optional[bool] = False,
        zoom_end_pct: Optional[float] = None
) -> Axes:
    """
    Plot curves with legends written vertically with position corresponding to the final values (final regrets, scores,
    ...) on the right of the plot.

    Args:
        data_lb: lower bound for confidence interval (for instance if values are known to be in [0, 1])
        data_ub: upper bound for confidence interval (for instance if values are known to be in [0, 1])
        data_key_to_label: map from keys of data_y to the labels that should appear as legend
        ci_level: show confidence interval over the mean at specified level (e.g. 0.95), otherwise uncertainty shows
                  n_std std around the mean
        show_std_error: show standard error (std / sqrt(n_samples)) as shadow area around mean curve
        zoom_end_pct: reset ylimits such that end performances occupies at least `zoom_end_pct` of the screen.

    Returns:
        ax: axis containing the plots

    """

    if marker_kwargs is None:
        marker_kwargs = {}

    if data_marker is None:
        data_marker = {data_key: MARKERS[i % len(MARKERS)] for i, data_key in enumerate(data_y)}

    if data_key_to_label is None:
        data_key_to_label = {data_k: data_k for data_k in data_y}
    if isinstance(data_key_to_label, dict):
        data_key_to_label_map = lambda k: data_key_to_label.get(k, k)
    else:
        data_key_to_label_map = data_key_to_label
    if not isinstance(data_x, dict):
        data_x = {data_key: data_x for data_key in data_y}

    if not isinstance(data_lb, dict):
        data_lb = {data_key: data_lb for data_key in data_y}

    if not isinstance(data_ub, dict):
        data_ub = {data_key: data_ub for data_key in data_y}

    if data_color is None:
        data_color = {data_key: COLORS[i % len(COLORS)] for i, data_key in enumerate(data_y)}

    max_y_end = -np.inf
    min_y_end = np.inf

    for data_key in data_y:
        x = data_x[data_key]
        y = data_y[data_key]

        if y.ndim == 1:
            y = y.reshape(1, -1)

        markers_on = np.round(np.linspace(0, len(x) - 1, 5)).astype(int)
        marker = data_marker.get(data_key)
        color = data_color.get(data_key)
        linestyle = data_linestyle.get(data_key)
        ax = plot_mean_std(
            x, y, lb=data_lb[data_key], ub=data_ub[data_key],
            linewidth=linewidth, ax=ax, color=color, alpha=alpha, n_std=n_std,
            ci_level=ci_level, show_std_error=show_std_error,
            label=data_key_to_label_map(data_key),
            marker=marker, markevery=markers_on, linestyle=linestyle, **marker_kwargs
        )

        max_y_end = max(y[:, -1].mean() + n_std * y[:, -1].std(), max_y_end)
        min_y_end = min(y[:, -1].mean() - n_std * y[:, -1].std(), min_y_end)

    ymin, ymax = ax.get_ylim()
    if zoom_end_pct:
        current_pct = (max_y_end - min_y_end) / (ymax - ymin)
        if current_pct < zoom_end_pct:
            gamma = 1 / ((ymax - ymin) - (max_y_end - min_y_end)) * (
                    ymax - ymin - (max_y_end - min_y_end) / zoom_end_pct)
            ymin = ymin + gamma * (min_y_end - ymin)
            ymax = ymax - gamma * (ymax - max_y_end)
            ax.set_ylim(ymin, ymax)

    return ax

# def broken_axis_plot(Gs, y1, y2, y3, y4,height_ratios=[1,1], **kwargs):
#     """
#     Creates a broken axis plot on a given grid spec slot with two y-axis ranges.
#
#     Parameters:
#         gs_slot (matplotlib.axes._subplots.AxesSubplot): The grid spec slot.
#         y1, y2 (float): Lower segment y-axis limits.
#         y3, y4 (float): Upper segment y-axis limits.
#         x_data (array-like): X values for the plot.
#         y_data (array-like): Y values for the plot.
#     """
#     # Create two subplots within the provided grid spec slot
#     gs_slot = gridspec.GridSpecFromSubplotSpec(2,1, subplot_spec=Gs,
#                                                height_ratios=height_ratios,hspace=0.1)
#                                                    #(y4-y3)/(y2-y1), 1], hspace=0.1)
#     ax1 = plt.subplot(gs_slot[0],**kwargs)  # Upper axis
#     ax2 = plt.subplot(gs_slot[1],**kwargs)  # Lower axis
#
#     # Set y-limits
#     ax1.set_ylim(y3, y4)  # Upper section
#     ax2.set_ylim(y1, y2)  # Lower section
#
#     # Hide x-ticks on the upper subplot
#     ax1.spines['bottom'].set_visible(True)
#     ax2.spines['top'].set_visible(True)
#     # ax1.xaxis.tick_top()
#     ax1.tick_params(
#         axis='x',          # changes apply to the x-axis
#         which='both',      # both major and minor ticks are affected
#         bottom=False,      # ticks along the bottom edge are off
#         top=False,         # ticks along the top edge are off
#         labelbottom=False,labeltop=False)
#     ax2.xaxis.tick_bottom()
#
#     # Create diagonal lines to indicate the axis break
#     d = .015  # size of break
#     kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
#     ax1.plot((-d, +d), (-d, +d), **kwargs)        # Upper left
#     ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # Upper right
#
#     kwargs.update(transform=ax2.transAxes)  # Apply to lower axis
#     ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # Lower left
#     ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # Lower right
#
#     return ax1, ax2
def broken_axis_plot(Gs, y_starts, y_ends, height_ratios=None, **kwargs):
    """
    Creates a broken axis plot with arbitrary number of y-axis ranges.

    Parameters:
        Gs (matplotlib.gridspec.SubplotSpec): The grid spec slot.
        y_starts (list of float): Start of each y-axis segment.
        y_ends (list of float): End of each y-axis segment.
        height_ratios (list of float, optional): Relative heights of the axis segments.
        **kwargs: Additional keyword arguments passed to each subplot.

    Returns:
        axes (list of matplotlib.axes.Axes): List of subplot axes.
    """
    assert len(y_starts) == len(y_ends), "y_starts and y_ends must be the same length"
    n_segments = len(y_starts)

    if height_ratios is None:
        height_ratios = [1] * n_segments

    gs_slot = gridspec.GridSpecFromSubplotSpec(n_segments, 1, subplot_spec=Gs,
                                               height_ratios=height_ratios, hspace=0.1)

    axes = [plt.subplot(gs_slot[i], **kwargs) for i in range(n_segments)]

    # Set y-limits and hide x-ticks for all but the bottom subplot
    for i, ax in enumerate(axes):
        ax.set_ylim(y_starts[i], y_ends[i])
        if i != n_segments - 1:
            ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    # Create diagonal lines to indicate breaks between axes
    d = .015  # size of break in axes
    for i in range(n_segments - 1):
        ax_upper = axes[i]
        ax_lower = axes[i + 1]
        kwargs_diag = dict(transform=ax_upper.transAxes, color='k', clip_on=False)
        # Upper ax (bottom break)
        ax_upper.plot((-d, +d), (-d, +d), **kwargs_diag)
        ax_upper.plot((1 - d, 1 + d), (-d, +d), **kwargs_diag)
        # Lower ax (top break)
        kwargs_diag.update(transform=ax_lower.transAxes)
        ax_lower.plot((-d, +d), (1 - d, 1 + d), **kwargs_diag)
        ax_lower.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs_diag)

    return axes

def make_plots2(n_col, n_el, data_color, data_linestyle, ssp_tasks_mix, linewidth=1.7,
                zoom_end_pct=.7, ci_level=.95, fontsize=9, legfontsize=6):
    x_evals = np.arange(0, 200)
    n_row = 1
    n_col = 5

    # fig, axes = plt.subplots(n_row, n_col, figsize=(4 * n_col, 3 * n_row),
    #                          width_ratios=[3 for _ in range(n_col - 1)] + [1], sharex=True,
    #                         )
    fig = plt.figure(figsize=(utils.doublecolwidth, 2))
    Gs = gridspec.GridSpec(n_row, n_col, figure=fig, hspace=0., wspace=0.15,
                           width_ratios=[2, 2, 0.2, 2, 0.5])

    # legend_ax_aux = axes[0, -1].get_gridspec()
    # # remove the underlying axes
    # for ax in axes[:, -1]:
    #     ax.remove()

    gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=Gs[:, -1], height_ratios=[0.1, 1])
    legend_ax = fig.add_subplot(gs[-1])

    ind = 0
    i_row = 0
    letters = ['\\textbf{a}$\\quad$', '\\textbf{b}$\\quad$', '\\textbf{c}$\\quad$', '\\textbf{d}$\\quad$',
               '\\textbf{e}$\\quad$', ]
    for task_id, task_group in all_res.groupby("Task"):
        if task_id == 'RNA Inverse Folding':
            j_col = 0
        elif task_id == 'Pest Control':
            j_col = 1
        else:
            continue

        ax = fig.add_subplot(Gs[j_col], sharex=ax if ind > 0 else None)

        res_regret_dic = {}
        for opt_name, task_opt_group in task_group.groupby("Optimizer"):
            if opt_name not in selected_algo_names:
                continue
            res = np.array([task_opt_seed_group["f(x*)"].values[x_evals] for _, task_opt_seed_group in
                            task_opt_group.groupby("Seed")])
            res_regret_dic[opt_name] = res

        ax = plot_task_regrets(ax=ax, data_x=x_evals, data_y=res_regret_dic, data_color=data_color,
                               data_key_to_label=data_key_to_label,
                               data_marker=data_marker,
                               ci_level=ci_level, show_std_error=False, zoom_end_pct=zoom_end_pct,
                               data_linestyle=data_linestyle, alpha=0.1, linewidth=linewidth)

        ax.set_title(letters[j_col] + task_id_to_name(task_id=task_id), fontsize=fontsize)
        ind += 1
        ax.tick_params(axis='both', which='major')
        ax.set_xlabel("Sample number ($n$)", fontsize=fontsize)
        # ax.set_xlabel("Sample number ($n$)")
        if j_col == 0:
            ax.set_ylabel("$\\leftarrow$ Best Objective Value", fontsize=fontsize)

        if task_id == 'Pest Control':
            # ax = fig.add_subplot(Gs[j_col+1], sharex = ax if ind>0 else None)
            # y1, y2, y3, y4 = 0.0, 0.05, 0.05, 65
            # ax, ax1 = broken_axis_plot(Gs[j_col + 2], y1, y2, y3, y4,
            #                            height_ratios=[1, 0.2],
            #                            sharex=ax)

            # y1, y2, y3, y4 = 0.0, 20, 20, 65
            # axs_broken = broken_axis_plot(Gs[j_col + 2], [y1, y3], [y2, y4],
            #                            height_ratios=[0.3,1],
            #                            sharex=ax)
            # y1, y2, y3, y4 = 0.0, 20, 20, 65
            axs_broken = broken_axis_plot(Gs[j_col + 2],
                                          [20, 0.03, 0.], [65, 20, 0.03],
                                          height_ratios=[0.4, 1, 0.2],
                                          sharex=ax)

            axs_broken[-1].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
            axs_broken[1].set_ylabel("Time (sec)", fontsize=fontsize)
            axs_broken[0].set_title('\\textbf{c}$\\quad$ Sample Selection Time: ' + task_id_to_name(task_id=task_id),
                         fontsize=fontsize)

            res_time_dic = {}
            for opt_name, task_opt_group in task_group.groupby("Optimizer"):

                if opt_name not in selected_algo_names:
                    continue
                res = np.array([task_opt_seed_group["Elapsed Time"].values[x_evals] for _, task_opt_seed_group in
                                task_opt_group.groupby("Seed")])
                # print(res.shape)
                res[:, 1:] = res[:, 1:] - res[:, :-1]
                res_time_dic[opt_name] = res
            for _ax in axs_broken:
                _ax = plot_task_regrets(ax=_ax, data_x=x_evals, data_y=res_time_dic, data_color=data_color,
                                       data_key_to_label=data_key_to_label,
                                       data_marker=data_marker, ci_level=ci_level, show_std_error=False, zoom_end_pct=None,
                                       data_linestyle=data_linestyle, alpha=0.1, linewidth=linewidth)
            # ax1 = plot_task_regrets(ax=ax1, data_x=x_evals, data_y=res_time_dic, data_color=data_color,
            #                         data_key_to_label=data_key_to_label,
            #                         data_marker=data_marker, ci_level=ci_level, show_std_error=False, zoom_end_pct=None,
            #                         data_linestyle=data_linestyle, alpha=0.1, linewidth=linewidth)

    handles, labels = ax.get_legend_handles_labels()
    # print(labels)
    ssplabidx = labels.index('SSP-BO')
    labels.insert(0, labels.pop(ssplabidx))
    handles.insert(0, handles.pop(ssplabidx))
    # ssplabidx = labels.index('SSP-BO (SNN)')
    # labels.insert(1, labels.pop(ssplabidx))
    # handles.insert(1, handles.pop(ssplabidx))

    legend_ax.set_axis_off()
    legend_ax.legend(handles, labels, bbox_to_anchor=(-.4, .5), loc='center left', labelspacing=1.1,
                     fontsize=legfontsize)
    return fig

# print(len(selected_algo_names))
COLORS = utils.cols + [utils.yellows[0], utils.grays[0], utils.reds[0], utils.greens[0], utils.grays[1]]
data_color = {data_key: COLORS[i] for i, data_key in enumerate(selected_algo_names)}
data_color['SSP-BO'] = utils.blues[0]
data_color['SSP-BO (SNN)'] = utils.purples[0]
data_marker = {data_key: None for i, data_key in enumerate(opt_to_rank_dict)}

LINESTYLES = [':', '--', '-.', (5, (10, 3)), (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (3, 1, 1, 1)), ':', '--', '-.',
              '-']
data_linestyle = {data_key: LINESTYLES[i] for i, data_key in enumerate(selected_algo_names)}
data_linestyle['SSP-BO'] = '-'
data_linestyle['SSP-BO (SNN)'] = '-'

n_el = len(task_ids)
n_col = 2 + 1
fig = make_plots2(n_col, n_el, data_color, data_linestyle, ssp_tasks_cmbo, linewidth=1, zoom_end_pct=0.1)

utils.save(fig, 'combinor_results_test.pdf')

timing_info = {}
for task_id, task_group in all_res.groupby("Task"):
    if (task_id != 'RNA Inverse Folding') and (task_id != 'Pest Control'):
        continue
    timing_info[task_id_to_name(task_id)] = {}
    for opt_name, task_opt_group in task_group.groupby("Optimizer"):
        if opt_name not in selected_algo_names:
            continue
        total_time = task_opt_group['Elapsed Time'][task_opt_group['Eval Num']==199]
        timing_info[task_id_to_name(task_id)][data_key_to_label(opt_name)] = (np.mean(total_time), np.std(total_time))

for task in ['RNA Inverse Folding', 'Pest Control']:
    methodss = list(timing_info[task].keys())
    min_method = methodss[np.argsort([t[1][0] for t in timing_info[task].items()])[1]]
    print(f'{task}, SSP-BO, {timing_info[task]["SSP-BO"]}')
    print(f'{task}, {min_method}, {timing_info[task][min_method]}')
    print(f'Improvement, {timing_info[task][min_method][0]/timing_info[task]["SSP-BO"][0]}')
timing_info