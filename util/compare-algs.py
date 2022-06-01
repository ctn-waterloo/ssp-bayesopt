'''
A short script to plot the outputs of the SSP Mutual Information sampling
Is currently assuming that the data is stored in the format:
    /<path-to>/<test-function-name>/<selection-agent>
    where <selection-agent> is one of {gp-mi,ssp-mi}
'''


import numpy as np
import numpy.matlib as matlib
import pandas as pd
import pytry
import matplotlib.pyplot as plt
import matplotlib as mpl
import os.path



from argparse import ArgumentParser
import best

def get_data(data_frame):
    regret = data_frame['regret']
    cum_regret = np.cumsum(regret)
    avg_regret = np.divide(cum_regret, np.arange(1, regret.shape[0] + 1))

    if 'times' in data_frame:
        time = data_frame['times'] * 1e-9
    else:
        time = data_frame['elapsed_time'] * 1e-9
    return regret, avg_regret, time, data_frame['seed']


def get_stats(data):
    (num_trials, _) = data.shape
    return np.mean(data, axis=0), np.std(data, axis=0) * 1.96 / np.sqrt(num_trials)

def pair_data(alg_regrets, alg_seeds, baseline_regrets, baseline_seeds):
    paired_alg = np.zeros((len(alg_seeds),))
    paired_baseline = np.zeros((len(alg_seeds),))
    for seed_idx, alg_seed in enumerate(alg_seeds):
        baseline_idx = baseline_seeds.index(alg_seed)
        paired_alg[seed_idx] = alg_regrets[seed_idx]
        paired_baseline[seed_idx] = baseline_regrets[baseline_idx]
    return paired_alg, paired_baseline

if __name__ == '__main__':


    parser = ArgumentParser(description='Plot results in folders')
    parser.add_argument('--alg', action='store', metavar=('name', 'dir'), nargs=2)
    parser.add_argument('--baseline', action='store', metavar=('name', 'dir'), nargs=2)

    args = parser.parse_args()


    def parse(func_dir):
        head, _ = os.path.split(func_dir)
        head, func = os.path.split(head)

        alg_data = (pytry.read(func_dir))
        alg_regret, avg_alg_regret, alg_time, alg_seeds = zip(*[get_data(f) for f in alg_data])
        alg_regret = np.array(alg_regret).squeeze()
        avg_alg_regret = np.array(avg_alg_regret).squeeze()

        (alg_num_trials, budget) = alg_regret.shape

        print(avg_alg_regret.shape)
        return avg_alg_regret, func, alg_seeds

    alg_name, alg_dir = args.alg
    baseline_name, baseline_dir = args.baseline

    alg_regret, func_name, alg_seeds = parse(alg_dir)
    baseline_regret, baseline_func_name, baseline_seeds = parse(baseline_dir)

    alg_terminal_regret = alg_regret[:,-1]
    baseline_terminal_regret = baseline_regret[:,-1]

    paired_alg_term_regret, paired_baseline_term_regret = pair_data(
                alg_terminal_regret, 
                alg_seeds, 
                baseline_terminal_regret, 
                baseline_seeds
    )

    assert func_name == baseline_func_name

    def cohens_d(xs, ys):
        mu_x = np.mean(xs)
        mu_y = np.mean(ys)
        n_x = xs.shape[0]
        n_y = ys.shape[0]
        s_x = np.std(xs)
        s_y = np.std(ys)

        pooled_std = np.sqrt(((n_x-1)*s_x**2 + (n_y-1)*s_y**2)/(n_x+n_y-2))

        return (mu_x - mu_y) / pooled_std

    from scipy.stats import sem, wilcoxon
#     print(f'{alg_name} terminal regret: {np.mean(alg_terminal_regret)} +- {np.std(alg_terminal_regret)/np.sqrt(30)}')
    print(f'{alg_name} terminal regret: {np.mean(alg_terminal_regret)} +- {sem(alg_terminal_regret)}')
#     print(f'{baseline_name} terminal regret: {np.mean(baseline_terminal_regret)} +- {np.std(baseline_terminal_regret)/np.sqrt(30)}')
    print(f'{baseline_name} terminal regret: {np.mean(baseline_terminal_regret)} +- {sem(baseline_terminal_regret)}')

    regret_diff = paired_baseline_term_regret - paired_alg_term_regret
    print('mean:', np.mean(regret_diff), sem(regret_diff))
    print(wilcoxon(paired_alg_term_regret,paired_baseline_term_regret))
    print('effect: ', cohens_d(paired_baseline_term_regret, paired_alg_term_regret))

    np.save(f'{func_name}_{alg_name}_term-regret.npz',paired_alg_term_regret)
    np.save(f'{func_name}_{baseline_name}_term-regret.npz',paired_baseline_term_regret)

    exit()





    end_results = best.unpaired(baseline_terminal_regret, alg_terminal_regret)

    num_obs = alg_regret.shape[1]
    print(f'{alg_name}, {baseline_name}')
    print(f'{func_name} & {num_obs} & {end_results["difference of means"]["mean"]:.2f} ' +
                  f'& [{end_results["difference of means"]["hdi"][0]:.2f}, {end_results["difference of means"]["hdi"][1]:.2f}] ' +
                  f'& {end_results["difference of stds"]["mean"]:.2f} ' +
                  f'& [{end_results["difference of stds"]["hdi"][0]:.2f}, {end_results["difference of stds"]["hdi"][1]:.2f}] ' +
                  f'& {end_results["effect size"]["mean"]:.2f} ' + f'& [{end_results["effect size"]["hdi"][0]:.2f}, {end_results["effect size"]["hdi"][1]:.2f}] ' +
                  '\\\\')

 
#         exit()

#         avg_alg_regret_mu, avg_alg_regret_ste = get_stats(avg_alg_regret)
#         alg_stats[(func,alg_name)] = (alg_regret, 
#                             alg_time,
#                             avg_alg_regret_mu,
#                             avg_alg_regret_ste,
#                             budget)
    ### end for


#     plt.figure(figsize=(5.5/3,1.5))
#     func_name = ''
#     line_styles = {0:'solid', 1:'dashed', 2:'dotted', 3:'dashdot'}
#     for idx, (k, v) in enumerate(alg_stats.items()):
#         print(k)
#         budget = v[4]
#         regret_steps = range(1, budget+1)
# 
#         avg_regret_mu = v[2]
#         avg_regret_ste = v[3]
#         plt.fill_between(regret_steps, 
#                          avg_regret_mu - avg_regret_ste,
#                          avg_regret_mu + avg_regret_ste, alpha=0.3)
#         plt.plot(regret_steps, avg_regret_mu, label=k[1], ls=line_styles[idx])
#         func_name = k[0]
# 
#     # Get rid of spines
#     plt.gca().spines['left'].set_position(('outward', 10))
#     plt.gca().spines['bottom'].set_position(('outward', 10))
# 
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False) 
# 
#     plt.legend()
#     plt.ylabel('Average Regret (a.u.)')#, fontsize=24)
#     plt.xlabel('Sample Number')#, fontsize=24)
#     plt.title(f'Average Regret: {func_name.title()}')
#     plt.tight_layout()
# 
