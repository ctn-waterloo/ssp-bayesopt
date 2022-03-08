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

display = True
if not display:
    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        })
### end if
mpl.rcParams.update({
    'font.family': 'serif',
    'figure.autolayout': True
    })


from argparse import ArgumentParser
import best

def get_data(data_frame):
    regret = np.vstack(data_frame['regret'].values)
#     avg_regret = np.vstack(data_frame['avg_regret'].values)
    avg_regret = np.divide(np.cumsum(regret, axis=1), np.cumsum(np.arange(1,regret.shape[1]+1)))
    if 'times' in data_frame.columns:
        time = np.vstack(data_frame['times'].values) * 1e-9
    else:
        time = np.vstack(data_frame['elapsed_time'].values) * 1e-9
    return regret, avg_regret, time

def get_stats(data):
    (num_trials, _) = data.shape
    return np.mean(data, axis=0), np.std(data, axis=0) * 1.96 / np.sqrt(num_trials)

if __name__ == '__main__':


    parser = ArgumentParser(description='Plot results in folders')
    parser.add_argument('folders', metavar='dirs', type=str, nargs='+')

    args = parser.parse_args()

    plot_filetype = 'png'

#     print(args.folders)
#     exit()
  
    print('Function & Test point & Diff of Mus & HDI & Diff of Stds & HDI & Effect Size & HDI\\\\' )
    for func_name in args.folders:
#         func_name='himmelblau'
#         gp_data = pd.DataFrame(pytry.read(f'{func_name}/gp-mi'))
#         gp_data = pd.DataFrame(pytry.read(f'{func_name}/matern.static-gp'))
        gp_data = pd.DataFrame(pytry.read(f'{func_name}/static-gp'))
        gp_regret, gp_avg_regret, gp_time = get_data(gp_data)

        ssp_data = pd.DataFrame(pytry.read(f'{func_name}/ssp-rand'))
        ssp_regret, ssp_avg_regret, ssp_time = get_data(ssp_data)

#         hex_data = pd.DataFrame(pytry.read(f'{func_name}/hex-mi'))
#         hex_regret, hex_avg_regret, hex_time = get_data(hex_data)

        (num_trials, num_obs) = gp_regret.shape
        if False:
            mid_results = best.unpaired(gp_regret[:,num_obs // 2], ssp_regret[:,num_obs // 2])
            print(f'{func_name} & {num_obs // 2} & {mid_results["difference of means"]["mean"]:.2f} ' +
                    f'& [{mid_results["difference of means"]["hdi"][0]:.2f}, {mid_results["difference of means"]["hdi"][1]:.2f}] ' +
                  f'& {mid_results["difference of stds"]["mean"]:.2f} ' +
                  f'& [{mid_results["difference of stds"]["hdi"][0]:.2f}, {mid_results["difference of stds"]["hdi"][1]:.2f}] ' +
                  f'& {mid_results["effect size"]["mean"]:.2f} ' +
                  f'& [{mid_results["effect size"]["hdi"][0]:.2f}, {mid_results["effect size"]["hdi"][1]:.2f}] ' +
                  '\\\\')
            end_results = best.unpaired(gp_regret[:,num_obs-1], ssp_regret[:,num_obs-1])
            print(f'{func_name} & {num_obs} & {end_results["difference of means"]["mean"]:.2f} ' +
                  f'& [{end_results["difference of means"]["hdi"][0]:.2f}, {end_results["difference of means"]["hdi"][1]:.2f}] ' +
                  f'& {end_results["difference of stds"]["mean"]:.2f} ' +
                  f'& [{end_results["difference of stds"]["hdi"][0]:.2f}, {end_results["difference of stds"]["hdi"][1]:.2f}] ' +
                  f'& {end_results["effect size"]["mean"]:.2f} ' + f'& [{end_results["effect size"]["hdi"][0]:.2f}, {end_results["effect size"]["hdi"][1]:.2f}] ' +
                  '\\\\')


        (gp_num_trials, budget) = gp_regret.shape
        (ssp_num_trials, budget) = ssp_regret.shape
#         (hex_num_trials, budget) = hex_regret.shape

#         assert budget == 250
#         assert ssp_num_trials == gp_num_trials, f'Expected same number of trials but gp has {gp_num_trials} and ssp has {ssp_num_trials}'

        avg_gp_regret = np.divide(np.cumsum(gp_regret, axis=1), matlib.repmat(range(1,budget+1), gp_num_trials,1))
        avg_gp_regret_mu, avg_gp_regret_ste = get_stats(avg_gp_regret)

        avg_ssp_regret = np.divide(np.cumsum(ssp_regret, axis=1), matlib.repmat(range(1,budget+1), ssp_num_trials,1))
        avg_ssp_regret_mu, avg_ssp_regret_ste = get_stats(avg_ssp_regret)

#         avg_hex_regret = np.divide(np.cumsum(hex_regret, axis=1), matlib.repmat(range(1,budget+1), hex_num_trials,1))
#         avg_hex_regret_mu, avg_hex_regret_ste = get_stats(avg_hex_regret)

        plt.figure()
#         plt.subplot(2, 1, 1)
        regret_steps = range(1, budget+1)
        plt.fill_between(regret_steps, avg_gp_regret_mu - avg_gp_regret_ste, avg_gp_regret_mu + avg_gp_regret_ste, alpha=0.6)
        plt.plot(regret_steps, np.mean(avg_gp_regret, axis=0), label='GP-MI', ls='--')
        plt.fill_between(regret_steps, avg_ssp_regret_mu - avg_ssp_regret_ste, avg_ssp_regret_mu + avg_ssp_regret_ste, alpha=0.6)
        plt.plot(regret_steps, np.mean(avg_ssp_regret, axis=0), label='Rand SSP-MI')
#         plt.fill_between(regret_steps, avg_hex_regret_mu - avg_hex_regret_ste, avg_hex_regret_mu + avg_hex_regret_ste, alpha=0.6)
#         plt.plot(regret_steps, np.mean(avg_hex_regret, axis=0), label='Hex SSP-MI')

        # Get rid of spines
        plt.gca().spines['left'].set_position(('outward', 10))
        plt.gca().spines['bottom'].set_position(('outward', 10))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False) 

        plt.legend()
        plt.ylabel('Average Regret (a.u.)')#, fontsize=24)
        plt.xlabel('Sample Number')#, fontsize=24)
        plt.title(f'Average Regret for target: {func_name.title()}')
#         plt.title(f'Regret vs Sample Number, {func_name.title()}, N={ssp_num_trials}', fontsize=18)
        plt.tight_layout()

        if not display:
            plt.savefig(f'{func_name}-regret.{plot_filetype}')

#         plt.subplot(2, 1, 2)
        plt.figure()
        N=0
        steps = range(N,budget+1)
#         gp_time_mu, gp_time_ste = get_stats(np.cumsum(gp_time, axis=1))
#         ssp_time_mu, ssp_time_ste = get_stats(np.cumsum(ssp_time, axis=1))
#         hex_time_mu, hex_time_ste = get_stats(np.cumsum(hex_time, axis=1))

#         gp_time_mu, gp_time_ste = get_stats(gp_time)
#         ssp_time_mu, ssp_time_ste = get_stats(ssp_time)
# #         hex_time_mu, hex_time_ste = get_stats(hex_time)
#         
#         plt.fill_between(steps[N+11:], (gp_time_mu - gp_time_ste)[N:], (gp_time_mu + gp_time_ste)[N:], alpha=0.6)
#         plt.plot(steps[N+11:], gp_time_mu[N:], label='GP-MI', ls='--')
# 
#         plt.fill_between(steps[N:], (ssp_time_mu - ssp_time_ste)[N:], (ssp_time_mu + ssp_time_ste)[N:], alpha=0.6)
#         plt.plot(steps[N:], ssp_time_mu[N:], label='Rand SSP-MI')

#         plt.fill_between(steps[N:], (hex_time_mu - hex_time_ste)[N:], (hex_time_mu + hex_time_ste)[N:], alpha=0.6)
#         plt.plot(steps[N:], hex_time_mu[N:], label='Hex SSP-MI')


        plt.gca().spines['left'].set_position(('outward', 10))
        plt.gca().spines['bottom'].set_position(('outward', 10))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False) 

        plt.legend()
        plt.ylabel('Query Time (sec)')#, fontsize=24)
        plt.xlabel('Sample Number')#, fontsize=24)
        plt.title(f'Query Time vs Sample Number, {func_name.title()}, N={ssp_num_trials}')#, fontsize=18)
        plt.tight_layout()

        if not display:
            plt.savefig(f'{func_name}-time.{plot_filetype}')
        else:
            plt.show()

#     plt.show()
