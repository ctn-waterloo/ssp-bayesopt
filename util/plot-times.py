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
    parser.add_argument('--alg', action='append', metavar=('name', 'dir'), nargs=2)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()
    names, folders = zip(*args.alg)

    if args.save:
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

    plot_filetype = 'pgf'

 
    alg_stats = {}
    for alg_name, func_dir in args.alg:
        head, _ = os.path.split(func_dir)
        head, func = os.path.split(head)

        alg_data = pd.DataFrame(pytry.read(func_dir))
        alg_regret, _, alg_time = get_data(alg_data)

        (alg_num_trials, budget) = alg_time.shape

        avg_alg_time_mu, avg_alg_time_ste = get_stats(alg_time)
        alg_stats[(func,alg_name)] = (alg_time, 
                            avg_alg_time_mu,
                            avg_alg_time_ste,
                            budget)
    ### end for


    plt.figure()#figsize=(5.5/3,1.5))
    func_name = ''
    line_styles = {0:'solid', 1:'dashed', 2:'dotted', 3:'dashdot'}
    for idx, (k, v) in enumerate(alg_stats.items()):
        print(k)
        budget = v[3]
        regret_steps = range(1, budget+1)

        avg_regret_mu = v[1]
        avg_regret_ste = v[2]
#         plt.subplot(len(line_styles.keys()),1,1+idx)
        plt.fill_between(regret_steps, 
                         avg_regret_mu - avg_regret_ste,
                         avg_regret_mu + avg_regret_ste, alpha=0.3)
        plt.plot(regret_steps, avg_regret_mu, label=k[1], ls=line_styles[idx])
        func_name = k[0]

    # Get rid of spines
    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 

    plt.legend()
    plt.ylabel('Query Time (sec)')#, fontsize=24)
    plt.xlabel('Sample Number')#, fontsize=24)
    plt.title(f'Query Time: {func_name.title()}')
    plt.tight_layout()

    if args.save:
        plt.savefig(f'{func_name}-time.{plot_filetype}')
    else:
        plt.show()

