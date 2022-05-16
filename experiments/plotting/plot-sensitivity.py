import numpy as np
import pytry
import glob
import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl

from argparse import ArgumentParser

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


def get_data(data_frame):
    ssp_dim = data_frame['ssp_dim']
    try:
        regret = data_frame['regret']
        cum_regret = np.cumsum(regret)
        avg_regret = np.divide(cum_regret, np.arange(1, regret.shape[0] + 1))
        return ssp_dim, avg_regret[-1]
    except:
        print(data_frame.keys())
        exit()
    ### end try
### end get_data

from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression

def preprocess_data(indep, dep):
    indeps = np.array(indep).reshape((-1,1))

    deps = np.array(dep).reshape(indeps.shape[0],1)

    return indeps, deps

def compute_stats(xs, ys):
    x_vals = np.unique(xs)
    mus = np.array([np.mean(ys[xs==x]) for x in x_vals])
    sems = np.array([stats.sem(ys[xs==x]) for x in x_vals])

    return x_vals, mus, sems


if __name__=='__main__':

    parser = ArgumentParser(description='Plot sensitivity to SSP dimension')
    parser.add_argument('folder', type=str)
    parser.add_argument('--save', action='store_true')

    args = parser.parse_args()

    files = pytry.read(os.path.abspath(args.folder))

    ssp_dim, terminal_avg_regret = zip(*[get_data(f) for f in files])

    X, Y = preprocess_data(ssp_dim, terminal_avg_regret)

    model = LinearRegression(fit_intercept=True)
    num_repeats = 10000
    num_splits = 2

    num_features = X.shape[1]
    kfold = RepeatedKFold(n_splits=num_splits, n_repeats=num_repeats)
    
    scores = np.zeros((num_splits * num_repeats,))
    coeffs = np.zeros((num_splits * num_repeats, num_features+1))
    for i, (train, test) in enumerate(kfold.split(X, Y)):
        model.fit(X[train,:], Y[train])
        coeffs[i,:num_features] = model.coef_
        coeffs[i,-1] = model.intercept_

        scores[i] = model.score(X[test,:], Y[test])


#     plt.figure()
#     stes = 1.96 * np.std(coeffs, axis=0) / np.sqrt(num_splits * num_repeats)
#     plt.bar([0, 1], np.mean(coeffs, axis=0), 0.5, yerr=stes)
   
    plt.figure()
    x_unique, mu, sem = compute_stats(X,Y)

    color =''
    line_style = ''
    if 'hex' in args.folder:
        alg_type = 'Hex'
        color = 'tab:orange'
        line_style = 'dashed'
    if 'rand' in args.folder:
        alg_type = 'Rand'
        color = 'tab:blue'
        line_style = 'dashed'

    plt.fill_between(x_unique, mu-sem, mu+sem, alpha=0.3, color=color)
    plt.plot(x_unique, mu, label=f'{alg_type} SSP', c=color, ls=line_style)
    plt.scatter([151], mu[-2], color='tab:blue', zorder=10)
    
    matern_mu = 5.15
    matern_ste = 3.38 / np.sqrt(30)

    sinc_mu = 14.12
    sinc_ste = 13.40 / np.sqrt(30)

    plt.fill_between(x_unique,
            (sinc_mu - sinc_ste) * np.ones(x_unique.shape),
            (sinc_mu + sinc_ste) * np.ones(x_unique.shape),
            alpha=0.3, color='tab:green')
    plt.plot(x_unique, sinc_mu * np.ones(x_unique.shape), 
            label='GP-Sinc', ls='dotted', color='tab:green')

    plt.fill_between(x_unique,
            (matern_mu - matern_ste) * np.ones(x_unique.shape),
            (matern_mu + matern_ste) * np.ones(x_unique.shape),
            alpha=0.3, color='tab:red')
    plt.plot(x_unique, matern_mu * np.ones(x_unique.shape), 
            label='GP-Matern', ls='dashdot', color='tab:red')


    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 

    plt.legend(fontsize=18)
    plt.ylabel('Terminal Average Regret (a.u.)', fontsize=18)
    plt.xlabel('SSP Dimension', fontsize=18)
    plt.title(f'Terminal Average Regret vs SSP Dimension\n(Branin-Hoo, N=30, ' + r'$\beta = ' + f'{np.mean(coeffs[:,0]):.3f}'+'$)', fontsize=18)
    plt.tight_layout()

    if args.save:
        plt.savefig(f'{alg_type}-regret-vs-ssp_dim.pgf')
    else:
        plt.show()
