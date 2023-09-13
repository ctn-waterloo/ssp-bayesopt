import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import figure_utils as utils
import numpy.matlib as matlib


from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib import rc
import sys
import zipfile
import pandas as pd
import os
from matplotlib import cm


def get_mean_and_ci(raw_data, n=3000, p=0.95):
    """
    Gets the mean and 95% confidence intervals of data *see Note
    NOTE: data has to be grouped along rows, for example: having 5 sets of
    100 data points would be a list of shape (5,100)
    """
    sample = []
    upper_bound = []
    lower_bound = []
    sets = np.array(raw_data).shape[0]  # pylint: disable=E1136
    data_pts = np.array(raw_data).shape[1]  # pylint: disable=E1136
    print("Mean and CI calculation found %i sets of %i data points" % (sets, data_pts))
    raw_data = np.array(raw_data)
    for i in range(data_pts):
        data = raw_data[:, i]
        index = int(n * (1 - p) / 2)
        samples = np.random.choice(data, size=(n, len(data)))
        r = [np.mean(s) for s in samples]
        r.sort()
        ci = r[index], r[-index]
        sample.append(np.mean(data))
        lower_bound.append(ci[0])
        upper_bound.append(ci[1])

    data = {"mean": sample, "lower_bound": lower_bound, "upper_bound": upper_bound}
    return data



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
    f = np.load(fn,allow_pickle=True)
    for k in f.files:
        if k!='ssp_space':
            d[k] = f[k]
            if d[k].shape == ():
                d[k] = d[k].item()
    return d

# rc('font',size=7)
# rc('font',family='serif')
# rc('axes',labelsize=7)



def plot_paths(folder,nrows,ncols,idxs, savename=None):
    data = pd.DataFrame(read(folder))
    fig, axs = plt.subplots(figsize=(7, nrows*7/ncols), nrows=nrows,ncols=ncols)
    startidx = 200
    axs=axs.reshape(-1)
    for j in range(len(idxs)):
        i=idxs[j]
        axs[j].plot(data['path'].values[i][:,0], data['path'].values[i][:,1],
                    '-', color=utils.grays[2], label='Ground\n truth', linewidth=1.5)
        axs[j].plot(data['pi_sim_path'].values[i][startidx:,0], 
                    data['pi_sim_path'].values[i][startidx:,1], 
                    '--', color=utils.oranges[0],label='PI', linewidth=1.5)
        axs[j].plot(data['slam_sim_path'].values[i][startidx:,0], 
                    data['slam_sim_path'].values[i][startidx:,1],
                    '--', color=utils.blues[0],label='SLAM', linewidth=1.5)
        axs[j].set_xlim([-1.1,1.1])
        axs[j].set_ylim([-1.1,1.1])
        #axs[j].set_aspect('equal')
        if j != ncols*(nrows-1):
            axs[j].set_xticklabels([])
            axs[j].set_yticklabels([])
        if j == ncols-1:
            axs[j].legend(loc='lower right')#,frameon=True)
        # ax[j].set_ylabel('$y$')
        # ax[j].set_xlabel('$x$')
        axs[j].spines['right'].set_visible(True)
        axs[j].spines['top'].set_visible(True)
    
    fig.subplots_adjust(wspace=0, hspace=0)
    
    if savename is not None:
        utils.save(fig, savename)
    plt.show(fig)
        
folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/var-dim-sensitivity/'
funcs = ["styblinski-tang"]
func=funcs[0]
agts=[  "ssp-rand","gp-matern" ] #"ssp-hex" ,
labels=[ 'SSP-BO-Rand', 'GP-MI-Matern'] #'SSP-BO-Hex',
dims = np.arange(1,7,1)
cols = [utils.blues[0], utils.oranges[0], utils.greens[0],  utils.reds[0]]
linestys = ['-','--',':','-.']

# 7.1413in 3.48761in


fig, ax = plt.subplots(1,1, figsize=(3.45,2.5))


means = []
for j,agt in enumerate(agts):
    regret_vs_dim = []
    for i,dim in enumerate(dims):
        data = pd.DataFrame(read(folder + str(dim) + '/' + func + str(dim) + '/' + agt))
        regrets= np.array([data['regret'][i] for i in range(len(data['regret']))])
        budget=regrets.shape[1]
        num_trials=regrets.shape[0]
        regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1,budget+1), num_trials,1))[:,-1]
        regret_vs_dim.append(regrets)
    plt_data = get_mean_and_ci( np.array(regret_vs_dim).T)
    means.append(plt_data["mean"])
    ax.fill_between(dims, plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[j])
    ax.plot(dims, plt_data["mean"], color=cols[j], label=labels[j], linestyle=linestys[j])
    

ax.set_title('Terminal Average Regret vs Varaible Dimension\n (' + funcs[0].title() + ', ' + str(num_trials) +' trials)')
 
ax.set_xlabel("Varaible Dimension ($n$)")
ax.set_ylabel("Terminal Regret (a.u.)")
ax.legend()


utils.save(fig, 'test-func_var_dims.pdf')


# from sklearn.linear_model import LinearRegression
# # Create linear regression object
# regr1 = LinearRegression()
# regr2 = LinearRegression()
# # Train the model 
# regr1.fit( np.array(dims).reshape(-1, 1), np.array(means[0]).reshape(-1, 1))
# regr2.fit( np.array(dims).reshape(-1, 1),  np.array(means[1]).reshape(-1, 1))
# print(regr1.coef_)
# print(regr2.coef_)

