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

# 7.1413in 3.48761in


folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/test-funcs/'
funcs = ["himmelblau" , "goldstein-price","branin-hoo"]
agts=[ "ssp-hex","ssp-rand" ,"gp-sinc", "gp-matern"]
labels=['SSP-BO-Hex','SSP-BO-Rand','GP-MI-sinc', 'GP-MI-Matern']
cols = [utils.blues[0], utils.oranges[0], utils.greens[0],  utils.reds[0]]
linestys = ['-','--',':','-.']

t_plt_datas = []
plt_datas = []
for i,func in enumerate(funcs):
    plt_datass = []
    for j,agt in enumerate(agts):
        data = pd.DataFrame(read(folder + func + '/' + agt))
        regrets= np.array([data['regret'][k] for k in range(len(data['regret']))])
        budget=regrets.shape[1]
        num_trials=regrets.shape[0]
        regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1,budget+1), num_trials,1))
        plt_datass.append( get_mean_and_ci(regrets).copy() )
    plt_datas.append(plt_datass)
        
for j,agt in enumerate(agts):
    data = pd.DataFrame(read(folder + funcs[-1] + '/' + agt))
    times= np.array([data['times'][k] for k in range(len(data['times']))])* 1e-9
    t_plt_datas.append(get_mean_and_ci( times )  )   

fig, axs = plt.subplots(2,2, figsize=(7.1,4))
axs = axs.reshape(-1)
for i,func in enumerate(funcs):
    
    axs[i].set_title('Average Reget: ' + func.title())
    
    for j,agt in enumerate(agts):
        data = pd.DataFrame(read(folder + func + '/' + agt))
        regrets= np.array([data['regret'][k] for k in range(len(data['regret']))])
        budget=regrets.shape[1]
        plt_data = plt_datas[i][j].copy()
        axs[i].fill_between(np.arange(1,budget+1), plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[j])
        axs[i].plot(np.arange(1,budget+1), plt_data["mean"], color=cols[j], label=labels[j], linestyle=linestys[j])
        
    axs[i].set_xlabel("Sample Number ($n$)")
    axs[i].set_ylabel("Average Regret")
axs[1].legend()
    

axs[-1].set_title('Sample Selection Time: ' + funcs[i].title())
for j,agt in enumerate(agts):
    data = pd.DataFrame(read(folder + func + '/' + agt))
    times= np.array([data['times'][k] for k in range(len(data['times']))])* 1e-9
    plt_data = t_plt_datas[j]
    axs[-1].fill_between(np.arange(times.shape[1]), plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[j])
    axs[-1].plot(np.arange(times.shape[1]), plt_data["mean"], color=cols[j], label=labels[j], linestyle=linestys[j])
axs[-1].set_xlabel("Sample Number ($n$)")
axs[-1].set_ylabel("Sample Selection Time (sec)")
fig.tight_layout()


fig.text(0.03, 0.95, '\\textbf{A}', size=12, va="baseline", ha="left")
fig.text(0.5,0.95, '\\textbf{B}', size=12, va="baseline", ha="left")
fig.text(0.03,0.49, '\\textbf{C}', size=12, va="baseline", ha="left")
fig.text(0.5,0.49, '\\textbf{D}', size=12, va="baseline", ha="left")

utils.save(fig, 'test-func_regret.pdf')


