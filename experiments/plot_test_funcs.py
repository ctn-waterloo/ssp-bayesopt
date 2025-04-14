import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import figure_utils as utils
import numpy.matlib as matlib
from matplotlib.gridspec import GridSpec
import sys,os
# sys.path.insert(1, os.path.dirname(os.getcwd()))
# os.chdir("..")
from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib import rc
import sys
import zipfile
import pandas as pd
import os
from matplotlib import cm
import pickle
import re


function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133,
    'styblinski-tang1': 39.16599,
    'styblinski-tang2': 39.16599*2,
    'styblinski-tang3': 39.16599*3 #1 dim ****
}

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

def read_nengo(folder_path,function_name="",agent_type=""):
    filename_pattern = re.compile(r'sspd97_N8_tau0\.05_seed(\d+)\.pkl')

    # List to collect data
    data = []

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        match = filename_pattern.match(filename)
        if match:
            seed = int(match.group(1))
            file_path = os.path.join(folder_path, filename)

            # Load the pickle file
            with open(file_path, 'rb') as f:
                loaded_data = pickle.load(f)
                # regrets = loaded_data['regrets']
                regrets = function_maximum_value[func] - loaded_data['f_vals']
                times = loaded_data['times']

            # Append data to the list
            data.append({
                'seed': seed,
                'function_name': function_name,
                'agent_type': agent_type,
                'regret': regrets,
                'times': times
            })
    return data

# rc('font',size=7)
# rc('font',family='serif')
# rc('axes',labelsize=7)

# 7.1413in 3.48761in
function_maximum_value = {
    'himmelblau':0.07076226300682818, # Determined from offline minimization of modified himmelblau.
    'branin-hoo': -0.397887, # Because the function is negated to make it a maximization problem.
    'goldstein-price': -3/1e5, # Because the true function is scaled and negated.
    'colville': 0,
    'rastrigin': 0,
    'ackley': 0,
    'rosenbrock': 0,
    'beale': 0,
    'easom': 1,
    'mccormick': 1.9133
}

# folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/test-funcs/'
funcs = ["himmelblau" , "branin-hoo", "goldstein-price",]
#
folders = [os.path.join(os.getcwd(), "data/new_ucb_beta0.1")]
# agts=[ "ssp-hex","ssp-rand" ,"gp-sinc", "gp-matern"]
# labels=['SSP-BO-Hex','SSP-BO-Rand','GP-MI-sinc', 'GP-MI-Matern']
# folder = '/run/media/furlong/Data/ssp-bayesopt/memory-test/'
agts=[ "ssp-hex","ssp-rand" ,"gp-sinc", "gp-matern", "rff"]#,
    # "ssp-hex_nengo-loihi-sim"]
     #  ]
labels=['SSP-BO-Hex','SSP-BO-Rand','GP-sinc', 'GP-Matern', 'RFF-BO', 'SSP-BO-Hex (Loihi)']
cols = [utils.blues[0], utils.oranges[0], utils.greens[0],  utils.reds[0], utils.blues[2]]
linestys = ['-','--',':','-.', (0, (3, 1, 1, 1, 1, 1))]

max_num_trials = 30

t_plt_datas = []
plt_datas = []
for i,func in enumerate(funcs):
    plt_datass = []

    if "rastrigin" in func:
        true_max_val = 0
    elif "rosenbrock " in func:
        true_max_val = 0
    else:
        true_max_val = function_maximum_value[func]


    for j,agt in enumerate(agts):
        if len(folders)==1:
            folder = folders[0]
        else:
            folder = folders[j]
        data = pd.DataFrame(read(os.path.join(folder,func,agt)))
        if 'regret' in data.keys():
            regrets = np.array([data['regret'][k] for k in range(len(data['regret']))])
        elif 'vals' in data.keys():
            vals = np.array([data['vals'][k] for k in range(len(data['vals']))])
            regrets=true_max_val - vals
        else:
            print(data.keys())
        if regrets.shape[0]>max_num_trials:
            regrets = regrets[:max_num_trials,:]

        budget=regrets.shape[1]
        num_trials=regrets.shape[0]
        regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1,budget+1), num_trials,1))
        plt_datass.append( get_mean_and_ci(regrets).copy() )
    plt_datas.append(plt_datass)

timess=[]
time_func = "branin-hoo"        
for j,agt in enumerate(agts):
    if len(folders) == 1:
        folder = folders[0]
    else:
        folder = folders[j]

    data = pd.DataFrame(read(os.path.join(folder,func,agt)))
    # times= np.array([data['times'][k] for k in range(len(data['times']))])* 1e-9
    times = np.array([data['full_times'][k] for k in range(len(data['full_times']))]) * 1e-9
    if times.shape[0] > max_num_trials:
        times = times[:max_num_trials, :]

    timess.append(times)
    t_plt_datas.append(get_mean_and_ci( times )  )   

letters = ["\\textbf{a}", "\\textbf{b}","\\textbf{c}","\\textbf{d}"]
letters = [l + " $\\quad$ " for l in letters]
fontsize=9
fig = plt.figure(figsize=(7.5,4))
gs = GridSpec(2, 2, figure=fig, hspace=0.7, wspace=0.3)
axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
for i,func in enumerate(funcs):
    
    axs[i].set_title(letters[i] + 'Average Regret: ' + func.title(), fontsize=fontsize)
    
    for j,agt in enumerate(agts):
        budget = len(plt_datas[i][j]["mean"])
        plt_data = plt_datas[i][j].copy()
        axs[i].fill_between(np.arange(1,budget+1), plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[j])
        axs[i].plot(np.arange(1,budget+1), plt_data["mean"], color=cols[j], label=labels[j], linestyle=linestys[j])
        
    axs[i].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
    axs[i].set_ylabel("Average Regret", fontsize=fontsize)
    axs[i].tick_params(axis='both', which='major', labelsize=fontsize)
axs[1].legend(fontsize=fontsize)
    
axs[-1].set_title(letters[-1] + 'Sample Selection Time: ' + time_func.title(), fontsize=fontsize)
for j,agt in enumerate(agts):
    if True:#"ssp" not in agt:
        # times = timess[j][:,10:]
        plt_data = t_plt_datas[j]
        times = np.arange(1,len(plt_data["mean"])+1)
        axs[-1].fill_between(times, plt_data["upper_bound"], plt_data["lower_bound"], alpha=.2, color=cols[j])
        axs[-1].plot(times, plt_data["mean"], color=cols[j], label=labels[j], linestyle=linestys[j])
axs[-1].set_xlabel("Sample Number ($n$)", fontsize=fontsize)
axs[-1].set_ylabel("Sample Selection Time (sec)", fontsize=fontsize)
axs[-1].tick_params(axis='both', which='major', labelsize=fontsize)
# fig.tight_layout()
#
# fig.text(0.03, 0.95, '\\textbf{A}', size=12, va="baseline", ha="left")
# fig.text(0.5,0.95, '\\textbf{B}', size=12, va="baseline", ha="left")
# fig.text(0.03,0.49, '\\textbf{C}', size=12, va="baseline", ha="left")
# fig.text(0.5,0.49, '\\textbf{D}', size=12, va="baseline", ha="left")

utils.save(fig, 'test-func_regret-new.pdf')
utils.save(fig, 'test-func_regret-new.png')

# plt.show()
fig


