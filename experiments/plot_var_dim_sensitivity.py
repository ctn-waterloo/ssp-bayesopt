import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
import figure_utils as utils
import numpy.matlib as matlib
import matplotlib.gridspec as gridspec
import re
import matplotlib.tri as tri

from matplotlib.ticker import MultipleLocator
from scipy.stats import gaussian_kde
from matplotlib import rc
import sys
import zipfile
import pandas as pd
import os
from matplotlib import cm


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
    'styblinski-tang3': 39.16599*3,
    'styblinski-tang4': 39.16599*4,
    'styblinski-tang5': 39.16599*5,
    'styblinski-tang6': 39.16599*6,
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

# rc('font',size=7)
# rc('font',family='serif')
# rc('axes',labelsize=7)


        
folder = os.path.join(os.getcwd(),  'data/var-dim')
func = "styblinski-tang"
agts=[ "ssp-hex", "ssp-rand","gp-matern", "gp-sinc","rff" ] #"ssp-hex" ,
embed_agts=[a for a in agts if 'gp' not in a]
n_embed_agt = len([a for a in agts if 'gp' not in a])
labels ={"ssp-hex": 'SSP-BO-Hex', "ssp-rand" : 'SSP-BO-Rand',
         "gp-sinc": 'GP-sinc', "gp-matern": 'GP-Matern',
         "rff": 'RFF-BO',
         "ssp-hex_nengo-loihi-sim": "SSP-BO-Hex (Loihi 1$^*$)",
         "ssp-hex_nengo-spinnaker": "SSP-BO-Hex (Spinnaker)",
         "ssp-hex_nengo-loihi": "SSP-BO-Hex (Loihi 2)"}
cols ={"ssp-hex": utils.blues[0], "ssp-rand" : utils.oranges[0],
         "gp-sinc": utils.greens[0], "gp-matern": utils.reds[0],
         "rff": utils.yellows[2],
         "ssp-hex_nengo-loihi-sim": utils.blues[1],
         "ssp-hex_nengo-spinnaker": utils.blues[2],
         "ssp-hex_nengo-loihi": utils.blues[1]}

linestys = {"ssp-hex": '-', "ssp-rand": '--',
         "gp-sinc": ':', "gp-matern": '-.',
         "rff": (0, (3, 1, 1, 1, 1, 1)),
         "ssp-hex_nengo-loihi-sim": '-.',
         "ssp-hex_nengo-spinnaker": ':',
         "ssp-hex_nengo-loihi": '--'}


pattern = re.compile(f'{func}' + r'\D*(\d+)')
var_dims = list(
    map(
        lambda filename: int(pattern.search(filename).group(1)),
        filter(lambda filename: pattern.search(filename) is not None, os.listdir(folder))
    )
)
var_dims = np.sort(var_dims)

pattern = re.compile(r'sspdim\D*(\d+)')
ssp_dims = list(
    map(
        lambda filename: int(pattern.search(filename).group(1)),
        filter(lambda filename: pattern.search(filename) is not None, os.listdir(folder))
    )
)
ssp_dims = np.sort(ssp_dims)
ssp_dim_to_var_dict = {}
for agt in embed_agts:
    ssp_dim_to_var_dict[agt] = {}
    for sspdim in ssp_dims:
        pattern = re.compile(f'{func}' + r'\D*(\d+)')
        dim_list= list(
                map(
                    lambda filename: int(pattern.search(filename).group(1)),
                    filter(lambda filename: pattern.search(filename) is not None, os.listdir(os.path.join(folder, f'sspdim{sspdim}')))
                )
            )
        dim_list = [d for d in dim_list if agt in os.listdir(os.path.join(folder, f'sspdim{sspdim}', f'{func}{d}'))]
        if len(dim_list)>0:
            ssp_dim_to_var_dict[agt][sspdim] = dim_list

var_dim_to_ssp_dict = {}
for agt in embed_agts:
    var_dim_to_ssp_dict[agt] = {}
    for key, values in ssp_dim_to_var_dict[agt].items():
        for value in values:
            if value not in var_dim_to_ssp_dict[agt]:
                var_dim_to_ssp_dict[agt][value] = []
            var_dim_to_ssp_dict[agt][value].append(key)

fixed_ssp_dim = 245
fixed_id_dict = {}
for agt in embed_agts:
    fixed_id_dict[agt] = {}
    for dim in var_dims:
        fixed_id_dict[agt][dim] = var_dim_to_ssp_dict[agt][dim][np.argmin(np.abs(np.array(var_dim_to_ssp_dict[agt][dim]) -fixed_ssp_dim))]

max_num_trials = 7 ###
plt_data = {}
for j,agt in enumerate(agts):
    plt_data[agt] = []

    for i,dim in enumerate(var_dims):
        if 'gp' not in agt:
            plt_data[agt + str(dim)] = []

        func_name = func + str(dim)
        true_max_val = function_maximum_value[func_name]
        if 'gp' in agt:
            data = pd.DataFrame(read(f'{folder}/{func_name}/{agt}'))
            if 'regret' in data.keys():
                regrets = np.array([data['regret'][k] for k in range(len(data['regret'])) ])
            elif 'vals' in data.keys():
                vals = np.array([data['vals'][k] for k in range(len(data['vals'])) ])
                regrets = true_max_val - vals

            budget = regrets.shape[1]
            num_trials = regrets.shape[0]
            regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1, budget + 1), num_trials, 1))[:, -1]
            plt_data[agt].append(regrets.copy())
        else:
            for sspdim in var_dim_to_ssp_dict[agt][dim]:
                data = pd.DataFrame(read(f'{folder}/sspdim{sspdim}/{func_name}/{agt}'))

                if 'regret' in data.keys():
                    regrets = np.array([data['regret'][k] for k in range(len(data['regret']))])
                elif 'vals' in data.keys():
                    vals = np.array([data['vals'][k] for k in range(len(data['vals']))])
                    regrets = true_max_val - vals
                budget = regrets.shape[1]
                regrets = regrets[:max_num_trials,:]
                num_trials = regrets.shape[0]

                regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1, budget + 1), num_trials, 1))[:,-1]
                plt_data[agt + str(dim)].append(regrets.copy())
                if sspdim==fixed_id_dict[agt][dim]:
                    plt_data[agt].append(regrets.copy())
        if 'gp' not in agt:
            plt_data[agt + str(dim)] = np.mean(np.array(plt_data[agt + str(dim)]).T, axis=0)

    plt_data[agt] = get_mean_and_ci(np.array(plt_data[agt]).T)


vmin = 100000
vmax=-10000
for agt in agts:
    if 'gp' not in agt:
        for dim in var_dims:
            vmin = np.minimum(vmin, np.min(plt_data[agt + str(dim)]))
            vmax = np.maximum(vmax, np.max(plt_data[agt + str(dim)]))

letters = ["\\textbf{a}", "\\textbf{b}", "\\textbf{c}", "\\textbf{d}"]
letters = [l + " $\\quad$ " for l in letters]
fontsize = 9
fig = plt.figure(figsize=(7.5, 3.3))
gs = gridspec.GridSpec(1, 2, figure=fig, hspace=0.7, wspace=0.3, width_ratios=[1, 1])
gs0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
gs1 = gridspec.GridSpecFromSubplotSpec(n_embed_agt, 1, subplot_spec=gs[1], hspace=0.9)
axs1 = [fig.add_subplot(gs1[i]) for i in range(n_embed_agt)]  # , fig.add_subplot(gs1[2])]

ax0 = fig.add_subplot(gs0[0])
j = 0
ax0.set_title(f'{letters[0]}  Terminal Average Regret vs Data Size for {func.title()}'  )
for agt in agts:
    ax0.fill_between(var_dims, plt_data[agt]["upper_bound"], plt_data[agt]["lower_bound"], alpha=.2, color=cols[agt])
    ax0.plot(var_dims, plt_data[agt]["mean"], color=cols[agt], label=labels[agt], linestyle=linestys[agt])
    if 'gp' not in agt:
        axs1[j].set_title(labels[agt])

        plt_X=[]
        plt_Y=[]
        plt_Z=[]
        for dim in var_dims:
            for k, sspdim in enumerate(var_dim_to_ssp_dict[agt][dim]):
                plt_X.append(dim)
                plt_Y.append(sspdim)
                plt_Z.append(plt_data[agt + str(dim)][k])
        plt_X = np.array(plt_X)
        plt_Y = np.array(plt_Y)
        plt_Z = np.array(plt_Z)

        yi = np.linspace(np.min(plt_Y), np.max(plt_Y), 10, endpoint=True)

        # Linearly interpolate the data (x, y) on a grid defined by (xi, yi).
        triang = tri.Triangulation(plt_X, plt_Y)
        interpolator = tri.LinearTriInterpolator(triang, plt_Z)
        Xi, Yi = np.meshgrid(var_dims, yi)
        zi = interpolator(Xi, Yi)

        sc = axs1[j].pcolormesh(Xi,Yi,zi, shading='nearest',
                                cmap='viridis', vmin=vmin, vmax=vmax)
        # for dim in var_dims:
            # sc = axs1[j].scatter(dim * np.ones(len(var_dim_to_ssp_dict[agt][dim])), var_dim_to_ssp_dict[agt][dim],
            #                 c=plt_data[agt + str(dim)], cmap='viridis', vmin=vmin, vmax=vmax)

        if j==1:
            axs1[j].set_ylabel("Embedding size ($d$)",fontsize=fontsize)

        axs1[j].tick_params(axis='both', which='major', labelsize=fontsize)
        axs1[j].set_title(f'{letters[j+1]} {labels[agt]}')
        j += 1
axs1[-1].set_xlabel("Data Size ($n$)",fontsize=fontsize)
ax0.tick_params(axis='both', which='major', labelsize=fontsize)

cbar=fig.colorbar(sc, ax=axs1, shrink=0.6)
cbar.set_label('Terminal Regret', rotation=270,fontsize=fontsize, labelpad=15)
# ax.set_title('Terminal Average Regret vs Varaible Dimension\n (' + funcs[0].title() + ', ' + str(num_trials) +' trials)')
ax0.set_xlabel("Data Size ($n$)",fontsize=fontsize)
ax0.set_ylabel("$\\leftarrow$ Terminal Regret",fontsize=fontsize)
ax0.legend()


utils.save(fig, 'test-func_var_dims3.pdf')
fig

# from sklearn.linear_model import LinearRegression
# # Create linear regression object
# regr1 = LinearRegression()
# regr2 = LinearRegression()
# # Train the model 
# regr1.fit( np.array(dims).reshape(-1, 1), np.array(means[0]).reshape(-1, 1))
# regr2.fit( np.array(dims).reshape(-1, 1),  np.array(means[1]).reshape(-1, 1))
# print(regr1.coef_)
# print(regr2.coef_)

# import numpy as np
# dim=6
# sspdim=101
# nscales=3
# nrotates = int(np.round((sspdim - 1) / (2 * nscales * (dim + 1))))
# realsspdim=int(np.round(1 + (2 * nscales * nrotates * (dim + 1))))
# print(realsspdim)