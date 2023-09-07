import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import figure_utils as utils
import numpy.matlib as matlib

from matplotlib.markers import TICKDOWN

from scipy.stats import sem
import pandas as pd
import os, zipfile
from scipy.stats import wilcoxon

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

def significance_bar(ax, start,end,height,displaystring,
                     linewidth = 1.1,
                     markersize = 6,
                     boxpad  =0.2,fontsize = 11,color = 'k'):
    # draw a line with downticks at the ends
    ax.plot([start,end],[height]*2,'-',color =color,
             lw=linewidth,
             marker = TICKDOWN,
             markeredgewidth=linewidth,
             markersize = markersize)
    # draw the text with a bounding box covering up the line
    ax.text(0.5*(start+end),height,displaystring,
             ha = 'center',va='center',
             bbox=dict(facecolor='1.', edgecolor='none',
                       boxstyle='Square,pad='+str(boxpad)),
             size = fontsize)

    

cols = [utils.blues[0], utils.oranges[0], utils.greens[0],  utils.reds[0]]
funcs = ["himmelblau" , "goldstein-price","branin-hoo"]
agts=[ "ssp-hex","ssp-rand" ,"gp-sinc", "gp-matern"]
labels=['SSP-BO-Hex','SSP-BO-Rand','GP-MI-Sinc', 'GP-MI-Matern']

def do_plot(ax,regrets):
    plots = ax.violinplot(regrets)
    # Set the color of the violin patches
    for pc, color in zip(plots['bodies'], cols):
        pc.set_facecolor(color)
    
    # Set the color of the median lines
    plots['cbars'].set_colors(cols)
    plots['cmaxes'].set_colors(cols)
    plots['cmins'].set_colors(cols)
#         plt.bar(range(len(labels)), mus, yerr=sems)#, tick_label=labels)
#         plt.gca().set_xticklabels(labels, rotation=75)
    # plt.gca().spines['left'].set_position(('outward', 10))
    # plt.gca().spines['bottom'].set_position(('outward', 10))

    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False) 
    return np.max(regrets,axis=0)


starsym = '$\\ast$'
folder = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/data/test-funcs/'
fig, axs = plt.subplots(1,3, figsize=(7.1,3.3))
axs = axs.reshape(-1)
for i,func in enumerate(funcs):
    all_regrets = []
    axs[i].set_title(funcs[i].title())
    for j,agt in enumerate(agts):
        data = pd.DataFrame(read(folder + func + '/' + agt))
        regrets= np.array([data['regret'][k] for k in range(len(data['regret']))])
        budget=regrets.shape[1]
        num_trials=regrets.shape[0]
        regrets = np.divide(np.cumsum(regrets, axis=1), matlib.repmat(range(1,budget+1), num_trials,1))[:,-1]
        all_regrets.append(regrets)
    max_vals = do_plot(axs[i], np.array(all_regrets).T)
    axs[i].set_xticks(range(1,len(labels)+1), labels, rotation=45, fontsize=9)
    for j in range(1,len(agts)):
        mval = np.max([max_vals[0],max_vals[j]])
        pvalue = wilcoxon(all_regrets[0],all_regrets[j], alternative='less').pvalue
        starstr = starsym*3 if pvalue<0.0001 else starsym*2 if pvalue<0.001  else starsym if pvalue<0.05 else ''
        if pvalue > 0.05:
            break
        significance_bar(axs[i], 1,j+1,1.1*mval, starstr)
    if func=='branin-hoo':
        mvals = np.max(max_vals) + np.array([0, 7, 4, 0])
        for j in np.arange(len(agts)-2,0,-1):
            mval = mvals[j]
            pvalue = wilcoxon(all_regrets[-1],all_regrets[j], alternative='less').pvalue
            starstr = starsym*3 if pvalue<0.0001 else starsym*2 if pvalue<0.001  else starsym if pvalue<0.05 else ''
            if pvalue > 0.05:
                break
            significance_bar(axs[i], len(agts),j+1,mval, starstr)

#     plt.title(f'Average Regret: {func_name.title()}', fontsize=24)
axs[0].set_ylabel('Terminal Regret (a.u.)')
fig.tight_layout()
utils.save(fig, 'test-func-terminal.pdf')


    

#     plt.ylabel('Terminal Regret (a.u.)', fontsize=24)
#     plt.title('Himmelblau',fontsize=16)

#     plt.subplot(1,3,2) # Branin-Hoo

#     branin_mus = [8.22,14.13,11.48,8.30]
#     branin_sems = [0.32,1.72,1.25,0.51]

# #     do_plot(tick_labels, branin_mus, branin_sems)
#     max_vals = do_plot(tick_labels, branin_files)

# #     significance_bar(1,2,np.max(max_vals)+10, '****')
# #     significance_bar(1,3,np.max(max_vals)+14,'*')
# # #     significance_bar(1,4,np.max(max_vals[[0,3]])+0.3,'****')
# #     significance_bar(2,3,np.max(max_vals)+3,'*')
# #     significance_bar(2,4,np.max(max_vals)+6,'****')

#     plt.xticks(range(1,len(tick_labels)+1), tick_labels, rotation=45, fontsize=16)
#     plt.title('Branin-Hoo',fontsize=16)

#     plt.subplot(1,3,3) # Goldstein-Price

#     goldstein_mus = [0.11,0.11,0.09,0.07]
#     goldstein_sems = [0.01,0.02,0.01,0.01]
# #     do_plot(tick_labels, goldstein_mus, goldstein_sems)
#     max_vals = do_plot(tick_labels, goldstein_files)

# #     significance_bar(1,2,np.max(max_vals)+0.15,'****')
#     significance_bar(1,3,np.max(max_vals)+0.15,'***')
#     significance_bar(1,4,np.max(max_vals)+0.2,'****')
#     significance_bar(2,3,np.max(max_vals)+0.05,'*')
#     significance_bar(2,4,np.max(max_vals)+0.1,'****')

#     plt.xticks(range(1,len(tick_labels)+1), tick_labels, rotation=45, fontsize=16)
#     plt.title('Goldstein-Price',fontsize=16)

#     if save:
#         plt.savefig(f'terminal-regret.{plot_filetype}')
#     else:
#         plt.show()
