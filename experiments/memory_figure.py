import numpy as np
import glob
import os.path
import matplotlib.pyplot as plt
import matplotlib as mpl
import figure_utils as utils
# plt.style.use('plot_style.txt')

y_min = -0.1
y_max = 1.5

# psr_folder ='/run/media/furlong/Data/bo-ssp/psr_blah/psr_timing'
# psr_folder =  os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/experiments/timing_data/psr/'
# # box_folder
# box_folder =  os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/experiments/timing_data/box/'
# # box_folder
# belt_folder =  os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/experiments/timing_data/belt/'
# 


folder = '/run/media/furlong/Data/ssp-bayesopt/memory-test/'

funcs = ["himmelblau" , "goldstein-price","branin-hoo"]
fig_let = ['A','B','C']
let_pos = [0, 0.32, 0.635]
agts=[ "gp-matern", "gp-sinc", "ssp-hex","ssp-rand"]
# labels=['SSP-BO-Hex','SSP-BO-Rand','GP-MI-sinc', 'GP-MI-Matern']
labels=['GP-MI-Matern', 'GP-MI-Sinc', 'SSP-BO-Hex','SSP-BO-Rand']
cols = [utils.reds[0], utils.greens[0], utils.blues[0], utils.oranges[0]]
linestys = ['-.',':','-','--']



def get_memory_data(filenames):
    data = []
    for f in filenames: 
        mem = np.load(f)['memory'][:,0]
        data.append( mem - mem[0] )
    return np.vstack(data) 

def get_stats(data):
    mean = np.mean(data, axis=0)
    ste = np.std(data, axis=0) / np.sqrt(data.shape[0])
    mean_sum = np.mean(np.sum(data, axis=1))
    ste_sum = np.std(np.sum(data, axis=1)) / np.sqrt(data.shape[0])
    return mean, ste, mean_sum, ste_sum

plt.figure(figsize=(6.5,2.75))
for f_idx, mem_func in enumerate(funcs):
    memory_data = []
    memory_mus = []
    memory_stes = []


    plt.subplot(1,len(funcs),1+f_idx)
    for j,agt in enumerate(agts):
        memory_data_agt = get_memory_data(glob.glob(f'{folder}/{mem_func}/{agt}/*npz'))
        memory_data.append(np.copy(memory_data_agt))
        agt_mu, agt_ste, agt_sum, agt_sum_ste = get_stats(memory_data_agt)

        steps = np.arange(1,agt_mu.shape[0]+1)
        plt.plot(steps, agt_mu, label=labels[j], ls=linestys[j], color=cols[j], lw=2)
        plt.fill_between(steps, agt_mu - 1.96 * agt_ste,
                 agt_mu + 1.96 * agt_ste, alpha=0.4, color=cols[j])

        if f_idx == 0:
            plt.ylabel(r'$\Delta$ Memory (MB)')
            plt.legend()
        elif f_idx == 2:
            plt.gca().annotate(r'GP-BO', 
             xy=(175,0.3), xycoords='data',
             xytext=(20,0.5), textcoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3', shrinkA=0.05))
            plt.gca().annotate(r'SSP-BO', 
             xy=(175,0.05), xycoords='data',
             xytext=(20,0.2), textcoords='data',
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3, rad=0.3', shrinkA=0.05))
        plt.gcf().text(0.06 + let_pos[f_idx], 0.95, f'\\textbf{{ {fig_let[f_idx]} \; \; }}', size=11, va="baseline", ha="left")
        plt.xlabel('Sample number ($n$)')
    ### end for
    plt.title(mem_func.title())
    plt.tight_layout()

# plt.gca().spines['left'].set_position(('outward', 10))
# plt.gca().spines['bottom'].set_position(('outward', 10))
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().yaxis.set_ticks_position('left')
# plt.gca().xaxis.set_ticks_position('bottom')
# plt.tight_layout()
# plt.subplots_adjust(hspace=0.4)
# plt.tight_layout()


# plt.gcf().align_ylabels()
# plt.show()
# plt.savefig('ssp_multi_memory_plot.pdf', dpi=600)
utils.save(plt.gcf(), 'gp_memory_usage.pdf')

# if display:
# else:
#     plt.savefig('ssp_multi_timing_plot.pgf')

