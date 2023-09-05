import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from matplotlib.markers import TICKDOWN

from scipy.stats import sem

def significance_bar(start,end,height,displaystring,
                     linewidth = 1.2,
                     markersize = 8,
                     boxpad  =0.3,fontsize = 15,color = 'k'):
    # draw a line with downticks at the ends
    plt.plot([start,end],[height]*2,'-',color =color,
             lw=linewidth,
             marker = TICKDOWN,
             markeredgewidth=linewidth,
             markersize = markersize)
    # draw the text with a bounding box covering up the line
    plt.text(0.5*(start+end),height,displaystring,
             ha = 'center',va='center',
             bbox=dict(facecolor='1.', edgecolor='none',
                       boxstyle='Square,pad='+str(boxpad)),
             size = fontsize)



if __name__ == '__main__':
    save = not False
    plot_filetype = 'pgf'

    if save:
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

    plt.figure(figsize=(10,5))

    def do_plot(labels, regret_filenames):
        regrets = [np.load(f) for f in regret_filenames]
        plt.violinplot(regrets)
#         plt.bar(range(len(labels)), mus, yerr=sems)#, tick_label=labels)
#         plt.gca().set_xticklabels(labels, rotation=75)
        plt.gca().spines['left'].set_position(('outward', 10))
        plt.gca().spines['bottom'].set_position(('outward', 10))

        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False) 
        return np.array([np.max(r) for r in regrets])

#     plt.title(f'Average Regret: {func_name.title()}', fontsize=24)
    plt.tight_layout()

    tick_labels = ['GP-MI-Matern', 'GP-MI-Sinc', 'SSP-BO-Rand', 'SSP-BO-Hex']
    plt.subplot(1,3,1) # Himmelblau

    himmelblau_files = ['neurips_submission/himmelblau_GP-Matern_term-regret.npz.npy',
                        'neurips_submission/himmelblau_GP-Sinc_term-regret.npz.npy',
                        'neurips_submission/himmelblau_SSP-Rand_term-regret.npz.npy',
                        'neurips_submission/himmelblau_SSP-Hex_term-regret.npz.npy']
    branin_files = ['neurips_submission/branin-hoo_GP-Matern_term-regret.npz.npy',
                        'neurips_submission/branin-hoo_GP-Sinc_term-regret.npz.npy',
                        'neurips_submission/branin-hoo_SSP-Rand_term-regret.npz.npy',
                        'neurips_submission/branin-hoo_SSP-Hex_term-regret.npz.npy']
    goldstein_files = ['neurips_submission/goldstein-price_GP-Matern_term-regret.npz.npy',
                        'neurips_submission/goldstein-price_GP-Sinc_term-regret.npz.npy',
                        'neurips_submission/goldstein-price_SSP-Rand_term-regret.npz.npy',
                        'neurips_submission/goldstein-price_SSP-Hex_term-regret.npz.npy']




#     himmelblau_mus = [0.69, 0.41, 0.35,0.26]
#     himmelblau_sems = [0.08, 0.04, 0.02, 0.02]
    max_vals = do_plot(tick_labels, himmelblau_files)

    # GP-Matern vs SSP-Rand
    significance_bar(1,2,np.max(max_vals[[0,1]])+0.1,'****')
    significance_bar(1,3,np.max(max_vals[[0,2]])+0.3,'****')
    significance_bar(1,4,np.max(max_vals[[0,3]])+0.5,'****')
    significance_bar(2,3,np.max(max_vals[[1,2]])+0.1,'*')
    significance_bar(2,4,np.max(max_vals[[1,2]])+0.3,'****')

    plt.xticks(range(1,len(tick_labels)+1), tick_labels, rotation=45, fontsize=16)

    plt.ylabel('Terminal Regret (a.u.)', fontsize=24)
    plt.title('Himmelblau',fontsize=16)

    plt.subplot(1,3,2) # Branin-Hoo

    branin_mus = [8.22,14.13,11.48,8.30]
    branin_sems = [0.32,1.72,1.25,0.51]

#     do_plot(tick_labels, branin_mus, branin_sems)
    max_vals = do_plot(tick_labels, branin_files)

    significance_bar(1,2,np.max(max_vals)+10, '****')
    significance_bar(1,3,np.max(max_vals)+14,'*')
#     significance_bar(1,4,np.max(max_vals[[0,3]])+0.3,'****')
    significance_bar(2,3,np.max(max_vals)+3,'*')
    significance_bar(2,4,np.max(max_vals)+6,'****')

    plt.xticks(range(1,len(tick_labels)+1), tick_labels, rotation=45, fontsize=16)
    plt.title('Branin-Hoo',fontsize=16)

    plt.subplot(1,3,3) # Goldstein-Price

    goldstein_mus = [0.11,0.11,0.09,0.07]
    goldstein_sems = [0.01,0.02,0.01,0.01]
#     do_plot(tick_labels, goldstein_mus, goldstein_sems)
    max_vals = do_plot(tick_labels, goldstein_files)

#     significance_bar(1,2,np.max(max_vals)+0.15,'****')
    significance_bar(1,3,np.max(max_vals)+0.15,'***')
    significance_bar(1,4,np.max(max_vals)+0.2,'****')
    significance_bar(2,3,np.max(max_vals)+0.05,'*')
    significance_bar(2,4,np.max(max_vals)+0.1,'****')

    plt.xticks(range(1,len(tick_labels)+1), tick_labels, rotation=45, fontsize=16)
    plt.title('Goldstein-Price',fontsize=16)

    if save:
        plt.savefig(f'terminal-regret.{plot_filetype}')
    else:
        plt.show()
