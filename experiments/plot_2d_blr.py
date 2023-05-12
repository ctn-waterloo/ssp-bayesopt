import matplotlib.pyplot as plt
import seaborn as sns
import time
import numpy as np
import functions
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import ssp_bayes_opt

import importlib
importlib.reload(ssp_bayes_opt)

# Apply the default theme
sns.set_theme()


target, pbounds, budget = functions.factory('himmelblau')
budget = 100

optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                               bounds=pbounds, 
                                               verbose=True,
                                               sampling_seed=0)
num_init_samples=20
start = time.thread_time_ns()
optimizer.maximize(init_points=num_init_samples, 
                   n_iter=budget,
                   num_restarts=1,
                   agent_type='ssp-hex',
                   ssp_dim=200,decoder_method='from-set')
elapsed_time = time.thread_time_ns() - start

vals = np.zeros((num_init_samples + budget,))
sample_locs = []

for i, res in enumerate(optimizer.res):
    vals[i] = res['target']
    sample_locs.append(res['params'])
    
samples_per_dim = 100
xs = optimizer.agt.ssp_space.get_sample_points(samples_per_dim=samples_per_dim, method='grid')
phis = optimizer.agt.encode(xs)
mu, var = optimizer.agt.blr.predict(phis)

xsamples =np.array(sample_locs)
ysamples = np.array(vals)
  
fig = plt.figure(figsize=(13, 7))
ax = Axes3D(fig, computed_zorder=False)
#ax = plt.axes(projection='3d', computed_zorder=False)
#my_col = cm.inferno(var.reshape(samples_per_dim,samples_per_dim))
norm = mpl.colors.Normalize(vmin=np.min(np.sqrt(var)), vmax=np.max(np.sqrt(var)))
cmap = mpl.cm.summer
ax.plot_wireframe(xs[:,0].reshape(samples_per_dim,samples_per_dim), xs[:,1].reshape(samples_per_dim,samples_per_dim),
                  target(xs).reshape(samples_per_dim,samples_per_dim), 
                  rcount=5, ccount=5,color='black')
surf = ax.plot_surface(xs[:,0].reshape(samples_per_dim,samples_per_dim),
                       xs[:,1].reshape(samples_per_dim,samples_per_dim),
                       mu.reshape(samples_per_dim,samples_per_dim), 
                       rstride=1, cstride=1, facecolors = cmap(norm(np.sqrt(var).reshape(samples_per_dim,samples_per_dim))),
                       edgecolor='none', alpha=0.6)

cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
ax.scatter(xsamples[:,0], xsamples[:,1], ysamples, '.', s=10, color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
# add color bar indicating the PDF
ax.view_init(60, 35)


# plt.colorbar()
# ax.plot(m1, m2, 'k.', markersize=5)