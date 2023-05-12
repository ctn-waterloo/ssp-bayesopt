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



class ContexualFunction:

    def __init__(self):
        self.data_dim = 2
        self.context_size = 2
        self.dim = self.data_dim +1
        self.bounds = np.array([[-5, 5],[-5, 5], [0,self.context_size]]) #[-5, 10],[-5, 15]
        self.budget = 500

    def reset(self):
        return np.array([[0]])
    
    def _func1(self, x):
        return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100
    
    def _func2(self, x):
        #res= -20*np.exp(-0.2*np.sqrt(0.5*np.sqrt(x[:,0]**2 + x[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[:,0]) + np.cos(2*np.pi*x[:,1]))) + np.exp(1) + 20
        #res = np.cos(x[:,0])*np.cos(x[:,1])*np.exp(-(x[:,0]-np.pi)**2 - (x[:,1]-np.pi)**2 )
        return  - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100
    
    def __call__(self, xc, info):
        x = xc[:,:self.data_dim]
        context = xc[:,-1]
       # return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100, np.array([[-1]])


        if context == 0:
            return self._func1(x), np.array([[1]])
        elif context==1:
            return self._func2(x), np.array([[0]])
 
    def call_vectorized(self, xc, info):
        x = xc[:,:self.data_dim]
        context = xc[:,-1].reshape(-1)
        # return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100, np.array([[-1]])
        res = np.zeros((x.shape[0]))
        res_c = np.zeros((x.shape[0]))
        res[context==0] = self._func1(x[context==0,:])
        res_c[context==0] = 1
        
        res[context==1] = self._func2(x[context==1,:])
        res_c[context==1] = 0
        return res, res_c

       
    
    def _maximum_value(self, xs):
        res = np.zeros(len(xs))
        context1 = np.array([x[0] == 0 for x in xs])
        res[context1]  = 0.07076226300682818
        res[~context1]  = 1
        return res


target = ContexualFunction()
pbounds = target.bounds
budget = 200
#target, pbounds = functions.rescale(target,pbounds)

optimizer = ssp_bayes_opt.BayesianOptimization(f=target,
                                               bounds=pbounds, 
                                               verbose=True,
                                               sampling_seed=0)
num_init_samples=20
start = time.thread_time_ns()
optimizer.maximize(init_points=num_init_samples, 
                   n_iter=budget,
                   num_restarts=1,
                   agent_type='ssp-discrete-context',
                   ssp_dim=500,
                   decoder_method='from-set')


elapsed_time = time.thread_time_ns() - start

vals = np.zeros((num_init_samples + budget,))
sample_locs = []

for i, res in enumerate(optimizer.res):
    vals[i] = res['target']
    sample_locs.append(res['params'])
    
con1 = [l[-1] ==0 for l in sample_locs]
con2 = [l[-1] ==1 for l in sample_locs]
masks = [con1, con2]

_xsamples =np.array(sample_locs)
_ysamples = np.array(vals)

samples_per_dim = 100
_xs = optimizer.agt.ssp_space.get_sample_points(samples_per_dim=samples_per_dim, method='grid')
ctxts = [0,1]
fig = plt.figure(figsize=plt.figaspect(0.5))
for i, context in enumerate(ctxts):
    xs = np.hstack([_xs,context*np.ones((_xs.shape[0],1))])
    phis = optimizer.agt.encode(xs)
    mu, var = optimizer.agt.blr.predict(phis)
    
    xsamples =_xsamples[masks[i],:-1]
    ysamples = _ysamples[masks[i]]
      
    ax = fig.add_subplot(1, 2, i+1, projection='3d', computed_zorder=False)

    res, _ = target.call_vectorized(xs, None)
    res = res.reshape(samples_per_dim,samples_per_dim)

    norm = mpl.colors.Normalize(vmin=np.min(np.sqrt(var)), vmax=np.max(np.sqrt(var)))
    cmap = mpl.cm.summer
    ax.plot_wireframe(xs[:,0].reshape(samples_per_dim,samples_per_dim), xs[:,1].reshape(samples_per_dim,samples_per_dim),
                      res, 
                      rcount=5, ccount=5,color='black')
    surf = ax.plot_surface(xs[:,0].reshape(samples_per_dim,samples_per_dim),
                           xs[:,1].reshape(samples_per_dim,samples_per_dim),
                           mu.reshape(samples_per_dim,samples_per_dim), 
                           rstride=1, cstride=1, facecolors = cmap(norm(np.sqrt(var.reshape(samples_per_dim,samples_per_dim)))),
                           edgecolor='none', alpha=0.6)
    
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    ax.scatter(xsamples[:,0], xsamples[:,1], ysamples, '.', s=10, color='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    # add color bar indicating the PDF
    ax.view_init(60, 35)
    
