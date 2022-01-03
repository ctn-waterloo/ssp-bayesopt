import numpy as np
import functions


from ssp_bayes_opt import agent

# func, bounds, T = functions.factory('branin-hoo')
func, bounds, T = functions.factory('himmelblau')

num_samples=25

a = np.linspace(bounds[0,0], bounds[0,1], num_samples)
b = np.linspace(bounds[1,0], bounds[1,1], num_samples)

X, Y = np.meshgrid(a,b)

init_xs =  np.vstack((X.flatten(), Y.flatten())).T
# init_ys = func(init_xs[:,0], x1=init_xs[:,1])
init_ys = func(init_xs).reshape((-1,1))

import matplotlib.pyplot as plt

# agt = SSPAgent(init_xs, init_ys.reshape((-1,1)))
agt = agent.factory('ssp-mi', init_xs[:,:], np.atleast_2d(init_ys[:,0]).T)

mus, var_s, phi_s = agt.eval(init_xs)

acq_func, jac_func = agt.acquisition_func()

acq_vals = np.array([-acq_func(x) for x in init_xs])
jac_vals = np.array([-jac_func(x) for x in init_xs])

jac_mag = np.sqrt(np.sum(np.power(jac_vals, 2), axis=1))


plt.subplot(2,2,1)
plt.matshow(init_ys.reshape(X.shape), fignum=False)
# plt.xticks(ticks=a)
# plt.yticks(ticks=b)
plt.title('True')

plt.subplot(2,2,2)
plt.matshow(mus.reshape(X.shape), fignum=False)

plt.subplot(2,2,3)
plt.matshow(acq_vals.reshape(X.shape), fignum=False)

plt.subplot(2,2,4)

U = jac_vals[:,0].reshape(X.shape)
V = jac_vals[:,1].reshape(X.shape)

# plt.matshow((mus+phi_s).reshape(X.shape), fignum=False)
plt.quiver(U, V)#, fignum=False)
# plt.xticks(ticks=a)
# plt.yticks(ticks=b)
plt.title('Est')

print(np.mean((init_ys.flatten() - mus.flatten())**2))
plt.show()


# agent = SSPAgent(init_xs, init_ys)
