import numpy as np
import functions


from ssp_bayes_opt.agent import SSPAgent

# func, bounds, T = functions.factory('branin-hoo')
func, bounds, T = functions.factory('himmelblau')


a = np.linspace(bounds[0,0], bounds[0,1], 100)
b = np.linspace(bounds[1,0], bounds[1,1], 100)

X, Y = np.meshgrid(a,b)

init_xs =  np.vstack((X.flatten(), Y.flatten())).T
# init_ys = func(init_xs[:,0], x1=init_xs[:,1])
init_ys = func(init_xs)

import matplotlib.pyplot as plt

agt = SSPAgent(init_xs, init_ys.reshape((-1,1)))

mus, var_s, phi_s = agt.eval(init_xs)


plt.subplot(1,2,1)
plt.matshow(init_ys.reshape(X.shape), fignum=False)
# plt.xticks(ticks=a)
# plt.yticks(ticks=b)
plt.title('True')

plt.subplot(1,2,2)
plt.matshow(mus.reshape(X.shape), fignum=False)
# plt.xticks(ticks=a)
# plt.yticks(ticks=b)
plt.title('Est')

print(np.mean((init_ys.flatten() - mus.flatten())**2))
plt.show()


# agent = SSPAgent(init_xs, init_ys)
