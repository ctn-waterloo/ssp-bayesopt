import numpy as np
import functions


from ssp_bayes_opt import agent


def plot(test_ys, mus, acq_vals, jac_vals, X,Y, x_t):

    X_shape = X.shape

    plt.subplot(2,2,1)
    plt.matshow(test_ys.reshape(X_shape), fignum=False, origin='lower')
    # plt.xticks(ticks=a)
    # plt.yticks(ticks=b)
    plt.title('True')

    plt.subplot(2,2,2)
    plt.matshow(mus.reshape(X_shape), fignum=False, origin='lower')

    plt.subplot(2,2,3)
    plt.matshow(acq_vals.reshape(X_shape), fignum=False, origin='lower')

    plt.subplot(2,2,4)

    U = jac_vals[:,0].reshape(X_shape)
    V = jac_vals[:,1].reshape(X_shape)

#     plt.quiver(Y, X, U, V)
    plt.quiver(X, Y, U, V)
    plt.gca().set_aspect('equal', 'box')
    print('query: ', x_t)
    plt.scatter(x_t[0], x_t[1])
    plt.title('Est')

    print('reconstruction error: ', np.mean((test_ys.flatten() - mus.flatten())**2))
    plt.show()

# func, bounds, T = functions.factory('branin-hoo')
func, bounds, T = functions.factory('himmelblau')

num_samples=25

a = np.linspace(bounds[0,0], bounds[0,1], num_samples)
b = np.linspace(bounds[1,0], bounds[1,1], num_samples)

X, Y = np.meshgrid(a,b)

test_xs =  np.vstack((X.flatten(), Y.flatten())).T
test_ys = func(test_xs).reshape((-1,1))
# init_ys = func(init_xs[:,0], x1=init_xs[:,1])
num_init_samps=100
init_xs = np.random.uniform(low=bounds[:,0], high=bounds[:,1], size=(num_init_samps,2))
init_ys = func(init_xs).reshape((-1,1))


import matplotlib.pyplot as plt

import warnings
warnings.warn('This test script is depricated.  It relies on producing an acquisiton function that takes unencoded domain points as input.', 
              DeprecationWarning,
              stacklevel=2)
exit()
if True:

    agt = agent.factory('ssp-rand', init_xs[:,:], np.atleast_2d(init_ys[:,0]).T)


    from scipy.optimize import minimize, Bounds
    n_iter = 50
    num_restarts = 10
    for t in range(n_iter):
        solns = []
        vals = []

        ### TODO fix jacobian so it returns dx in x space
        mus, var_s, phi_s = agt.eval(test_xs)

        acq_func, jac_func = agt.acquisition_func()

        acq_vals = np.array([-acq_func(x) for x in test_xs])
        jac_vals = np.array([-jac_func(x) for x in test_xs])

        jac_mag = np.sqrt(np.sum(np.power(jac_vals, 2), axis=1))


        # Use optimization to find a sample location
        for res_idx in range(num_restarts):
            
            x_init = np.random.uniform(low=bounds[:,0],
                                       high=bounds[:,1],
                                       size=(len(bounds[:,0]),))
            bnds = Bounds(bounds[:,0], bounds[:,1], keep_feasible=True)
            print('x_init:', x_init)
            # Do bounded optimization to ensure x stays in bound
            soln = minimize(acq_func, x_init,
                            jac=jac_func, 
                            method='L-BFGS-B', 
                            bounds=bnds)
    #                         bounds=bounds)
            vals.append(-soln.fun)
            solns.append(np.copy(soln.x))
        ### end for num restarts

        best_val_idx = np.argmax(vals)
    #             x_t = solns[best_val_idx].reshape((1,-1))
    #     x_t = np.atleast_2d(solns[best_val_idx].flatten())#.reshape((1,-1))

        x_t = np.atleast_2d(np.random.uniform(low=bounds[:,0],
                                       high=bounds[:,1],
                                       size=(len(bounds[:,0]),)))

        plot(test_ys, mus, acq_vals, jac_vals, X, Y, x_t.flatten())
    #     plot(test_ys, phi_s, acq_vals, jac_vals, X, Y, x_t.flatten())

        # Log actions
        query_point = np.atleast_2d(x_t.flatten())
        y_t = np.atleast_2d(func(query_point))

        print(f'| {t}\t | {y_t}\t | {query_point}\t |')
        mu_t, var_t, phi_t = agt.eval(x_t)
        agt.update(x_t, y_t, var_t)


if False:
    agt = agent.factory('ssp-mi', init_xs[:,:], np.atleast_2d(init_ys[:,0]).T)
    mus, var_s, phi_s = agt.eval(test_xs)

    acq_func, jac_func = agt.acquisition_func()

    acq_vals = np.array([-acq_func(x) for x in test_xs])
    jac_vals = np.array([-jac_func(x) for x in test_xs])

    jac_mag = np.sqrt(np.sum(np.power(jac_vals, 2), axis=1))

    plot(test_ys, mus, acq_vals, jac_vals, X, Y, [0,0])

