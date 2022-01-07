import numpy as np
import functions


from ssp_bayes_opt import agent
import matplotlib.pyplot as plt

# Generate a bunch of data
def make_data(func, bounds, num_samples=25):

    # func, bounds, T = functions.factory('branin-hoo')
    a = np.linspace(bounds[0,0], bounds[0,1], num_samples)
    b = np.linspace(bounds[1,0], bounds[1,1], num_samples)

    X, Y = np.meshgrid(a,b)

    init_xs =  np.vstack((X.flatten(), Y.flatten())).T
    # init_ys = func(init_xs[:,0], x1=init_xs[:,1])
    init_ys = func(init_xs).reshape((-1,1))
    return init_xs, init_ys, X, Y


if __name__ == '__main__':

    checkpoint = True

    from numpy.random import default_rng
    rng = default_rng()


#     func, bounds, T = functions.factory('himmelblau')
    func, bounds, T = functions.factory('branin-hoo')
    init_xs, init_ys, X, Y = make_data(func, bounds)
    num_init = 10
    shuffled_idxs = np.arange(0,init_xs.shape[0])
    rng.shuffle(shuffled_idxs)
    
    agt_all_data = agent.factory('ssp-mi', init_xs[:,:], np.atleast_2d(init_ys[:,0]).T)
    all_data_mus, all_data_var_s, all_data_phi_s = agt_all_data.eval(init_xs)


    agt_learning = agent.factory('ssp-mi', init_xs[shuffled_idxs[:10],:], np.atleast_2d(init_ys[shuffled_idxs[:10],0]).T)

    avg_func_errors = np.zeros((len(shuffled_idxs[10:]),))
    phi_func_errors = np.zeros((len(shuffled_idxs[10:]),))


    for t, sample_idx in enumerate(shuffled_idxs[10:]):
        # compute the current average function and acquisiton function
        learning_mus, learning_var_s, learning_phi_s = agt_learning.eval(init_xs)
        # compute error against "true" average function and true acq func
        avg_func_errors[t] = np.sqrt(np.mean(np.power(all_data_mus - learning_mus, 2)))
        phi_func_errors[t] = np.sqrt(np.mean(np.power(all_data_phi_s- learning_phi_s, 2)))
        print('errors:', avg_func_errors[t], phi_func_errors[t])
        # update the agent with random selection.
        x_t = np.atleast_2d(init_xs[sample_idx,:].flatten())
#         y_t = np.atleast_2d(func(x_t))
        y_t = np.atleast_2d(init_ys[sample_idx,:].flatten())
        mu_t, var_t, phi_t = agt_learning.eval(x_t)
        agt_learning.update(x_t, y_t, var_t)


        if checkpoint and t % 100 == 0: #plot current vs true.
            plt.subplot(2,2,1)
            plt.matshow(all_data_mus.reshape(X.shape), fignum=False)
            plt.ylabel('avg func')
            plt.title('All Data')

            plt.subplot(2,2,2)
            plt.matshow(learning_mus.reshape(X.shape), fignum=False)
            plt.title('Learning')

            plt.subplot(2,2,3)
            plt.matshow(init_ys.reshape(X.shape), fignum=False)
            plt.ylabel('Phi')

            plt.subplot(2,2,4)
            plt.matshow(learning_phi_s.reshape(X.shape), fignum=False)
#             acq_func, _ = agt_learning.acquisition_func()
#             acq_vals = np.array([-acq_func(x) for x in init_xs])
#             plt.matshow(acq_vals.reshape(X.shape), fignum=False)
            plt.show()
        ### end if

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(avg_func_errors, label='Avg error')
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(phi_func_errors, label='Phi error')
    plt.legend()
    plt.show()

