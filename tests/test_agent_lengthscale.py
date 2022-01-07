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

    num_trials = 10

    from numpy.random import default_rng
    rng = default_rng()


    # Select a test function
#     func, bounds, T = functions.factory('himmelblau')
    func, bounds, T = functions.factory('branin-hoo')
    init_xs, init_ys, X, Y = make_data(func, bounds)
    num_init = 625
    shuffled_idxs = np.arange(0,init_xs.shape[0])


    length_scales = np.zeros((num_trials, init_xs.shape[1]))
    for t in range(num_trials):
        # Shuffle the data
        rng.shuffle(shuffled_idxs)
#         print(shuffled_idxs[:num_init])
        # Create an agent
        ## TODO: use random subset of shuffled_idxs
        agt = agent.factory('ssp-mi', 
                                    init_xs[shuffled_idxs[:num_init],:], 
                                    np.atleast_2d(
                                        init_ys[shuffled_idxs[:num_init],0]).T
                                    )
        # Evaluate the quality of the lengthscale parameter.
        length_scales[t,:] = agt.length_scale
        mus, var_s, phi_s = agt.eval(init_xs)
        rmse = np.sqrt(np.mean(np.power(init_ys.flatten() - mus.flatten(),2)))
        print(rmse)

        ## At the very least should be less than the bounds of the domain.
        # Plot the acquisition function

        plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(init_ys.reshape(X.shape), 
                vmin=np.min(init_ys),vmax=np.max(init_ys),
                origin='lower', label='True Values')
        plt.subplot(1,3,2)
        plt.imshow(mus.reshape(X.shape), 
#                 vmin=np.min(init_ys),vmax=np.max(init_ys),
                origin='lower', label='Predicted')

        plt.subplot(1,3,3)
        sample_points = np.zeros(X.shape)
        for i in range(num_init):
            p_x, p_y = init_xs[shuffled_idxs[i],:]
            sample_points += np.logical_and(X == p_x, Y==p_y)

        plt.imshow(sample_points, origin='lower', label='Sample Loc')
        plt.show()
    ### end for
    print(np.mean(length_scales, axis=0), np.std(length_scales, axis=0))
