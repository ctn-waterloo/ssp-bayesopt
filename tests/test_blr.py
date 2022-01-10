import numpy as np
import functions


from ssp_bayes_opt import blr, sspspace
import matplotlib.pyplot as plt

import old_ssp

def generate_signal(T, dt, rms, limit, seed=1):
    np.random.seed(seed)
    limit_radians = limit * 2 * np.pi
    n = int(T / dt)
    freqs = np.fft.fftfreq(n, d=dt)
    power = np.zeros(freqs.shape, dtype=complex)
    power[0] = np.random.normal()
    power[1:n//2+1] = np.random.normal(size=(n//2,)) + 1j*np.random.normal(size=(n//2,))
    power[n//2+1:] = np.conjugate(power[n//2:1:-1])

    power[np.abs(freqs) > limit_radians] = 0

    x = np.fft.irfft(power[:n//2+1])

    integral_val = np.sqrt(np.sum(np.power(x,2)) * dt / T)
    scale = rms / integral_val

    return x * scale, power * scale, freqs

def train_ssp_space(ssp_space, train_xs, train_ys, test_xs):
    train_ssps = ssp_space.encode(train_xs).T
    test_ssps = ssp_space.encode(test_xs).T

    pred = blr.BayesianLinearRegression(ssp_space.ssp_dim)
    pred.update(train_ssps, train_ys)
    mu, var = pred.predict(test_ssps)
    return mu, var

def train_old_ssp(ptrs, train_xs, train_ys, test_xs, length_scale=1/12):

    train_ssps = np.array([old_ssp.encode(ptrs, x / length_scale) for x in train_xs]).squeeze()
    test_ssps = np.array([old_ssp.encode(ptrs, x / length_scale) for x in test_xs]).squeeze()

    pred = blr.BayesianLinearRegression(train_ssps.shape[1])
    pred.update(train_ssps, train_ys)
    mu, var = pred.predict(test_ssps)

    return mu, var


if __name__=='__main__':

    T = 2
    dt = 1e-3
    rms = 0.5
    limit_hz = 1
    x, X, omega = generate_signal(T, dt, rms, limit_hz)
    time = np.arange(0,T,dt)

    # Sample a bunch of points, 
    sample_idxs = list(range(len(x)))
    np.random.shuffle(sample_idxs)

    num_samples = 100
    train_xs = np.atleast_2d(time[sample_idxs[:num_samples]]).T
    train_ys = np.atleast_2d(x[sample_idxs[:num_samples]]).T

    test_xs = np.copy(time)
    # encode them as ssps

    # Using ssp_space
    length_scale = 1/12
    rand_ssp_space = sspspace.RandomSSPSpace(1, 127,  
                                domain_bounds=np.array([[0,T]]),
                                length_scale=length_scale)
    hex_ssp_space = sspspace.HexagonalSSPSpace(1, length_scale=length_scale, 
                                    n_scales=30,
                                    scale_min=0.1,
                                    scale_max=3.4)

    rand_mu, rand_var = train_ssp_space(rand_ssp_space, train_xs, train_ys, test_xs)
    hex_mu, hex_var = train_ssp_space(hex_ssp_space, train_xs, train_ys, test_xs)

    # Using old SSP
    rand_ssp_dim = 256
    rand_ptrs = np.array([old_ssp.make_good_unitary(rand_ssp_dim) for i in range(train_xs.shape[1])]).squeeze()
    hex_ptrs = np.array([old_ssp.make_hex_unitary(train_xs.shape[1], 
                                    n_scales=30, 
                                    scale_min=0.1, 
                                    scale_max=3.4) for i in range(train_xs.shape[1])]
        ).squeeze(axis=0)



    old_rand_mu, old_rand_var = train_old_ssp(rand_ptrs, train_xs, train_ys, test_xs)
    old_hex_mu, old_hex_var = train_old_ssp(hex_ptrs, train_xs, train_ys, test_xs)

#     sim_mat = np.zeros((num_samples, num_samples))
#     for i in range(num_samples):
#         for j in range(num_samples):
#             sim_mat[i,j] = np.dot(train_ssps[i,:], train_ssps[j,:])
#     plt.matshow(sim_mat)
#     plt.show()



    # display the BLR prediction.

    plt.subplot(2,2,1)

    plt.fill_between(time.flatten(),
            (rand_mu - np.sqrt(rand_var)).flatten(),
            (rand_mu + np.sqrt(rand_var)).flatten(),
            alpha=0.4
    )
    plt.plot(time.flatten(), rand_mu.flatten(), label='Pred')

    plt.plot(time.flatten(), x.flatten(), ls='--', label='True')
    plt.scatter(train_xs.flatten(), train_ys.flatten())

    plt.legend()
    plt.title('ssp_space')
    plt.ylabel('Rand')


    plt.subplot(2,2,3)

    plt.fill_between(time.flatten(),
            (hex_mu - np.sqrt(hex_var)).flatten(),
            (hex_mu + np.sqrt(hex_var)).flatten(),
            alpha=0.4
    )
    plt.plot(time.flatten(), hex_mu.flatten(), label='Pred')

    plt.plot(time.flatten(), x.flatten(), ls='--', label='True')
    plt.scatter(train_xs.flatten(), train_ys.flatten())

    plt.legend()
    plt.ylabel('Hex')


    plt.subplot(2,2,2)

    plt.fill_between(time.flatten(),
            (old_rand_mu - np.sqrt(old_rand_var)).flatten(),
            (old_rand_mu + np.sqrt(old_rand_var)).flatten(),
            alpha=0.4
    )
    plt.plot(time.flatten(), old_rand_mu.flatten(), label='Pred')

    plt.plot(time.flatten(), x.flatten(), ls='--', label='True')
    plt.scatter(train_xs.flatten(), train_ys.flatten())

    plt.legend()
    plt.title('old_ssp')


    plt.subplot(2,2,4)

    plt.fill_between(time.flatten(),
            (old_hex_mu - np.sqrt(old_hex_var)).flatten(),
            (old_hex_mu + np.sqrt(old_hex_var)).flatten(),
            alpha=0.4
    )
    plt.plot(time.flatten(), old_hex_mu.flatten(), label='Pred')

    plt.plot(time.flatten(), x.flatten(), ls='--', label='True')
    plt.scatter(train_xs.flatten(), train_ys.flatten())

    plt.legend()


    plt.show()

