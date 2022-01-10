import numpy as np
import functions


from ssp_bayes_opt import blr, sspspace
import matplotlib.pyplot as plt

import ssp

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
    using_ssp_space = False 
    rand_ssp = False
    if using_ssp_space:
        length_scale = 1/12
        if rand_ssp:
            ssp_space = sspspace.RandomSSPSpace(1, 127,  domain_bounds=np.array([[0,T]]), length_scale=length_scale)
        else:
            ssp_space = sspspace.HexagonalSSPSpace(1, length_scale=length_scale, 
                                        n_scales=60,
                                        scale_min=0.1,
                                        scale_max=3.4)
        ### end if
        train_ssps = ssp_space.encode(train_xs).T
        test_ssps = ssp_space.encode(test_xs).T

        blr = blr.BayesianLinearRegression(ssp_space.ssp_dim)
        blr.update(train_ssps, train_ys)
        mu, var = blr.predict(test_ssps)
    else:
        if rand_ssp:
            ssp_dim = 256
            ptrs = np.array([ssp.make_good_unitary(ssp_dim) for i in range(train_xs.shape[1])]).squeeze()
        else:
            ptrs = np.array([ssp.make_hex_unitary(train_xs.shape[1], 
                                    n_scales=30, 
                                    scale_min=0.1, 
                                    scale_max=3.4) for i in range(train_xs.shape[1])]
                ).squeeze(axis=0)
            ssp_dim = ptrs.shape[1]
        ### end if

        length_scale = 20
        train_ssps = np.array([ssp.encode(ptrs, x * length_scale) for x in train_xs]).squeeze()
        test_ssps = np.array([ssp.encode(ptrs, x * length_scale) for x in test_xs]).squeeze()
        print(train_ssps.shape)
        print(test_ssps.shape)

        blr = blr.BayesianLinearRegression(ssp_dim)
        blr.update(train_ssps, train_ys)
        mu, var = blr.predict(test_ssps)

    sim_mat = np.zeros((num_samples, num_samples))
    for i in range(num_samples):
        for j in range(num_samples):
            sim_mat[i,j] = np.dot(train_ssps[i,:], train_ssps[j,:])
    plt.matshow(sim_mat)
    plt.show()



    # display the BLR prediction.

    plt.fill_between(time.flatten(),
            (mu - np.sqrt(var)).flatten(),
            (mu + np.sqrt(var)).flatten(),
            alpha=0.4
    )
    plt.plot(time.flatten(), mu.flatten(), label='Pred')

    plt.plot(time.flatten(), x.flatten(), ls='--', label='True')
    plt.scatter(train_xs.flatten(), train_ys.flatten())

    plt.legend()
    plt.title(f'Rand SSP, lengthscale = {ssp_space.length_scale}')
    plt.show()

