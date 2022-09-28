import numpy as np
import matplotlib.pyplot as plt
import sspspace

dim = 1001

ls=np.pi

gauss_space = sspspace.GaussianSSPSpace(
                 domain_dim=1,
                 ssp_dim=dim,
                 length_scale=ls,
                 domain_bounds=np.array([[-2,2]]),
)
rand_space = sspspace.RandomSSPSpace(
                 domain_dim=1,
                 ssp_dim=dim,
                 length_scale=ls,
                 domain_bounds=np.array([[-2,2]]),
)

def get_sims(space):

    xs = np.atleast_2d(np.linspace(-20,20,1000)).T

    phi0 = space.encode(np.array([[0]]))
    phis =  space.encode(xs)

    sims = np.einsum('nd,jd -> n', phis, phi0)
    return xs, sims

xs, gauss_sims = get_sims(gauss_space)
_, rand_sims = get_sims(rand_space)

plt.plot(xs, gauss_sims, label='Gaussian SSP')
plt.plot(xs, rand_sims, label='Uniform SSP')
plt.legend()
plt.show()
