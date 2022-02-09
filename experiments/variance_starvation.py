import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

display = True
if not display:
    mpl.use('pgf')
    mpl.rcParams.update({
        'pgf.texsystem': 'pdflatex',
        'text.usetex': True,
        'figure.autolayout': True
        })
### end if
mpl.rcParams.update({
    'font.family': 'serif',
    'pgf.rcfonts': False,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'figure.autolayout': True
    })


from sklearn.model_selection import KFold
from sklearn.linear_model import BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor 
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from scipy.optimize import minimize
from scipy.interpolate import interp1d

from ssp_bayes_opt import sspspace





def sample_gp(rng):
    num_points = 1000
    xs = np.atleast_2d(np.linspace(-10,2., num_points)).T

    kernel = RBF(length_scale=0.5)
    K = kernel(xs, xs)
    ys = rng.multivariate_normal(np.zeros((num_points,)), K)
    func = interp1d(xs.flatten(), ys.flatten())
    return xs, func

def make_ssp_predictor(train_xs, train_ys, ssp_space):
    train_phis = ssp_space.encode(np.atleast_2d(train_xs))
    blr = BayesianRidge()
    blr.fit(train_phis, train_ys)
    def predictor(test_xs, ssp_space=ssp_space, pred=blr):
        test_phis = ssp_space.encode(np.atleast_2d(test_xs))
        return blr.predict(test_phis, return_std=True)
    return predictor

def optimize_lengthscale(train_xs, train_ys, ssp_space):

#     ls_0 = 1/4
#     return ls_0
    def min_func(length_scale, xs=train_xs, ys=train_ys,
                        space=ssp_space, n_splits=10, shuffle=True):

        space.update_lengthscale(length_scale)

        errors = []
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=0 if shuffle else None)
        for train_idx, test_idx in kfold.split(xs):
            train_x, test_x = xs[train_idx], xs[test_idx]
            train_y, test_y = ys[train_idx], ys[test_idx]

            ssp_pred = make_ssp_predictor(train_x, train_y.flatten(), space)

            mu, std = ssp_pred(test_x)
            diff = test_y.flatten() - mu.flatten()
            loss = -0.5*np.log(std**2) - np.divide(np.power(diff,2),std**2)
            errors.append(np.sum(-loss))
        return np.sum(errors)

#     domain_width = train_xs.max() - train_xs.min()
    domain_width = 2
    ls = np.geomspace(1/np.sqrt(train_xs.shape[0]),domain_width,30)
#     scores = [min_func(l) for l in ls]
    scores = [min_func(l) + min_func(l,n_splits=2,shuffle=False) for l in ls]

    plt.figure(figsize=(3.5,3.5))
    plt.plot(ls, scores, label='CV Score')
    plt.vlines(0.25,np.min(scores)*1.1, 0, ls='--', label='Hand-tuned length scale')

    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 
    plt.gca().set_yticks([])
    plt.gca().set_xscale('log')

    plt.xlabel('Lengthscale (l)')
    plt.ylabel('Neg. Log Likelihood')
    plt.legend()
    plt.tight_layout()

    if display:
        plt.show()
    else:
        plt.savefig('negll-vs-lenscale.pgf')
    exit()

    ls_0 = ls[np.argmin(scores)]
# #     plt.figure()
# #     plt.scatter(train_xs, train_ys)
#     plt.figure()
#     plt.scatter(ls, scores)
#     plt.vlines(1.5,np.min(scores), np.max(scores), ls='--')
#     plt.vlines(1/np.sqrt(train_xs.shape[0]),np.min(scores), np.max(scores), ls='--')
#     plt.show()
#     print(ls_0, np.min(scores))
    def reg_func(l):
        return min_func(l) + min_func(l, n_splits=2, shuffle=False)

    retval = minimize(min_func, x0=ls_0, method='l-bfgs-b',
                      bounds=[(1/np.sqrt(train_xs.shape[0]),None)],
                      options={'ftol':1e-6})
    return np.abs(retval.x) 

def get_data(rng, func='sine1'):
    num_samples=5000
    if func == 'gp_1d':
        xs, gp_func = sample_gp(rng)

        train_xs = rng.uniform(low=-10,high=0.5, size=(num_samples,)).reshape(-1,1)
        train_ys = gp_func(train_xs).reshape(-1,1)

        test_xs = np.linspace(-1., 2.0, num_samples).reshape(-1,1)
        test_ys = gp_func(test_xs).reshape(-1,1)
    elif func == 'sine_squared':
        train_xs = rng.uniform(low=-0.5, high=0, size=(num_samples,)).reshape(-1,1)
#         train_xs = np.linspace(-0.5, 0, num_samples).reshape(-1,1)
        train_ys = np.cos((2.*np.pi*train_xs)**2) + rng.normal(loc=0, scale=0.1, size=train_xs.shape)

        test_xs = np.linspace(-0.5, 0.5, 1000).reshape(-1,1)
        test_ys = np.cos((2.*np.pi*test_xs)**2).reshape(-1,1)

    elif func == 'sine1':
        train_xs = np.atleast_2d(
                    np.hstack((np.linspace(-2*np.pi,-np.pi, num_samples//2),
                               np.linspace(np.pi, 2*np.pi, num_samples//2)))
                    ).T
        train_ys = np.sin(train_xs) + rng.normal(loc=0, scale=0.1,size=train_xs.shape)

        test_xs = np.atleast_2d(np.linspace(-2*np.pi, 2.*np.pi, 1000)).T
        test_ys = np.sin(test_xs)
    elif func == 'sine2':
        train_xs = np.atleast_2d(
                    np.hstack((np.linspace(-2*np.pi,-np.pi/2, num_samples//2),
                               np.linspace(np.pi/2, 2*np.pi, num_samples//2)))
                    ).T
        train_ys = np.sin(train_xs) + rng.normal(loc=0, scale=0.1,size=train_xs.shape)

        test_xs = np.atleast_2d(np.linspace(-2.*np.pi, 2.*np.pi, 1000)).T
        test_ys = np.sin(test_xs)
    else:
        raise RuntimeWarning(f'Unrecognized function {func}')
    return train_xs, train_ys, test_xs, test_ys

if __name__ == '__main__':

    rng = np.random.default_rng(seed=1)
    # Generate training data
#     train_xs, train_ys, test_xs, test_ys = get_data(rng, func='gp_1d')
#     train_xs, train_ys, test_xs, test_ys = get_data(rng, func='sine_squared')
    train_xs, train_ys, test_xs, test_ys = get_data(rng, func='sine1')

#     plt.scatter(train_xs, train_ys)
#     plt.show()
#     exit()

    # Train GP
    gpr = None
    if True:
        kernel = RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(train_xs,train_ys)

        gpr_mu, gpr_std = gpr.predict(test_xs, return_std=True)
        gpr_mu = gpr_mu.flatten()
        print(gpr.get_params())


    lengthscale=1.5 #1/4
#     lengthscale= 0.06 #1/16
    # Train Rand-SSP
    rand_ssp_space = sspspace.RandomSSPSpace(1, 161)
#     lengthscale = optimize_lengthscale(train_xs, train_ys, rand_ssp_space)
#     l = optimize_lengthscale(train_xs, train_ys, rand_ssp_space)
    print(lengthscale)

    rand_ssp_space.update_lengthscale(lengthscale)

    rand_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), rand_ssp_space)
    rand_mus, rand_stds = rand_ssp_pred(test_xs)

    # Train Hex-SSP
    hex_ssp_space = sspspace.HexagonalSSPSpace(1, n_scales=35, scale_max=8)
#     lengthscale = optimize_lengthscale(train_xs, train_ys, hex_ssp_space)
    hex_ssp_space.update_lengthscale(lengthscale)

    hex_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), hex_ssp_space)
    hex_mus, hex_stds = hex_ssp_pred(test_xs)

    plt.figure(figsize=(20,10))
    plt.subplot(2,1,1)
    plt.plot(test_xs, rand_mus, label='Rand SSP Mean')
    plt.fill_between(test_xs.flatten(), 
                     rand_mus-rand_stds,
                     rand_mus+rand_stds, alpha=0.2, label='Rand SSP Var')
   
    if gpr is not None:
        plt.plot(test_xs, gpr_mu, label='Test GPR Mean')
        plt.fill_between(test_xs.flatten(),
                        gpr_mu-gpr_std,
                        gpr_mu+gpr_std,
                        alpha=0.2, label='Test GPR Variance')
    ### end if
    plt.plot(test_xs, test_ys, label='True')
#     plt.scatter(train_xs, train_ys, label='Data')
#     plt.xlim([test_xs.min(), test_xs.max()])
    plt.legend()
    plt.title(f'Random SSP, {rand_ssp_space.ssp_dim}, {rand_ssp_space.length_scale[0]:.2f}')

    plt.subplot(2,1,2)
    plt.plot(test_xs, hex_mus, label='Hex SSP Mean')
    plt.fill_between(test_xs.flatten(), 
                     hex_mus-hex_stds,
                     hex_mus+hex_stds, alpha=0.2, label='Hex SSP Var')
   
    if gpr is not None:
        plt.plot(test_xs, gpr_mu, label='Test GPR Mean')
        plt.fill_between(test_xs.flatten(),
                        gpr_mu-gpr_std,
                        gpr_mu+gpr_std,
                        alpha=0.2, label='Test GPR Variance')
    ### end if
    plt.plot(test_xs, test_ys, label='True')
#     plt.scatter(train_xs, train_ys, label='Data')
#     plt.xlim([test_xs.min(), test_xs.max()])
    plt.legend()
    plt.title(f'Hex SSP, {hex_ssp_space.ssp_dim}, {hex_ssp_space.length_scale[0]:.2f}')
    plt.show()
