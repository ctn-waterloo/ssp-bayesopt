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



def cv_score(length_scale, xs=None, ys=None,
                    space=None, n_splits=10, shuffle=True):

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


def get_data(rng, func='sine1'):
    num_samples=5000

    func_names = {'sine1': 'Noisy Sinusoid', 
                  'sine2': 'Noisy Sinusoid',
                  'gp_1d': 'Function Sampled from GP',
                  'sine_squared': r'$cos((2\pi x)^{2})$'}

    lenscales  = {'sine1': 1.5, 
                  'sine2': 1.5,
                  'gp_1d': 1/4,
                  'sine_squared': 1/16}
    if func == 'gp_1d':
        xs, gp_func = sample_gp(rng)

        train_xs = rng.uniform(low=-10,high=0.5, size=(num_samples,)).reshape(-1,1)
        train_ys = gp_func(train_xs).reshape(-1,1) + rng.normal(loc=0,scale=0.1 , size=train_xs.shape)

        test_xs = np.linspace(-0.5, 1.0, num_samples).reshape(-1,1)
        test_ys = gp_func(test_xs).reshape(-1,1)
    elif func == 'sine_squared':
        train_xs = rng.uniform(low=-0.5, high=0, size=(num_samples,)).reshape(-1,1)
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
    return train_xs, train_ys, test_xs, test_ys, func_names[func], lenscales[func]

if __name__ == '__main__':

    rng = np.random.default_rng(seed=1)
    func = 'sine_squared'
    # Generate training data
#     train_xs, train_ys, test_xs, test_ys, func_name, lenscale  = get_data(rng, func='gp_1d')
#     train_xs, train_ys, test_xs, test_ys, func_name, lenscale  = get_data(rng, func='sine_squared')
    train_xs, train_ys, test_xs, test_ys, func_name, lenscale  = get_data(rng, func=func)




    # Train GP

#     lengthscale=1/4
#     lenscale = 1/16 # 0.065 #0.08
#     lenscale = np.sqrt(2)/20 #1/16 # 0.065 #0.08
    # Train Rand-SSP
    rand_ssp_space = sspspace.RandomSSPSpace(1, 161)
    hex_ssp_space = sspspace.HexagonalSSPSpace(1, n_scales=35, scale_max=8)

    def min_func(l, xs=train_xs, ys=train_ys, space=None):
        return cv_score(l, xs=xs, ys=ys, space=space)

#     domain_width = train_xs.max() - train_xs.min()
    domain_width = 2
    ls = np.geomspace(1/np.sqrt(train_xs.shape[0]),domain_width,30)
    rand_scores = [min_func(l, space=rand_ssp_space) for l in ls]
    hex_scores = [min_func(l, space=hex_ssp_space) for l in ls]


    # Make ls vs variance starvation plot
    gpr = None
    if True:
        kernel = RBF() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(train_xs,train_ys)

        gpr_mu, gpr_std = gpr.predict(test_xs, return_std=True)
        gpr_mu = gpr_mu.flatten()
        print(gpr.get_params())

    rand_var_ratio_mu = np.zeros((ls.size,))
    rand_var_ratio_std = np.zeros((ls.size,))
    hex_var_ratio_mu = np.zeros((ls.size,))
    hex_var_ratio_std = np.zeros((ls.size,))

    for l_idx, l in enumerate(ls):
        rand_ssp_space = sspspace.RandomSSPSpace(1, 161)
        rand_ssp_space.update_lengthscale(l)
        rand_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), 
                                           rand_ssp_space)
        rand_mus, rand_stds = rand_ssp_pred(test_xs)
        rand_ratio = np.divide(rand_stds, gpr_std)
        rand_var_ratio_mu[l_idx] = np.min(rand_ratio)
        rand_var_ratio_std[l_idx] = np.std(rand_ratio)

        hex_ssp_space = sspspace.HexagonalSSPSpace(1, n_scales=35, scale_max=8)
        hex_ssp_space.update_lengthscale(l)
        hex_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), 
                                          hex_ssp_space)
        hex_mus, hex_stds = hex_ssp_pred(test_xs)
        hex_ratio =  np.divide(hex_stds, gpr_std)
        hex_var_ratio_mu[l_idx] = np.min(hex_ratio)
        hex_var_ratio_std[l_idx] = np.std(hex_ratio)


    plt.figure(figsize=(3.5,3.5))
    plt.subplot(2,1,1)
    plt.plot(ls, rand_scores, label='Rand SSP CV Score', ls='--')
    plt.plot(ls, hex_scores, label='Hex SSP CV Score')
    plt.vlines(lenscale,np.min(rand_scores)*1.5, 0, ls='dotted',
                label=f'Length scale = {lenscale:.2f}')

    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 
    plt.gca().set_yticks([])
    plt.gca().set_xticks([])
    plt.gca().set_xscale('log')

#     plt.xlabel('Lengthscale (l)')
    plt.ylabel('Neg. Log Likelihood')
    plt.legend(loc='upper left')
    plt.tight_layout()

#     if display:
#         plt.show()
#     else:
#         plt.savefig('negll-vs-lenscale.pgf')


#     plt.figure(figsize=(3.5,3.5))
    plt.subplot(2,1,2)
    plt.plot(ls, rand_var_ratio_mu, label=r'Rand SSP', ls='--')
    plt.plot(ls, hex_var_ratio_mu, label=r'Hex SSP')
    plt.vlines(lenscale, 0, 10,
               ls='dotted', label=f'length scale = {lenscale:.2f}')
    plt.hlines(1.0, np.min(ls), np.max(ls))
    plt.gca().spines['left'].set_position(('outward', 10))
    plt.gca().spines['bottom'].set_position(('outward', 10))

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 
#     plt.gca().set_yticks([])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')

    plt.xlabel('Length scale (l)')
    plt.ylabel(r'min $\sigma_\mathrm{SSP}$/$\sigma_{GP}$ (log scale)')
    plt.legend(loc='upper left')
    plt.tight_layout()

#     if display:
#         plt.show()
#     else:
#         plt.savefig('nll-and-var-ratio-vs-lenscale.pgf')
#     exit()


#     lenscale = 0.25
    rand_ssp_space = sspspace.RandomSSPSpace(1, 161)
    rand_ssp_space.update_lengthscale(lenscale)
    rand_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), 
                                       rand_ssp_space)
    rand_mus, rand_stds = rand_ssp_pred(test_xs)

    hex_ssp_space = sspspace.HexagonalSSPSpace(1, n_scales=35, scale_max=8)
    hex_ssp_space.update_lengthscale(lenscale)
    hex_ssp_pred = make_ssp_predictor(train_xs, train_ys.flatten(), 
                                      hex_ssp_space)
    hex_mus, hex_stds = hex_ssp_pred(test_xs)


    plt.figure(figsize=(3.5,3.5))
    plt.subplot(2,1,1)
    plt.plot(test_xs, rand_mus, label=f'Rand SSP dim={rand_ssp_space.ssp_dim}')
    plt.fill_between(test_xs.flatten(), 
                     rand_mus-rand_stds,
                     rand_mus+rand_stds, alpha=0.2)#, label='Rand SSP Var')
   
    if gpr is not None:
        plt.plot(test_xs, gpr_mu, label='Test GPR', ls='-.')
        plt.fill_between(test_xs.flatten(),
                        gpr_mu-gpr_std,
                        gpr_mu+gpr_std,
                        alpha=0.2)#, label='Test GPR Variance')
    ### end if
    plt.plot(test_xs, test_ys, label='True', ls='dotted')
    plt.scatter(train_xs, train_ys, label='Data')
    plt.xlim([test_xs.min(), test_xs.max()])
    plt.legend(loc='upper left')
#     plt.title(f'Random SSP, {rand_ssp_space.ssp_dim}, {rand_ssp_space.length_scale[0]:.2f}')
    plt.title(f'Estimating {func_name}')

    plt.subplot(2,1,2)
    plt.plot(test_xs, hex_mus, label=f'Hex SSP dim={hex_ssp_space.ssp_dim}')
    plt.fill_between(test_xs.flatten(), 
                     hex_mus-hex_stds,
                     hex_mus+hex_stds, alpha=0.2)#, label='Hex SSP Var')
   
    if gpr is not None:
        plt.plot(test_xs, gpr_mu, label='Test GPR', ls='-.')
        plt.fill_between(test_xs.flatten(),
                        gpr_mu-gpr_std,
                        gpr_mu+gpr_std,
                        alpha=0.2)#, label='Test GPR Variance')
    ### end if
    plt.plot(test_xs, test_ys, label='True', ls='dotted')
    plt.scatter(train_xs, train_ys, label='Data')
    plt.xlim([test_xs.min(), test_xs.max()])
    plt.legend(loc='upper left')
#     plt.title(f'Hex SSP, {hex_ssp_space.ssp_dim}, {hex_ssp_space.length_scale[0]:.2f}')
    if display:
        plt.show()
    else:
        plt.safefig('.pgf')

    pass
