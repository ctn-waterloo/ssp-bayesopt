import numpy as np
import pickle
from scipy.interpolate import griddata
#from util.generate_gp import sample_gp2, sample_gp4, sample_gmm

from typing import Tuple
from collections.abc import Callable

sample_points_cache = {}

def factory(function_name: str) -> Tuple[Callable, dict, int]:
    '''
    Returns the target functions for testing the SSP Bayes optimization algorithm.

    @param function_name the name of the function.  One of:
        himmelblau | branin-hoo | goldstein-price | colville | gp_2d | gp_4d | gmm | mackey-glass | tsunamis

    @return The function as a callable object, 
            The dictionary specifying the bounds of the algorithm, 
            The budget (number of samples) allowed for this function.
    '''

    if function_name == 'himmelblau':
        return himmelblau, {'x0': (-5, 5), 'x1':(-5, 5)}, 250
    elif function_name == 'branin-hoo':
        return branin_hoo, {'x0':(-5, 10), 'x1':(0, 15)}, 250
    elif function_name == 'goldstein-price':
        return goldstein_price, {'x0':(-2, 2), 'x1':(-2, 2)}, 250
    elif function_name == 'colville':
        return colville, {'x0':(-10,10), 'x1':(-10,10), 'x2':(-10,10), 'x3':(-10,10)}, 500
    elif function_name == 'gp_2d':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[-1, 1], [-1,1]])
        data = pickle.load(open('./gp2d.pkl', 'rb')) 
        return InterpolatedFunction('gp_2d', data['xs'], data['ys'], domain), {'x0':(-1,1),'x1':(-1,1)}, 250
    elif function_name == 'gp_4d':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[-1, 1], [-1,1], [-1, 1], [-1, 1]])
        return Function('gp_4d', sample_gp4, domain), {'x0':(-1,1), 'x1':(-1,1), 'x2':(-1,1), 'x3':(-1,1)}, 500
    elif function_name == 'gmm':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[0, 1], [0,1]])
        data = pickle.load(open('./gmm2d.pkl', 'rb')) 
        return InterpolatedFunction('gmm', data['xs'], data['ys'], domain), {'x0':(0,1), 'x1':(0,1)}, 1000
#         return Function('gmm', sample_gmm, domain)
    elif function_name == 'mackey-glass':
        raise NotImplemented('TODO: Implement the Mackey Glass function')
        return None, None, 1000
    elif function_name == 'tsunamis':
        raise NotImplemented('Need to implement the tsunami function')
        return None, None, 500
    else:
        raise RuntimeError(f'Unknown function {function_name}')

def sample_points(func, num_pts):
    if not (func.name(), num_pts) in sample_points_cache:
        doms = [np.linspace(d[0], d[1], num_pts) for d in func.domain()]
        grids = np.meshgrid(*doms)
        sample_points = np.array([x.flatten() for x in grids]).T
        vals = np.expand_dims(func(sample_points.T), axis=1)

        sample_points_cache[(func.name(), num_pts)] = (sample_points, vals)
        return sample_points, vals
    else:
        return sample_points_cache[(func.name(), num_pts)]

class Function:

    def __init__(self, name, f, domain):
        self._name = name
        self.f = f
        self._domain = domain
    ### end if

    def __call__(self, **kwargs):
        return self.f(x)

    def domain(self):
        return self._domain

    def name(self):
        return self._name

class InterpolatedFunction:

    def __init__(self, name, xs, ys, domain):
        self._name = name
        self._xs = xs
        self._ys = ys
        self._domain = domain
    ### end if

    def __call__(self, **kwargs):
        x = np.array([kwargs[kw] for kw in kwargs])
        return griddata(self._xs, self._ys, x, method='linear')

    def domain(self):
        return self._domain

    def name(self):
        return self._name


def himmelblau(x0=None, x1=None):
    '''
    Himmelblau function, scaled by -1/100.  Negative to make it a minimzation problem, 100 to let the GP algorithm
    converge when optimizing the parameters.
    '''
    return - ((x0**2 + x1 - 11)**2 + (x0 + x1**2 - 7)**2 + x0 + x1) / 100

def branin_hoo(x0=None, x1=None):
    '''
    Branin-Hoo function scaled by -1 to make it a minimization function.
    '''
    a = 1
    b = 5.1 / (4. * np.pi**2)
    c = 5 / np.pi
    r = 6.
    s = 10.
    t = 1 / (8. * np.pi)
    return -(a * (x1 - b * x0**2 + c * x0 - r)**2 + s * (1 - t) * np.cos(x0) + s)

def goldstein_price(x0=None, x1=None):
    '''
    Scaled Goldstein-Price function.  It is scalled to be roughly in the range [-10,0].  Negative to make it 
    a minimization function and by 1/1e5 to let the GP solution converge.

    Function definition from http://www.sfu.ca/~ssurjano/goldpr.html
    '''
    term_a = 1 + (x0 + x1 + 1)**2 * (19 - 14*x0 + 3*x0**2  - 14*x1 + 6*x0*x1 + 3*x1**2)
    term_b = 30 + (2*x0 - 3*x1)**2 * (18 - 32*x0 + 12*x0**2 + 48*x1 - 36*x0*x1 + 27*x1**2)
    return -(term_a * term_b) / 1e5

def colville(x0=None, x1=None, x2=None, x3=None):
    fval = 100 * (x0**2 - x1)**2 + (x0 - 1)**2
    fval += 90 * (x2**2 - x3)**2 + 10.1 * ((x1-1)**2 + (x3-1)**2)
    fval += 19.8 * (x1 - 1) * (x3 - 1)

    return -fval



if __name__=='__main__':
    import matplotlib.pyplot as plt

#     xs = np.linspace(-5, 5,100)
#     ys = np.linspace(-5, 5,100)
#     X, Y = np.meshgrid(xs, ys)
#     zs = himmelblau(np.stack((X, Y), axis=1))
# 
#     print(himmelblau(np.array([[-3.779310, -3.283186]])))
#     plt.contour(X,Y,zs)
#     plt.colorbar()
#     exit()

#     xs = np.linspace(-5, 10,100)
#     ys = np.linspace(0, 15,100)
#     X, Y = np.meshgrid(ys, xs)
#     zs = branin_hoo(np.stack((X, Y), axis=1))
# 
#     print('Branin-Hoo: f(x*) = 0.397887') 
#     print(-branin_hoo(np.array([[-np.pi, 12.275],[np.pi, 2.275], [9.42478, 2.475]])))
#     fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
#     surf = ax.plot_surface(X, Y, -zs)
#     fig.colorbar(surf)

    xs = np.linspace(-2, 2, 100)
    ys = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(ys, xs)
    zs = goldstein_price(np.stack((X, Y), axis=1))

    print('Goldstein-Price: f(x*) = 3')
    print(-goldstein_price(np.array([[0,-1]])))
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    surf = ax.plot_surface(X, Y, -zs)
    fig.colorbar(surf)
    plt.show()



