import numpy as np
import pickle
from scipy.interpolate import griddata
#from util.generate_gp import sample_gp2, sample_gp4, sample_gmm
import re
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
        return himmelblau, np.array([[-5, 5], [-5, 5]]), 250
    elif function_name == 'branin-hoo':
        return branin_hoo, np.array([[-5, 10],[0, 15]]), 250
    elif function_name == 'goldstein-price':
        return goldstein_price, np.array([[-2, 2], [-2, 2]]), 250
    elif function_name == 'colville':
        return colville, np.array([[-10,10], [-10,10], [-10,10], [-10,10]]), 500
    elif function_name == 'gp_2d':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[-1, 1], [-1,1]])
        data = pickle.load(open('./gp2d.pkl', 'rb')) 
        return InterpolatedFunction('gp_2d', data['xs'], data['ys'], domain), domain, 250
    elif function_name == 'gp_4d':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[-1, 1], [-1,1], [-1, 1], [-1, 1]])
        return Function('gp_4d', sample_gp4, domain), domain, 500
    elif function_name == 'gmm':
        raise NotImplemented('TODO: fix location of cached values')
        domain = np.array([[0, 1], [0,1]])
        data = pickle.load(open('./gmm2d.pkl', 'rb')) 
        return InterpolatedFunction('gmm', data['xs'], data['ys'], domain), domain, 1000
#         return Function('gmm', sample_gmm, domain)
    elif function_name == 'mackey-glass':
        raise NotImplemented('TODO: Implement the Mackey Glass function')
        return None, None, 1000
    elif function_name == 'tsunamis':
        raise NotImplemented('Need to implement the tsunami function')
        return None, None, 500
    elif re.match('rastrigin', function_name): ## Looking for names like rastrigin1, rastrigin3 etc
        num = re.search('\d+', function_name)
        assert num is not None
        dim = int(num[0])
        return rastrigin, np.array([[-5.12,5.12] for i in range(dim)]), 500
    elif function_name=='ackley':
        return ackley, np.array([[-5,5], [-5,5]]), 500
    elif re.match('rosenbrock', function_name):
        num = re.search('\d+', function_name)
        assert num is not None
        dim = int(num[0])
        return rosenbrock, np.array([[-10,10] for i in range(dim)]), 500 # actually theres no range, just need to cover 1,1,..,1
    elif function_name=='beale':
        return beale, np.array([[-4.5,4.5], [-4.5,4.5]]), 500
    elif function_name=='easom':
        return easom, np.aray([[-10,10], [-10,10]]), 500
    elif function_name=='mccormick':
        return mccormick, np.array([[-1.5,4],[-3,4]]), 500
    elif re.match('styblinski-tang', function_name):
        num = re.search('\d+', function_name)
        assert num is not None
        dim = int(num[0])
        return styblinski_tang, np.array([[-5,5] for i in range(dim)]), 500
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
    
def rescale(func, domain):
    dim = domain.shape[0]
    old_min = domain[:,0].reshape(1,-1)
    domain_new = np.array([[-5,5] for i in range(dim)])
    new_range = (domain_new[:,1] - domain_new[:,0]).reshape(1,-1)
    func_new = lambda x: func(new_range*(x + 5)/10 + old_min) 

    return func_new, domain_new

class Function:

    def __init__(self, name, f, domain):
        self._name = name
        self.f = f
        self._domain = domain
    ### end if

    def __call__(self, **kwargs):
        return self.f(**kwargs)

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


def himmelblau(x):
    '''
    Himmelblau function, scaled by -1/100.  Negative to make it a minimzation problem, 100 to let the GP algorithm
    converge when optimizing the parameters.
    '''
    return - ((x[:,0]**2 + x[:,1] - 11)**2 + (x[:,0] + x[:,1]**2 - 7)**2 + x[:,0] + x[:,1]) / 100

def branin_hoo(x):
    '''
    Branin-Hoo function scaled by -1 to make it a minimization function.
    '''
    a = 1
    b = 5.1 / (4. * np.pi**2)
    c = 5 / np.pi
    r = 6.
    s = 10.
    t = 1 / (8. * np.pi)
    return -(a * (x[:,1] - b * x[:,0]**2 + c * x[:,0] - r)**2 + s * (1 - t) * np.cos(x[:,0]) + s)

def goldstein_price(x):
    '''
    Scaled Goldstein-Price function.  It is scalled to be roughly in the range [-10,0].  Negative to make it 
    a minimization function and by 1/1e5 to let the GP solution converge.
    Function definition from http://www.sfu.ca/~ssurjano/goldpr.html
    '''
    term_a = 1 + (x[:,0] + x[:,1] + 1)**2 * (19 - 14*x[:,0] + 3*x[:,0]**2  - 14*x[:,1] + 6*x[:,0]*x[:,1] + 3*x[:,1]**2)
    term_b = 30 + (2*x[:,0] - 3*x[:,1])**2 * (18 - 32*x[:,0] + 12*x[:,0]**2 + 48*x[:,1] - 36*x[:,0]*x[:,1] + 27*x[:,1]**2)
    return -(term_a * term_b) / 1e5

def colville(x):
    fval = 100 * (x[:,0]**2 - x[:,1])**2 + (x[:,0] - 1)**2
    fval += 90 * (x[:,2]**2 - x[:,3])**2 + 10.1 * ((x[:,1]-1)**2 + (x[:,3]-1)**2)
    fval += 19.8 * (x[:,1] - 1) * (x[:,3] - 1)

    return -fval

def rastrigin(xs):
    A = 10
    return -(A*len(xs) + sum(xs**2 - A*np.cos(2*np.pi*xs), axis=1))

def ackley(xs):
    fval = -20*np.exp(-0.2*np.sqrt(0.5*(xs[:,0]**2 + xs[:,1]**2))) - np.exp(0.5*(np.cos(2*np.pi*xs[:,0]) + np.cos(2*np.pi*xs[:,1]))) + np.exp(1) + 20
    return -fval
    
def rosenbrock(xs):
    return -np.sum(100*(xs[:,1:] - xs[:,:-1]**2)**2 + (1-xs[:,:-1])**2, axis=1)
    
def beale(xs):
    fval = (1.5 - xs[:,0] + xs[:,0]*xs[:,1])**2 + (2.25 - xs[:,0] + xs[:,0]*xs[:,1]**2)**2 + (2.625 - xs[:,0] + xs[:,0]*xs[:,1]**3)**2
    return -fval

def easom(xs):
    # scaled by 10
    xs = xs/10
    fval = -np.cos(xs[:,0])*np.cos(xs[:,1])*np.exp(-(xs[:,0]-np.pi)**2 - (xs[:,1]-np.pi)**2)
    return -fval

def mccormick(xs):
    fval = np.sin(xs[:,0] + xs[:,1]) + (xs[:,0] - xs[:,1])**2 - 1.5*xs[:,0] + 2.5*xs[:,1] + 1
    return -fval

def styblinski_tang(xs):
    return -np.sum(xs**4 - 16*xs**2 + 5*xs, axis=1)/2


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
