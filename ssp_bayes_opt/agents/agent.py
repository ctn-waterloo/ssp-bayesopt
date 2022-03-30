import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import warnings


from .. import sspspace
from .. import blr

import functools
import warnings

from scipy.stats import qmc

# def factory(agent_type, init_xs, init_ys, **kwargs):
# 
#     data_dim = init_xs.shape[1]
#     # Initialize the agent
#     agt = None
#     if agent_type=='ssp-hex':
#         ssp_space = sspspace.HexagonalSSPSpace(data_dim, **kwargs)
#         agt = SSPAgent(init_xs, init_ys,ssp_space) 
#     elif agent_type=='ssp-rand':
#         ssp_space = sspspace.RandomSSPSpace(data_dim, **kwargs)
#         agt = SSPAgent(init_xs, init_ys,ssp_space) 
#     elif agent_type == 'gp':
#         agt = GPAgent(init_xs, init_ys)
#     elif agent_type == 'static-gp':
#         agt = GPAgent(init_xs, init_ys, updating=True, **kwargs)
#     else:
#         raise RuntimeWarning(f'Undefined agent type {agent_type}')
#     return agt


class Agent:
    def __init__(self):
        pass

    def eval(self, xs):
        pass

    def update(self, x_t, y_t, sigma_t):
        pass

    def acquisition_func(self):
        pass


class PassthroughScaler:
    def __init__(self):
        pass
    def fit(self, x):
        pass
    def transform(self, x):
        return x
    def inverse_transform(self, x):
        return x
