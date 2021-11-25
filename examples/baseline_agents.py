import numpy as np
import matplotlib.pyplot as plt

import GPy
import ssp_bayes_opt
from ssp_bayes_opt.agent import Agent, SSPAgent

import functools

def factory(agent_type):
    if agent_type == 'random':
        return RandomAgent
    elif agent_type == 'gp-mi':
        return GPAgent  
    elif agent_type == 'ssp-mi':
        return lambda *args: SSPAgent(*args, axis_type = 'rand') 
    elif agent_type == 'hex-mi':
#         return SSPAgent(128, init_xs, init_ys, ssp_type='hex',  n_scales=3, n_rotates=3) 
        return  lambda *args: SSPAgent(*args, axis_type = 'hex') 
    else:
        raise RuntimeError(f'unknown agent type {agent_type}')

class RandomAgent(Agent):
    def eval(self, xs):
        return np.zeros((xs.shape[1],)), np.random.normal(loc=0,scale=1,size=xs.shape[1])

class GPAgent(Agent):
    def __init__(self, init_xs, init_ts):
        self.obs = init_xs
        self.ts = init_ts
        self.gp = GPy.models.GPRegression(self.obs, self.ts)#, self.kernel)
        try:
            self.gp.optimize(messages=True)
        except:
            print(init_xs, init_ts)
            exit()

        self.sqrt_alpha = np.log(2/1e-6)

        self.gamma_t = 0
    ### end __init__

    def eval(self, xs):
        mu, var = self.gp.predict(xs)
        phi = self.sqrt_alpha * (np.sqrt(var + self.gamma_t) - np.sqrt(self.gamma_t)) 
        return mu, var, phi

    def update(self, x_t, y_t, sigma_t):
        self.obs = np.vstack((self.obs, x_t))
        self.ts = np.vstack((self.ts, y_t))
        self.gamma_t = self.gamma_t + sigma_t
        self.gp.set_XY(X=self.obs, Y=self.ts)
#         self.gp.parameters_changed()

