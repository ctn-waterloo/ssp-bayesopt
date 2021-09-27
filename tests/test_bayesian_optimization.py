import pytest
import numpy as np

from bayes_opt import BayesianOptimization

def target_func(**kwargs):
    return sum(kwargs.values())

pbounds = {'x':(-1,1), 'y':(-1,1)}

def test_init():
    optimizer = BayesianOptimization(f=target_func,bounds=pbounds,random_state=1)

