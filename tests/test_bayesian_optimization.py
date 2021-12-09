import pytest
import numpy as np

from bayes_opt import BayesianOptimization

def target_func(**kwargs):
    return sum(kwargs.values())

pbounds = {'x':(-1,1), 'y':(-1,1)}

def test_init():
    optimizer = BayesianOptimization(f=target_func,bounds=pbounds,random_state=1)

    optimizer.maximize(init_points = 0, n_iter=0, num_restarts=10)

    print(optimizer.max())


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
