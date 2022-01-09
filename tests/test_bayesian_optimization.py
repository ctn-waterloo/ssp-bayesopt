import pytest
import numpy as np

import ssp_bayes_opt 

def target_func(**kwargs):
    return sum(kwargs.values())

bounds = np.array([[-5, 5], [-5, 5]])

def test_init():
    optimizer = ssp_bayes_opt.BayesianOptimization(f=target_func,bounds=bounds,random_state=1)

    optimizer.maximize(init_points = 0, n_iter=0, num_restarts=10)

    print(optimizer.max())


if __name__ == '__main__':
    r"""
    CommandLine:
        python tests/test_bayesian_optimization.py
    """
    pytest.main([__file__])
