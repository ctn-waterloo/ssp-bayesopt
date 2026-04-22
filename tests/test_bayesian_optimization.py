import numpy as np
import pytest

import ssp_bayes_opt

BOUNDS = np.array([[-5.0, 5.0], [-5.0, 5.0]])


def himmelblau(x):
    return -((x[:, 0] ** 2 + x[:, 1] - 11) ** 2
             + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2) / 100


def test_init():
    optimizer = ssp_bayes_opt.BayesianOptimization(
        f=himmelblau, bounds=BOUNDS, random_state=1)
    optimizer.maximize(init_points=5, n_iter=0, num_restarts=3,
                       agent_type='ssp-hex', ssp_dim=51,
                       length_scale=1.0, decoder_method='from-set')
    assert optimizer.xs is not None
    assert optimizer.ys is not None


def test_max_property():
    optimizer = ssp_bayes_opt.BayesianOptimization(
        f=himmelblau, bounds=BOUNDS, random_state=2)
    optimizer.maximize(init_points=5, n_iter=2, num_restarts=2,
                       agent_type='ssp-hex', ssp_dim=51,
                       length_scale=1.0,
                       decoder_method='from-set')
    result = optimizer.max
    assert 'target' in result
    assert 'params' in result
    assert np.isfinite(result['target'])


def test_res_property():
    optimizer = ssp_bayes_opt.BayesianOptimization(
        f=himmelblau, bounds=BOUNDS, random_state=3)
    optimizer.maximize(init_points=5, n_iter=1, num_restarts=2,
                       agent_type='ssp-hex', ssp_dim=51,
                       length_scale=1.0,
                       decoder_method='from-set')
    res = optimizer.res
    assert isinstance(res, list)
    assert len(res) > 0
    assert 'target' in res[0]
