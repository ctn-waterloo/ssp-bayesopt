import numpy as np
import pytest

from ssp_bayes_opt import sspspace


def himmelblau(x):
    return -((x[:, 0] ** 2 + x[:, 1] - 11) ** 2
             + (x[:, 0] + x[:, 1] ** 2 - 7) ** 2
             + x[:, 0] + x[:, 1]) / 100


def branin_hoo(x):
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6., 10., 1 / (8 * np.pi)
    return -(a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
             + s * (1 - t) * np.cos(x[:, 0]) + s)


BOUNDS_2D = np.array([[-5.0, 5.0], [-5.0, 5.0]])
N_INIT = 8
SSP_DIM = 51


@pytest.fixture(scope='session')
def bounds_2d():
    return BOUNDS_2D.copy()


@pytest.fixture(scope='session')
def init_data_2d():
    rng = np.random.default_rng(42)
    xs = rng.uniform(BOUNDS_2D[:, 0], BOUNDS_2D[:, 1], size=(N_INIT, 2))
    ys = himmelblau(xs).reshape(-1, 1)
    return xs, ys


@pytest.fixture(scope='session')
def hex_ssp_space(bounds_2d):
    return sspspace.HexagonalSSPSpace(
        2, ssp_dim=SSP_DIM, domain_bounds=bounds_2d, length_scale=1.0, rng=0)


@pytest.fixture(scope='session')
def rand_ssp_space(bounds_2d):
    return sspspace.RandomSSPSpace(
        2, ssp_dim=SSP_DIM, domain_bounds=bounds_2d, length_scale=1.0, rng=0)
