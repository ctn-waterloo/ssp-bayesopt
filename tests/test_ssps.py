import numpy as np
import scipy

import pytest

from ssp_bayes_opt import sspspace

def make_hex_space(domain_dim=2, ls=1, bounds = 15*np.array([[-1,1],[-1,1]])):
    ssp_space = sspspace.HexagonalSSPSpace(
                    domain_dim,
                    ssp_dim=151, 
                    scale_min=2*np.pi/np.sqrt(6) - 0.5, 
                    scale_max=2*np.pi/np.sqrt(6) + 0.5, 
                    domain_bounds=bounds, 
                    length_scale=ls)
    return ssp_space


def make_rand_space(domain_dim=2, ls=1, bounds = 15*np.array([[-1,1],[-1,1]])):
    ssp_space = sspspace.RandomSSPSpace(
                    domain_dim,
                    ssp_dim=151, 
                    domain_bounds=bounds, 
                    length_scale=ls)
    return ssp_space
    
def test_constructor():
    ssp_space = make_hex_space()
    assert True

def test_encoder():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    assert np.inner(S0,S0).size == 1
    assert np.isclose(np.inner(S0,S0), 1)

def test_hex_from_set_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='from-set', num_samples=300)
    assert np.all(np.isclose(test_x, recov_x, atol=1e-1))


def test_rand_from_set_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    rand_ssp_space = make_rand_space()
    S0 = rand_ssp_space.encode(test_x)
    assert S0.shape == (1, rand_ssp_space.ssp_dim)
    recov_x = rand_ssp_space.decode(S0, method='from-set', num_samples=300)
    assert np.all(np.isclose(test_x, recov_x, atol=1e-1))


def test_hex_direct_optim_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='direct-optim')
    assert np.all(np.isclose(test_x, recov_x))

def test_rand_direct_optim_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    rand_ssp_space = make_rand_space()
    S0 = rand_ssp_space.encode(test_x)
    assert S0.shape == (1, rand_ssp_space.ssp_dim)
    recov_x = rand_ssp_space.decode(S0, method='direct-optim')
    assert np.all(np.isclose(test_x, recov_x))


def test_hex_small_lenscale_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space(ls=0.05)
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='from-set', num_samples=500)
    assert np.all(np.isclose(test_x, recov_x, atol=1e-2))

def test_rand_small_lenscale_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    rand_ssp_space = make_rand_space(ls=0.05)
    S0 = rand_ssp_space.encode(test_x)
    assert S0.shape == (1, rand_ssp_space.ssp_dim)
    recov_x = rand_ssp_space.decode(S0, method='direct-optim')
    assert np.all(np.isclose(test_x, recov_x))


def test_hex_large_lenscale_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space(ls=2)
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='direct-optim')
    assert np.all(np.isclose(test_x, recov_x))

def test_rand_large_lenscale_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    rand_ssp_space = make_rand_space(ls=2)
    S0 = rand_ssp_space.encode(test_x)
    assert S0.shape == (1, rand_ssp_space.ssp_dim)
    recov_x = rand_ssp_space.decode(S0, method='direct-optim')
    assert np.all(np.isclose(test_x, recov_x))
