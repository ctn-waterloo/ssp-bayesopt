import numpy as np
import scipy

import pytest

from ssp_bayes_opt import sspspace

def make_hex_space(domain_dim=2, bounds = 15*np.array([[-1,1],[-1,1]])):
    ssp_space = sspspace.HexagonalSSPSpace(
                    domain_dim,
                    ssp_dim=151, 
                    scale_min=2*np.pi/np.sqrt(6) - 0.5, 
                    scale_max=2*np.pi/np.sqrt(6) + 0.5, 
                    domain_bounds=bounds, 
                    length_scale=1)
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

def test_from_set_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='from-set')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-1))

def test_direct_optim_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='direct-optim')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-1))

@pytest.mark.skip(reason='known bad')
def test_scaled_set_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='weighted-from-set')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-1))


@pytest.mark.skip(reason='known bad')
def test_lst_sqs_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='least-squares')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-4))

@pytest.mark.skip(reason='known bad')
def test_grad_descent_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='grad_descent')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-4))




@pytest.mark.skip(reason='known bad')
def test_grad_descent_decode():
    test_x = np.atleast_2d(np.array([1.3,-3.4]))
    ssp_space = make_hex_space()
    S0 = ssp_space.encode(test_x)
    assert S0.shape == (1, ssp_space.ssp_dim)
    recov_x = ssp_space.decode(S0, method='nonlin-reg')
    ### Note: Set the absolute tolerance high because of the doding method we are using
    assert np.all(np.isclose(test_x, recov_x, atol=1e-4))
