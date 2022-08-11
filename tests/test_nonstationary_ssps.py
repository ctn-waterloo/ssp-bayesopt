from ssp_bayes_opt import sspspace, nonstationary_sspspace

import pytest
import numpy as np

def make_hex_space(domain_dim=2, ls=1, bounds = 15*np.array([[-1,1],[-1,1]])):
    ssp_space = sspspace.HexagonalSSPSpace(
                    domain_dim,
                    ssp_dim=151, 
                    scale_min=2*np.pi/np.sqrt(6) - 0.5, 
                    scale_max=2*np.pi/np.sqrt(6) + 0.5, 
                    domain_bounds=bounds, 
                    length_scale=ls)
    return ssp_space

def make_regions(bounds=np.array([[-1,1],[-1,1]])):
    s1 = np.arange(0,3)
    s2 = np.arange(3,5)
    r1 = nonstationary_sspspace.IncludeRegion(
                bounds=0.5 * bounds,
                scales=s1)
    r2 = nonstationary_sspspace.ExcludeRegion(
            bounds=0.5 * bounds,
            scales=s2)
    return (r1,r2)

def test_include_region():
    r, _ = make_regions()
    memberships = r(np.array([[0,0],[-1,1]]))
    assert memberships[0]
    assert not memberships[1]

def test_exclude_region():
    _, r = make_regions()
    memberships = r(np.array([[0,0],[-1,1]]))
    assert not memberships[0]
    assert memberships[1]

def test_constructor():
    ssp_space = make_hex_space(bounds=np.array([[-1,1],[-1,1]]))
    regions = make_regions()

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

    s0 = np.array([[0,0]])
    p1 = nonstationary_space.encode(s0)
    p1 = p1 / np.linalg.norm(p1)
    print(np.dot(p1,p1.T))
    assert np.isclose(np.dot(p1,p1.T), 1)


def test_1d():
    ssp_space = make_hex_space(domain_dim=1, bounds=np.array([[-1,1]]))
    regions = make_regions(bounds=np.array([[-1,1]]))

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

    s0 = np.array([[0]])
    p1 = nonstationary_space.encode(s0)
    xs = np.atleast_2d(np.linspace(-1,1,100)).T
    ps = nonstationary_space.encode(xs)

    sims = np.einsum('nd,jd->nj', ps, p1)
    import matplotlib.pyplot as plt
    plt.plot(xs, sims)
    plt.show()
