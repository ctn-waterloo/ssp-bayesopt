from ssp_bayes_opt import sspspace, nonstationary_sspspace

import pytest
import numpy as np
import functools

import polygons

def make_hex_space(domain_dim=2, ls=1, bounds = 15*np.array([[-1,1],[-1,1]])):
    ssp_space = sspspace.HexagonalSSPSpace(
                    domain_dim,
                    ssp_dim=151, 
                    scale_min=0.5, 
                    scale_max=1, 
                    n_rotates=10,
                    domain_bounds=bounds, 
                    length_scale=ls)
    return ssp_space


def m1(x, bounds):
    dims = bounds.shape[0] 
    membership = [np.logical_and( x[:,i] > bounds[i,0],
                                  x[:,i] < bounds[i,1]) for i in range(dims)]
    return functools.reduce(np.logical_and, membership)

def m2(x, bounds):
    dims = bounds.shape[0] 
    r1 = [np.logical_or( x[:,i] < bounds[i,0],
                         x[:,i] > bounds[i,1]) for i in range(dims)]
    return functools.reduce(np.logical_or, r1)

def make_regions(ssp_space, bounds=np.array([[-1,1],[-1,1]])):
    return make_banana_regions(ssp_space, bounds=bounds)

def make_banana_regions(ssp_space, bounds=np.array([[-1,1],[-1,1]])):
    
    pts = [
            [(0.,0.),(0.5,-0.25), (0,-0.5), (-0.5, -0.25),(-0.5,0.5)],
    ]

    num_edges_children = 4
    num_nodes_children = 4
    tree = polygons.build_search_tree(pts,
                                      num_edges_children,
                                      num_nodes_children)

    def m1(xs, t=tree):
        ps = list(map(tuple, xs))
        return np.array(polygons.points_are_inside(t,ps))

    def m2(xs, t=tree):
        ps = list(map(tuple, xs))
        return np.logical_not(np.array(polygons.points_are_inside(t,ps)))

    memberships = [lambda x: m1(x), lambda x: m2(x)]

#     rs = nonstationary_sspspace.SSPRegion(
    rs = nonstationary_sspspace.MLPRegion(
                bounds=bounds,
                memberships=memberships,
                encoder=ssp_space)
    return rs



def make_rectangular_regions(ssp_space, bounds=np.array([[-1,1],[-1,1]])):


    memberships = [lambda x: m1(x, 0.5 * bounds), lambda x: m2(x, 0.5 * bounds)]

#     rs = nonstationary_sspspace.SSPRegion(
    rs = nonstationary_sspspace.MLPRegion(
                bounds=bounds,
                memberships=memberships,
                encoder=ssp_space)
    return rs

def test_ssp_regions():
    print('Test ssp regions')
    ssp_space = make_hex_space(bounds=np.array([[-1,1],[-1,1]]))
#     ssp_space.update_lengthscale(0.1)
    ssp_space.update_lengthscale(1.5)

    regions = make_regions(ssp_space)

    test_pts = np.array([[-0.99,0.99], [0,0], [0.1, 0.1], [-0.1, -0.1], [0.45, 0.45], [-0.99,-0.99]])
    for p in test_pts:
        print('loop')
        print(p, m1(np.atleast_2d(p), np.array([[-0.5, 0.5],[-0.5,0.5]])))
        print(p, m2(np.atleast_2d(p), np.array([[-0.5, 0.5],[-0.5,0.5]])))
        print(regions(np.atleast_2d(p)))
    print(test_pts)
    print(regions(test_pts))

def plot_regions():

    ssp_space = make_hex_space(bounds=np.array([[-1,1],[-1,1]]))
    ssp_space.update_lengthscale(0.1)

    regions = make_regions(ssp_space)

    xs = np.linspace(-1,1,400)

    X,Y = np.meshgrid(xs,xs)
    domain_samples = np.vstack((X.flatten(),Y.flatten())).T


    sims = regions(domain_samples)
    inside_sims = sims[0,:]
    outside_sims = sims[1,:]

    import matplotlib.pyplot as plt

    plt.subplot(1,2,1)
    plt.matshow(inside_sims.reshape((len(xs),len(xs))), fignum=False)
#     plt.colorbar()
    plt.title('inside')

    plt.subplot(1,2,2)
    plt.matshow(outside_sims.reshape((len(xs),len(xs))), fignum=False)
#     plt.colorbar()
    plt.title('outside')

    plt.suptitle(f'Membership function: {type(regions).__name__}')
    plt.show()

def plot_similarities():

    ssp_space = make_hex_space(bounds=np.array([[-1,1],[-1,1]]))
    ssp_space.update_lengthscale(0.1)

    regions = make_regions(ssp_space)

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

    xs = np.linspace(-1,1,400)

    X,Y = np.meshgrid(xs,xs)
    domain_samples = np.vstack((X.flatten(),Y.flatten())).T

    s0 = np.array([[0,0]])
    p0 = nonstationary_space.encode(s0)

    s1 = np.array([[-0.75,-0.75]])
    p1 = nonstationary_space.encode(s1)
    ps = nonstationary_space.encode(domain_samples)

    sims0 = np.einsum('nd,jd->nj', ps, p0)
    sims1 = np.einsum('nd,jd->nj', ps, p1)

    import matplotlib.pyplot as plt

    plt.subplot(1,2,1)
    plt.matshow(sims0.reshape((len(xs),len(xs))), fignum=False)
    plt.title(f'Similarity to x={s0}')

    plt.subplot(1,2,2)
    plt.matshow(sims1.reshape((len(xs),len(xs))), fignum=False)
    plt.title(f'Similarity to x={s1}')

    plt.suptitle(f'Membership function: {type(regions).__name__}')
    plt.show()

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
    ssp_space.update_lengthscale(0.1)
    regions = make_regions(bounds=ssp_space.domain_bounds)

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

    s0 = np.array([[0]])
    p1 = nonstationary_space.encode(s0)
    xs = np.atleast_2d(np.linspace(-1,1,100)).T
    ps = nonstationary_space.encode(xs)

    sims = np.einsum('nd,jd->nj', ps, p1)
    import matplotlib.pyplot as plt
    plt.plot(xs, sims)
    plt.show()

def test_2d_pt():
    ssp_space = make_hex_space(domain_dim=2, bounds=np.array([[-1,1],[-1,1]]))
    ssp_space.update_lengthscale(0.01)
    regions = make_regions(bounds=ssp_space.domain_bounds)

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

#     s0 = np.array([[0,0]])
    s0 = np.array([[-1,0]])
    p0 = nonstationary_space.encode(s0)
    s1 = np.array([[-0.25,0]])
    p1 = nonstationary_space.encode(s1)
    print(np.dot(p0.flatten(),p1.flatten()))

def test_2d():
    ssp_space = make_hex_space(domain_dim=2, bounds=np.array([[-1,1],[-1,1]]))
    ssp_space.update_lengthscale(0.1)
    regions = make_regions(bounds=ssp_space.domain_bounds)

    nonstationary_space = nonstationary_sspspace.NonstationarySSPSpace(ssp_space, regions)

    s0 = np.array([[0,0]])
    p1 = nonstationary_space.encode(s0)
    x1 = np.linspace(-1,1,100)
    x2 = np.copy(x1)
    X1, X2 = np.meshgrid(x1,x2)
    xs = np.vstack((X1.flatten(), X2.flatten())).T
    ps = nonstationary_space.encode(xs)

    sims = np.einsum('nd,jd->nj', ps, p1)
    import matplotlib.pyplot as plt
    plt.matshow(sims.reshape((len(x1),len(x2))))
    plt.show()


if __name__=='__main__':
#     test_1d()
#     test_2d()
#     test_2d_pt()
#     test_ssp_regions()
#     plot_similarities()
    plot_regions()
