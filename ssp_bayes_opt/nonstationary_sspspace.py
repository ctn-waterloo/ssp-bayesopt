from .sspspace import HexagonalSSPSpace
import functools

import numpy as np
from scipy.special import softmax

from .util import sample_domain

from sklearn.neural_network import MLPRegressor


class Region:
    def __init__(self, bounds, scales):
        self.bounds = bounds
        self.scales = scales

    def __call__(self, x):
        np.any(x > self.bounds[:,0] and x < self.bounds[:,1])

    def num_regions(self):
        raise NotImplementedError()

class MLPRegion(Region):
    def __init__(self, bounds, memberships, encoder):
        self.bounds = bounds
        self.encoder = encoder
        self.memberships = memberships

        # Sample the bounds densely
        bound_diff = bounds[:,1] - bounds[:,0]
        length_scales = encoder.length_scale
        samples_per_dim = np.ceil(4*np.divide(bound_diff, length_scales)).astype(int)
        domain_samples = sample_domain(bounds, samples_per_dim)
        # Filter based on membership
        region_indicies = np.array([m(domain_samples) for m in memberships]).astype(float).squeeze().T
        encoded_samples = encoder.encode(domain_samples)

        self.mlp = MLPRegressor(max_iter=200, early_stopping=True)
        self.mlp.fit(encoded_samples, region_indicies)

    def __call__(self, x):
        x_f = self.encoder.encode(x)
        return np.minimum(1,np.maximum(0,self.mlp.predict(x_f).T))

    def num_regions(self):
        return len(self.memberships)

class SSPRegion(Region):
    def __init__(self, bounds, memberships, encoder):
        self.bounds = bounds
        self.encoder = encoder
        self.memberships = memberships

        self.scales = np.zeros((encoder.n_scales,len(self.memberships))).astype(int)

        # Sample the bounds densely
        bound_diff = bounds[:,1] - bounds[:,0]
        length_scales = encoder.length_scale
        samples_per_dim = np.ceil(4*np.divide(bound_diff, length_scales)).astype(int)
        domain_samples = sample_domain(bounds, samples_per_dim)
        # Filter based on membership
        region_indicies = [m(domain_samples) for m in memberships]


        # Encode region ssps
        phis = np.zeros((len(memberships), encoder.ssp_dim))
        for r_idx, rs in enumerate(region_indicies):
            sub_samples = domain_samples[rs,:]
            r_phis = self.encoder.encode(sub_samples)
            phis[r_idx,:] = np.mean(r_phis, axis=0) / np.prod(length_scales)

        self.membership_fourier_phis = phis

    def __call__(self, x):
        x_f = self.encoder.encode(x)
        sims = np.real(np.einsum('rd,nd->rn',
                                 self.membership_fourier_phis,
                                 x_f))
        return softmax(np.maximum(0, sims), axis=0)

    def num_regions(self):
        return len(self.memberships)


class IncludeRegion(Region):
    def __call__(self, x):
        (samples, dims) = x.shape
        assert dims == self.bounds.shape[0]
        membership = [np.logical_and( x[:,i] > self.bounds[i,0],
                                      x[:,i] < self.bounds[i,1]) for i in range(dims)]
        return functools.reduce(np.logical_and, membership)

class ExcludeRegion(Region):
    def __call__(self, x):
        (samples, dims) = x.shape
        assert dims == self.bounds.shape[0]
        membership = [np.logical_or( x[:,i] <= self.bounds[i,0],
                                      x[:,i] >= self.bounds[i,1]) for i in range(dims)]
        return functools.reduce(np.logical_or, membership)

class NonstationarySSPSpace:
    def __init__(self, ssp_space, regions):
        self.ssp_space = ssp_space
        self.regions = regions
        self.masks = self.make_mask()

    def encode(self, x):
        # Get the fourier frequency components

        freqs = self.ssp_space.encode_fourier(x)
        assert freqs.shape[0] == x.shape[0]

        # Compute membership and weight masks
        memberships = np.sqrt(self.regions(x))
        weighted_masks = np.einsum('dr,rn->dn',self.masks, memberships)

        # Weigh frequency components by masks
        weighted_components = np.einsum('nd,nd->nd',freqs, weighted_masks.T)
    
#         nonstationary_comps = np.sum(weighted_components, axis=2)
        data = np.fft.ifft(weighted_components, axis=1 ).real
        return data

    def make_mask(self):

        if self.ssp_space.domain_dim == 1:
            n_scales = self.ssp_space.n_scales + self.ssp_space.n_rotates
            n_rotates = 1
        else:
            n_scales = self.ssp_space.n_scales
            n_rotates = self.ssp_space.n_rotates

        scale_mask = np.ones((n_scales, self.regions.num_regions()), dtype=int)

        region_span = n_scales // self.regions.num_regions()
        for r_idx in range(self.regions.num_regions()):
            scale_mask[r_idx*region_span:(r_idx+1)*region_span, r_idx] = 0

        mask_half = np.tile(np.repeat(scale_mask, 
                                      self.ssp_space.domain_dim+1, 
                                      axis=0),
                            (n_rotates, 1))
        mask = np.vstack([np.ones((1,self.regions.num_regions())),
                          mask_half,
                          np.flip(mask_half, axis=0)])
        return mask
    ### end make_mask
