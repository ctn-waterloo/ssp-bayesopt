from .sspspace import HexagonalSSPSpace
import functools

import numpy as np

class Region:
    def __init__(self, bounds, scales):
        self.bounds = bounds
        self.scales = scales

    def __call__(self, x):
        np.any(x > self.bounds[:,0] and x < self.bounds[:,1])

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
        print(x.shape)

        freqs = self.ssp_space.encode_fourier(x)
        assert freqs.shape[0] == x.shape[0]

        # Compute membership and weight masks
        memberships = np.array([r(x) for r in self.regions])
        memberships = np.sqrt(memberships / np.sum(memberships))
        weighted_masks = np.einsum('dr,rn->drn',self.masks, memberships)

        # Weigh frequency components by masks
        weighted_components = np.einsum('nd,drn->ndr',freqs, weighted_masks)
    
        nonstationary_comps = np.sum(weighted_components, axis=2)
        data = np.fft.ifft(nonstationary_comps, axis=1 ).real
        return data

    def make_mask(self):
        scale_mask = np.ones((self.ssp_space.n_scales, len(self.regions)))

        for r_idx, r in enumerate(self.regions):
            scale_mask[r.scales, r_idx] = 0
        mask_half = np.tile(np.repeat(scale_mask, 3, axis=0),
                            (self.ssp_space.n_rotates, 1))
        mask = np.vstack([np.ones((1,len(self.regions))),
                          mask_half,
                          np.flip(mask_half, axis=0)])
        return mask
    ### end make_mask
