import numpy as np
from scipy.stats import qmc


def sample_domain(bounds, samples_per_dim):
    
    num_points = np.prod(samples_per_dim)
    sampler = qmc.Sobol(d=bounds.shape[0]) 
    lbounds = bounds[:,0]
    ubounds = bounds[:,1]
    u_sample_points = sampler.random(num_points)
    sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
    return sample_points
