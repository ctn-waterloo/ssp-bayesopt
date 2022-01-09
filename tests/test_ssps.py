import numpy as np
import scipy

import ssp_bayes_opt

domain_dim = 2
bounds = 15*np.array([[-1,1],[-1,1]])
ssp_space = ssp_bayes_opt.sspspace.HexagonalSSPSpace(domain_dim,ssp_dim=151, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=bounds, length_scale=1)
S0 = ssp_space.encode(np.array([1.3,-3.4]))

print(ssp_space.decode(S0, method='from-set'))
