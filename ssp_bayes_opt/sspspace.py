import numpy as np
from scipy.stats import qmc
from scipy.stats import special_ortho_group
from scipy.optimize import minimize

class SSPSpace:
    def __init__(self, domain_dim: int, ssp_dim: int, axis_matrix=None, phase_matrix=None,
                 domain_bounds=None, length_scale=1):

        self.domain_dim = domain_dim
        self.ssp_dim = ssp_dim
        self.length_scale = length_scale * np.ones((self.domain_dim,1))
        
        if domain_bounds is not None:
            assert domain_bounds.shape[0] == domain_dim
        
        self.domain_bounds = domain_bounds
        
        if (axis_matrix is None) & (phase_matrix is None):
            raise RuntimeError("SSP spaces must be defined by either a axis matrix or phase matrix. Use subclasses to construct spaces with predefined axes.")
        elif (phase_matrix is None):
            assert axis_matrix.shape[0] == ssp_dim
            assert axis_matrix.shape[1] == domain_dim
            self.axis_matrix = axis_matrix
            self.phase_matrix = (-1.j*np.log(np.fft.fft(axis_matrix,axis=0))).real
        elif (axis_matrix is None):
            assert phase_matrix.shape[0] == ssp_dim
            assert phase_matrix.shape[1] == domain_dim
            self.phase_matrix = phase_matrix
            self.axis_matrix = np.fft.ifft(np.exp(1.j*phase_matrix), axis=0).real
            
    def update_lengthscale(self, scale):
        self.length_scale = scale
        
    def optimize_lengthscale(self, init_xs, init_ys):
        ls_0 = self.length_scale
        self.length_scale = np.ones((self.domain_dim,1))
        
        def min_func(length_scale):
            init_phis = self.encode(init_xs/ length_scale)
            W = np.linalg.pinv(init_phis.T) @ init_ys
            mu = np.dot(init_phis.T,W)
            diff = init_ys - mu.T
            err = np.sum(np.power(diff, 2))
            return err

        retval = minimize(min_func, x0=ls_0, method='L-BFGS-B', bounds = self.domain_dim*[(1e-8,1e5)])
        self.length_scale = retval.x.reshape(-1,1)
    
    def encode(self,x):
        x= x.reshape(self.domain_dim, -1)
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ (x / self.length_scale) ), axis=0 ).real
        return data
    
    def encode_and_deriv(self,x):
        x= x.reshape(self.domain_dim, -1)
        data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ (x / self.length_scale) ), axis=0 ).real
        ddata = np.fft.ifft( 1.j * (self.phase_matrix / self.length_scale) @ np.exp( 1.j * self.phase_matrix @ (x / self.length_scale) ), axis=0 ).real
        return data, ddata
    
    def encode_fourier(self,x):
        x= x.reshape(self.domain_dim, -1)
        data =  np.exp( 1.j * self.phase_matrix @ (x / self.length_scale) )
        return data
    
    # def encode_as_SSP(self,x):
    #     assert x.shape[0] == self.domain_dim
    #     data = np.fft.ifft( np.exp( 1.j * self.phase_matrix @ x / self.length_scale ), axis=0 ).real
    #     return SSP(data,self)
    
 
    def decode(self,ssp,method='from-set',
               num_sample_pts=10000,from_set_method='grid',num_init_pts =10): # other args for specfic methods
        if method=='least-squares':
            # problems due to complex log
            x = np.linalg.lstsq(self.phase_matrix, (1.j*np.log(np.fft.fft(ssp,axis=0))).real)[0]
            #raise NotImplementedError()
            #fssp = np.fft.fft(ssp,axis=0)
            #x = np.linalg.lstsq(np.tile(self.phase_matrix,(2,1)), np.hstack([np.arccos(fssp.real), np.arcsin(fssp.imag)]))
            return x
        elif method=='from-set': ## ONLY ONE THAT WORKS WELL
            sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_sample_pts,from_set_method)
            sims = sample_ssps.T @ ssp
            return sample_points[:,np.argmax(sims)]
        elif method=='grad_descent':
            sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_init_pts) 
            sims = sample_ssps.T @ ssp
            x = sample_points[:,np.argmax(sims)]
            fssp = np.fft.fft(ssp,axis=0)
            for j in range(10):
                grad = (1.j * self.phase_matrix.T * np.exp(1.j * self.phase_matrix @ x)) @ fssp
                x = x - 0.1*grad.real
            return x
        elif method=='nonlin-reg':
            sample_ssps, sample_points = self.get_sample_pts_and_ssps(num_init_pts) 
            sims = sample_ssps.T @ ssp
            x = sample_points[:,np.argmax(sims)]
            fssp = np.fft.fft(ssp,axis=0)
            dy = np.hstack([fssp.real, fssp.imag])
            for j in range(10):
                J = np.vstack([self.phase_matrix * np.sin(self.phase_matrix @ x).reshape(-1,1),
                               -self.phase_matrix * np.cos(self.phase_matrix @ x).reshape(-1,1)])
                soln = np.linalg.pinv(J.T @ J) @ J.T @ dy
                x = x + soln
            return x
        else:
            raise NotImplementedError()
        
    def clean_up(self,ssp,method='from-set'):
        if method=='least-squares':
            x = self.decode(ssp,method)
            return self.encode(x)
        elif method=='from-set':
            sample_ssps = self.get_sample_ssps(500)
            sims = sample_ssps.T @ ssp
            return sample_ssps[:,np.argmax(sims)]
        else:
            raise NotImplementedError()
        
    def get_sample_points(self,num_points,method='grid'):
        if self.domain_bounds is None:
            bounds = np.vstack([-10*np.ones(self.domain_dim), 10*np.ones(self.domain_dim)]).T
        else:
            bounds = self.domain_bounds
        if method=='grid':
            n_per_dim = int(num_points**(1/self.domain_dim))
            xs = np.linspace(bounds[:,0],bounds[:,1],n_per_dim)
            xxs = np.meshgrid(*[xs[:,i] for i in range(self.domain_dim)])
            return np.array([x.reshape(-1) for x in xxs])
        elif method=='sobol':
            sampler = qmc.Sobol(d=self.domain_dim) 
            lbounds = bounds[:,0]
            ubounds = bounds[:,1]
            u_sample_points = sampler.random(num_points)
            sample_points = qmc.scale(u_sample_points, lbounds, ubounds)
        else:
            raise NotImplementedError()
        return sample_points.T 
        
    
    def get_sample_ssps(self,num_points): 
        sample_points = self.get_sample_points(num_points)
        sample_ssps = self.encode(sample_points)
        return sample_ssps
    
    def get_sample_pts_and_ssps(self,num_points,method='grid'): 
        sample_points = self.get_sample_points(num_points,method)
        sample_ssps = self.encode(sample_points)
        return sample_ssps, sample_points
    
    def normalize(self,ssp):
        return ssp/np.sqrt(np.sum(ssp**2))
    
    def make_unitary(self,ssp):
        fssp = np.fft.fft(ssp)
        fssp = fssp/np.sqrt(fssp.real**2 + fssp.imag**2)
        return np.fft.ifft(fssp).real  
    
    def identity(self):
        s = np.zeros(self.ssp_dim)
        s[0] = 1
        return s
            
class RandomSSPSpace(SSPSpace):
    def __init__(self, domain_dim: int, ssp_dim: int,  domain_bounds=None, length_scale=1, rng=np.random.default_rng()):
#         partial_phases = rng.random.rand(ssp_dim//2,domain_dim)*2*np.pi - np.pi
        partial_phases = rng.random((ssp_dim//2,domain_dim))*2*np.pi - np.pi
        axis_matrix = _constructaxisfromphases(partial_phases)
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
        
class HexagonalSSPSpace(SSPSpace):
    def __init__(self,  domain_dim:int,ssp_dim: int=151, n_rotates:int=5, n_scales:int=5, 
                 scale_min=2*np.pi/np.sqrt(6) - 0.5, scale_max=2*np.pi/np.sqrt(6) + 0.5,
                 domain_bounds=None, length_scale=1):
        #if (n_rotates==5) & (n_scales==5) & (ssp_dim!=151): # user wants to define ssp with total dim, not number of simplex rotates and scales
        n_rotates = np.max([1,int(np.sqrt((ssp_dim-1)/6))])
        n_scales = n_rotates
            
        phases_hex = np.hstack([np.sqrt(1+ 1/domain_dim)*np.identity(domain_dim) - (domain_dim**(-3/2))*(np.sqrt(domain_dim+1) + 1),
                         (domain_dim**(-1/2))*np.ones((domain_dim,1))]).T
        
        self.grid_basis_dim = domain_dim + 1
        self.num_grids = n_rotates*n_scales

        scales = np.linspace(scale_min,scale_max,n_scales)
        phases_scaled = np.vstack([phases_hex*i for i in scales])
        
        if (n_rotates==1):
            phases_scaled_rotated = phases_scaled
        elif (domain_dim==1):
            scales = np.linspace(scale_min,scale_max,n_scales+n_rotates)
            phases_scaled_rotated = np.vstack([phases_hex*i for i in scales])
        elif (domain_dim == 2):
            angles = np.linspace(0,2*np.pi/3,n_rotates)
            R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                        np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        else:
            R_mats = special_ortho_group.rvs(domain_dim, size=n_rotates)
            phases_scaled_rotated = (R_mats @ phases_scaled.T).transpose(0,2,1).reshape(-1,domain_dim)
        
        axis_matrix = _constructaxisfromphases(phases_scaled_rotated)
        ssp_dim = axis_matrix.shape[0]
        super().__init__(domain_dim,ssp_dim,axis_matrix=axis_matrix,
                       domain_bounds=domain_bounds,length_scale=length_scale)
  
    def sample_grid_encoders(self, n):
        sample_pts = self.get_sample_points(n)
        N = self.num_grids
        if N < n:
            sorts = np.hstack([np.arange(N), np.random.randint(0, N - 1, size = n - N)])
        else:
            sorts = np.arange(n)
        encoders = np.zeros((self.ssp_dim,n))
        for i in range(n):
            sub_mat = _get_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            proj_mat = _proj_sub_SSP(sorts[i],N,sublen=self.grid_basis_dim)
            sub_space = SSPSpace(self.domain_dim,2*self.grid_basis_dim + 1, axis_matrix= sub_mat @ self.axis_matrix)
            encoders[:,i] = N * proj_mat @ sub_space.encode(sample_pts[:,i])
        return encoders

    
    
def _constructaxisfromphases(K):
    d = K.shape[0]
    F = np.ones((d*2 + 1,K.shape[1]), dtype="complex")
    F[0:d,:] = np.exp(1.j*K)
    F[-d:,:] = np.flip(np.conj(F[0:d,:]),axis=0)
    axes =  np.fft.ifft(np.fft.ifftshift(F,axes=0),axis=0).real
    return axes

def _get_sub_FourierSSP(n, N, sublen=3):
    # Return a matrix, \bar{A}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \bar{A}_n F{S_{total}} = F{S_n}
    # i.e. pick out the sub vector in the Fourier domain
    tot_len = 2*sublen*N + 1
    FA = np.zeros((2*sublen + 1, tot_len))
    FA[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FA[sublen, sublen*N] = 1
    FA[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FA

def _get_sub_SSP(n,N,sublen=3):
    # Return a matrix, A_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # A_n S_{total} = S_n
    # i.e. pick out the sub vector in the time domain
    tot_len = 2*sublen*N + 1
    FA = _get_sub_FourierSSP(n,N,sublen=sublen)
    W = np.fft.fft(np.eye(tot_len))
    invW = np.fft.ifft(np.eye(2*sublen + 1))
    A = invW @ np.fft.ifftshift(FA) @ W
    return A.real

def _proj_sub_FourierSSP(n,N,sublen=3):
    # Return a matrix, \bar{B}_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n \bar{B}_n F{S_{n}} = F{S_{total}}
    # i.e. project the sub vector in the Fourier domain such that summing all such projections gives the full vector in Fourier domain
    tot_len = 2*sublen*N + 1
    FB = np.zeros((2*sublen + 1, tot_len))
    FB[0:sublen, sublen*n:sublen*(n+1)] = np.eye(sublen)
    FB[sublen, sublen*N] = 1/N # all sub vectors have a "1" zero freq term so scale it so full vector will have 1 
    FB[sublen+1:, tot_len - np.arange(sublen*(n+1),sublen*n,-1)] = np.eye(sublen)
    return FB.T

def _proj_sub_SSP(n,N,sublen=3):
    # Return a matrix, B_n
    # Consider the multi scale representation (S_{total}) and sub vectors (S_n) described in the paper 
    # Then
    # \sum_n B_n S_{n} = S_{total}
    # i.e. project the sub vector in the time domain such that summing all such projections gives the full vector
    tot_len = 2*sublen*N + 1
    FB = _proj_sub_FourierSSP(n,N,sublen=sublen)
    invW = np.fft.ifft(np.eye(tot_len))
    W = np.fft.fft(np.eye(2*sublen + 1))
    B = invW @ np.fft.ifftshift(FB) @ W
    return B.real
