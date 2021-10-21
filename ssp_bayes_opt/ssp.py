'''
Handy utils for spatial semantic pointers implemented in 
'''

import numpy as np

from typing import Any, List
import numpy.typing as npt


def fractional_bind(basis:np.ndarray, position:np.ndarray):
    return np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**position, axis=1), axis=0)
    
# Given your axis vectors (as a matrix) and an array of positions, compute SSP rep for all positions
def ssp_vectorized(basis, positions):
    positions = positions.reshape(-1,basis.shape[1])
    S_list = np.zeros((basis.shape[0],positions.shape[0]),dtype=complex)
    for i in np.arange(positions.shape[0]):
        S_list[:,i] = np.fft.ifft(np.prod(np.fft.fft(basis, axis=0)**positions[i,:], axis=1), axis=0)  
    return S_list
    

# Using a matrix of phases to generate axis vectors
def PlaneWaveBasis(K):
    # K is a matrix of phases
    d = K.shape[0]
    n = K.shape[1]
    axes = []
    for i in range(n):
        F = np.ones((d*2 + 1,), dtype="complex")
        F[0:d] = np.exp(1.j*K[:,i])
        F[-d:] = np.flip(np.conj(F[0:d]))
        F = np.fft.ifftshift(F)
#         axis = ssp.SpatialSemanticPointer(data=np.fft.ifft(F).real)
        axis = data=np.fft.ifft(F).real
        # or spa.SemanticPointer if not using nengo_ssp (see github.com/nsdumont/nengo_ssp)
        axes.append(axis)
    return axes

def RandomBasis(dim,d):
    K = np.random.rand(d//2,dim)*2*np.pi - np.pi
    basis = PlaneWaveBasis(K)
    return basis, K

def HexagonalBasis(dim=2,n_rotates=8,n_scales=8,scale_min=0.8, scale_max=3):
    # Create dim axis vectors consisting of multiple sets of hexagonal bases
    K_hex = np.hstack([np.sqrt(1+ 1/dim)*np.identity(dim) - (dim**(-3/2))*(np.sqrt(dim+1) + 1),
                     (dim**(-1/2))*np.ones((dim,1))]).T

    scales = np.linspace(scale_min,scale_max,n_scales)
    K_scales = np.vstack([K_hex*i for i in scales])
    if (dim == 2) and (n_rotates > 1):
        angles = np.linspace(0,2*np.pi/3,n_rotates)
        R_mats = np.stack([np.stack([np.cos(angles), -np.sin(angles)],axis=1),
                    np.stack([np.sin(angles), np.cos(angles)], axis=1)], axis=1)
        K_scale_rotates = (R_mats @ K_scales.T).transpose(0,2,1).reshape(-1,dim)
    elif (dim>1) and (n_rotates>1):
        R_mats = scipy.stats.special_ortho_group.rvs(dim, size=n_rotates)
        K_scale_rotates = (R_mats @ K_scales.T).transpose(0,2,1).reshape(-1,dim)
    else: 
        K_scale_rotates = K_scales
    basis = PlaneWaveBasis(K_scale_rotates)
    return basis, K_scale_rotates



def make_rand_ssp(data_dim : int, ptr_dim : int) -> np.ndarray:
    '''
    Creates data_dim SSPs using the random generation method developed in 
    Brent Kormer's thesis. 
    '''
    assert data_dim > 0, f'Data dimension must be greater than zero, got {data_dim}.'
    assert ptr_dim > 0, f'Data dimension must be greater than zero, got {ptr_dim}.'

    retval = np.vstack([make_good_unitary(ptr_dim) for _ in range(data_dim)])

    assert retval.shape == (data_dim, ptr_dim), f'Expected SSP array to be ({data_dim},{ptr_dim}), got {retval.shape}'

    return retval

def encode(s : np.ndarray, e: float) -> np.ndarray:
    x = np.fft.ifft(np.fft.fft(s)**e).real
    return x


def bind(s1 : np.ndarray, s2 : np.ndarray) -> np.ndarray:
    n = len(s1)
    assert len(s1) == len(s2), f'SSPs should have same length, but have {len(s1)} and {len(s2)}.'
    return np.fft.irfft(np.fft.rfft(s1)*np.fft.rfft(s2), n=n)

# Does the two part of binding x dimensions together
def vector_encode(ptrs : np.ndarray, xs : npt.ArrayLike) -> np.ndarray:
    (data_dim, ptr_dim) = ptrs.shape
    (num_data, data_dim) = xs.shape

    S_list = np.zeros((num_data, ptr_dim))
    for x_idx, x in enumerate(xs):
        ps = np.fft.fft(ptrs,axis=1)
        S_list[x_idx,:] = np.fft.ifft(np.prod(np.fft.fft(ptrs.T, axis=0) ** x, axis=1), axis=0).T
    # end for
    return S_list

def to_unitary(v : np.ndarray) -> np.ndarray:
    fft_val = np.fft.fft(v)
    fft_imag = fft_val.imag
    fft_real = fft_val.real
    fft_norms = np.sqrt(fft_imag ** 2 + fft_real ** 2)
    invalid = fft_norms <= 0.
    fft_val[invalid] = 1.0
    fft_norms[invalid] = 1.0
    fft_unit = fft_val / fft_norms
    return np.array((np.fft.ifft(fft_unit, n=len(v))).real)

def make_good_unitary(dim : int, eps: float=1e-3 , rng: Any =np.random ) -> np.ndarray:
    a = rng.rand((dim - 1) // 2)
    sign = rng.choice((-1, +1), len(a))
    phi = sign * np.pi * (eps + a * (1 - 2 * eps))
    assert np.all(np.abs(phi) >= np.pi * eps)
    assert np.all(np.abs(phi) <= np.pi * (1 - eps))

    fv = np.zeros(dim, dtype='complex64')
    fv[0] = 1
    fv[1:(dim + 1) // 2] = np.cos(phi) + 1j * np.sin(phi)
    fv[-1:dim // 2:-1] = np.conj(fv[1:(dim + 1) // 2])
    if dim % 2 == 0:
        fv[dim // 2] = 1

    assert np.allclose(np.abs(fv), 1)
    v = np.fft.ifft(fv)
    
    v = v.real
    assert np.allclose(np.fft.fft(v), fv)
    assert np.allclose(np.linalg.norm(v), 1)
    return v


# This code from Nicole Dumont.  See her repo:
# http://github.com/nsdumont/nengo_ssp 
def PlaneWaveBasis(K : np.ndarray) -> List:
    # K is a matrix of phases
    d = K.shape[0]
    n = K.shape[1]
    axes = []
    for i in range(n):
        F = np.ones((d*2 + 1,), dtype="complex")
        F[0:d] = np.exp(1.j*K[:,i])
        F[-d:] = np.flip(np.conj(F[0:d]))
        F = np.fft.ifftshift(F)
        axis = np.fft.ifft(F).real
        axes.append(axis)
    return axes

def make_hex_unitary(data_dim : int, n_scales: int =1, n_rotates: int =1, scale_min : float=0.8, scale_max : float= 3.4) -> np.ndarray:
    assert data_dim > 0, f'Must have positive data dimension, received {data_dim}.'

    scales = np.linspace(scale_min, scale_max, n_scales)

    basis_vertices = None
    if data_dim == 1:
        vertex = np.array([1])
        basis_vertices = np.vstack([vertex * i for i in scales])
    elif data_dim == 2:
        vertices = np.array([[0,1], [np.sqrt(3)/2,-0.5], [-np.sqrt(3)/2, -0.5]])
        scaled_vertices = np.vstack([vertices * i for i in scales])
        angles = np.arange(0, n_rotates) * np.pi / (3  * n_rotates)
        rotation_matrices = np.stack([np.stack([np.cos(angles), -np.sin(angles)], axis=1),
                                      np.stack([np.sin(angles), np.cos(angles)], axis=1)],
                                     axis=1)
        basis_vertices = (rotation_matrices @ scaled_vertices.T).transpose(1, 2, 0).T.reshape(-1, data_dim)
    elif data_dim == 3:
        vertices = np.array([[0,0,1], 
                             [-np.sqrt(2/9), -np.sqrt(2/3), -1/3],
                             [-np.sqrt(2/9), np.sqrt(2/3), -1/3],
                             [np.sqrt(8/9), 0, -1/3]])
        scaled_vertices = np.vstack([vertices * i for i in scales])
        if n_rotates == 1:
            basis_vertices = np.copy(scaled_vertices)
        else:
            sq_n_rotates = int(np.sqrt(n_rotates))
            angles = np.arange(0, sq_n_rotates) * 2 * np.pi / sq_n_rotates
            vec = np.zeros(len(angles))
            Ry = np.stack([
                            np.stack([np.cos(angles), vec, -np.sin(angles)], axis=1),
                            np.stack([vec, vec+1, vec], axis=1),
                            np.stack([-np.sin(angles), vec, np.cos(angles)], axis=1)],
                          axis=1)
            Rz = np.stack([
                            np.stack([np.cos(angles), -np.sin(angles), vec], axis=1),
                            np.stack([np.sin(angles), np.cos(angles), vec], axis=1),
                            np.stack([vec, vec, vec+1], axis=1)],
                          axis=1)
            basis_vectors = (Rz @ (Ry @ scaled_verticies.T).transpose(1,2,0).T.reshape(-1,data_dim).T).transpose(0,2,1).reshape(-1,data_dim)
        ### end if n_rotates == 1
    elif data_dim == 4:
        vertices = np.array([[1 / np.sqrt(10), 1 / np.sqrt(6), 1 / np.sqrt(3), 1],
                             [1 / np.sqrt(10), 1 / np.sqrt(6), -2 / np.sqrt(3), 0],
                             [1 / np.sqrt(10), -np.sqrt(3/2), 0, 0],
                             [-2 * np.sqrt(2/5), 0, 0, 0]])
        scaled_vertices = np.vstack([vertices * i for i in scales])
        if n_rotates == 1:
            basis_vertices = np.copy(scaled_vertices)
        else:
            raise NotImplementedError(f'Rotations not implemented for {data_dim}d data')
        ### end if
    else:
        raise NotImplementedError(f'{data_dim} Hexagonal SSPs not implemented')
    ### end if data_dim == 
        
    basis = PlaneWaveBasis(basis_vertices)
    if data_dim == 1:
        basisv = np.copy(basis)
    else:
        basisv = np.vstack([v for v in basis])
    return basisv


if __name__ == '__main__':
    ptrs = make_hex_unitary(2)

    xs = np.array([[-1,1]])
    print(vector_encode(ptrs, xs))

