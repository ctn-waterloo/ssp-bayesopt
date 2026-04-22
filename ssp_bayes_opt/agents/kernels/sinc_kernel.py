import numpy as np

from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, StationaryKernelMixin, NormalizedKernelMixin, _check_length_scale

def d_sinc(x):
    x = np.asanyarray(x)
    y = np.where(x == 0, 1.0e-20, x)
    return (np.cos(np.pi * y)/y) - (np.sin(np.pi * y)/(np.pi * y**2))

class SincKernel(StationaryKernelMixin, NormalizedKernelMixin, Kernel):
    '''
    A sinc-function kernel. 

    The SincKernel is a stationary kernel.  It is parameterized by a
    length scale parameter 

    Parameters:
    -----------
    length_scale: float or ndarray of shape (n_features,), default=1.0
        The length scale of the kernel. If a float, an isotropic kernel is
        used.  If an array, an anisotropic kernel is used where each 
        dimension of l defines the length-scale of the respective feature
        dimensions

    length_scale_bounds: pair of floats >= 0 or 'fixed', default=(1e-5,1e5)
        The lower and upper bound of 'length_scale'. If set to 'fixed', 
        length_scale cannot be changed during hyperparameter tuning.
    '''

    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5,1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def anisotropic(self):
        return np.iterable(self.length_scale) and len(self.length_scale) > 1

    @property
    def hyperparameter_length_scale(self):
        if self.anisotropic:
            return Hyperparameter(
                'length_scale', 
                'numeric', 
                self.length_scale_bounds, 
                len(self.length_scale)
            )
        return Hyperparameter('length_scale', 'numeric', self.length_scale_bounds)
    def __call__(self, X, Y=None, eval_gradient=False):
        '''
        Return the kernel k(X,Y) and optimally it's gradient

        Parameters
        ----------
        X : ndarray of shape (n_samples_X, n_features)
        Y : ndarray of shape (n_samples_Y, n_features). If None, k(X,X) 
            evaluated instead.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims)
        '''

        X = np.atleast_2d(X)
        length_scale = _check_length_scale(X, self.length_scale)

        if Y is None:
            dists = (X[:,None,:] - X[None,:,:]) / length_scale
            K = np.prod(np.sinc(dists), axis=-1)
        else:
            if eval_gradient:
                raise ValueError('Gradient can only be evaluted when Y is None')
            dists = (X[:,None,:] - Y[None,:,:]) / length_scale
            K = np.prod(np.sinc(dists), axis=-1)

        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            elif not self.anisotropic or length_scale.shape[0] == 1:
                K_gradient = np.zeros((X.shape[0], 
                                       X.shape[0], 
                                       1))
                sinc_mat = np.sinc(dists) 
                d_sinc_mat = d_sinc(dists)
                for x_idx in range(X.shape[1]):
                    temp = np.prod(sinc_mat[:,:,:x_idx], axis=-1)
                    temp *= np.prod(sinc_mat[:,:,x_idx+1:], axis=-1)
                    temp *= d_sinc_mat[:,:,x_idx] * (-dists[:,:,x_idx]/(length_scale**2))
                    K_gradient[:,:,0] += temp
                return K, K_gradient
            elif self.anisotropic:
                # we need to compute the pairwise dimension-wise distances.
                K_gradient = np.zeros((X.shape[0],
                                       X.shape[0],
                                       len(length_scale)))
                sinc_mat = np.sinc(dists)
                d_sinc_mat = d_sinc(dists)
                for l_idx in range(len(length_scale)):
                    K_gradient[:,:,l_idx] = np.prod(sinc_mat[:,:,:l_idx], axis=-1)
                    K_gradient[:,:,l_idx] *= np.prod(sinc_mat[:,:,l_idx+1:], axis=-1)
                    K_gradient[:,:,l_idx] *= d_sinc_mat[:,:,l_idx]
                    K_gradient[:,:,l_idx] *= -dists[:,:,l_idx]/(length_scale[l_idx]**2)
                    return K, K_gradient

        else:
            return K

    def __repr__(self):
        if self.anisotropic:
            return '{0}(length_scale=[{1}]'.format(
                self.__class__.__name__,
                ', '.join(map('{0:.3g}'.format, self.length_scale)),
            )
        else:
            return '{0}(length_scale={1:.3g}'.format(
               self.__class__.__name__, np.ravel(self.length_scale)[0]
            )


