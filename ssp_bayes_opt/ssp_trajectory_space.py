from . import sspspace


class SSPTrajectorySpace:
    '''
    Converts sampled time-varying signals into an SSP representation
    '''
    def __init__(self, spatial_dim:int = 1, basis_space:SSPSpace = None):
        '''
        Initialize the agent with a specific basis representation.

        Parameters:
        -----------
        spatial_dim : int
            The number of dimensions in the spatial component time-varying 
            signal. The number of dimensions of the ssp space needs to be
            spatial_dim + 1, to account for time.

        basis_space : SSPSpace
            The SSP space representing the axes vectors that will be used
            to encode the trajectory.

        Returns:
        --------
            Constructed object
        '''
        self.dims = spatial_dim + 1
        self.ssp_space = basis_space

    def update_lengthscale(self, scale:np.ndarray):
        '''
        Parameters:
        -----------
        scale : np.ndarray 
            The scale parameter(s) used by the underlying ssp encoding
        '''
        self.ssp_space.update_lengthscale(scale)

    def encode(self, x:np.ndarray):
        '''
        Transforms a trajectory into an SSP.

        Parameters:
        -----------
        x : np.ndarray
            A (num trajectories, num_time_samples, spatial_dim) array of
            trajectories to be encoded. 

        Returns:
        ssp_trajs : np.ndarray
            A (num_trajectories, ssp_dim) array of trajectories represented
            as ssps.
        '''

        ssp_trajs = np.fromiter(
                        (np.sum(self.ssp_space.encode(x), axis=0) for x in x), 
                        float
                    )
        return ssp_trajs

    def decode(self, ssp):
        raise NotImplementedError('still implementing decoding code')



