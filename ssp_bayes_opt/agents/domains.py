import numpy as np

from scipy.stats import qmc

class Domain:
    def __init__(self):
        '''
        Parent class to define function domains
        '''
        pass

    def sample(self, num_points:int=10, method:str='sobol') -> np.ndarray:
        '''
        Parameters
        ----------
        num_points : int
            The number of points that are sampled from the domain.

        methods : {'sobol', 'uniform'}
            The method used to generate sample points.  Default is sobol
            sampling. Also supporting Uniformly random sampling from 
            the domain.
        '''
        pass

class BoundedDomain(Domain):
    def __init__(self, bounds):
        self.bounds = bounds
        self.data_dim = self.bounds.shape[0]
        self.sampler = qmc.Sobol(d=self.data_dim) 
    
    def sample(self, num_points: int=10, method:str='sobol') -> np.ndarray:
        if not method == 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        ### end if
        u_sample_points = self.sampler.random(num_points)
        sample_points = qmc.scale(u_sample_points, 
                                  self.bounds[:,0],
                                  self.bounds[:,1])
        return sample_points
    ### end sample
### end class ###


class MultiTrajectoryDomain(Domain):
    def __init__(self, n_agents:int, trajectory_length:int, spatial_dim:int, 
                 bounds:np.ndarray):
   
        self.n_agents=n_agents
        self.traj_len = trajectory_length
#         self.domain_bounds = bounds
        self.domain_bounds = np.array([bounds[:,0].min() * np.ones(spatial_dim), 
                             bounds[:,1].max() * np.ones(spatial_dim)]).T
 
        self.spatial_dim = spatial_dim
        self.sampler = qmc.Sobol(d=self.spatial_dim) 

#         print('dims: ', self.traj_len, self.domain_bounds.shape)
#         print('bounds: ', self.domain_bounds)
#         exit()


    def sample(self, num_points:int=10, method:str='sobol'):
        if not method == 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        ### end if

        u_sample_points = self.sampler.random(num_points * self.traj_len * self.n_agents)
        sample_points = qmc.scale(u_sample_points,
                                  self.domain_bounds[:,0], 
                                  self.domain_bounds[:,1])

        return sample_points.reshape(num_points,
                                     self.traj_len * self.spatial_dim * self.n_agents)

class TrajectoryDomain(Domain):
    def __init__(self, trajectory_length:int, spatial_dim:int, 
                 bounds:np.ndarray):
        self.traj_len = trajectory_length
#         self.domain_bounds = bounds
        self.domain_bounds = np.array([bounds[:,0].min() * np.ones(spatial_dim), 
                             bounds[:,1].max() * np.ones(spatial_dim)]).T
 
        self.spatial_dim = spatial_dim
        self.sampler = qmc.Sobol(d=self.spatial_dim) 

#         print('dims: ', self.traj_len, self.domain_bounds.shape)
#         print('bounds: ', self.domain_bounds)
#         exit()


    def sample(self, num_points:int=10, method:str='sobol'):
        if not method == 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        ### end if

        u_sample_points = self.sampler.random(num_points * self.traj_len)
        sample_points = qmc.scale(u_sample_points,
                                  self.domain_bounds[:,0], 
                                  self.domain_bounds[:,1])

        return sample_points.reshape(num_points,
                                     self.traj_len * self.spatial_dim)
    ### end sample 

