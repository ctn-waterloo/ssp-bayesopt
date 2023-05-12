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

class DiscreteContextDomain(Domain):
    def __init__(self, bounds):
        self.bounds = bounds[:-1,:]
        self.context_size = int(bounds[-1,1])
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
        sample_contexts = np.random.randint(self.context_size, size=(num_points,1))
        return np.hstack([sample_points, sample_contexts])


class MultiTrajectoryDomain(Domain):
    def __init__(self, n_agents:int, trajectory_length:int, spatial_dim:int, 
                 bounds:np.ndarray,
                 goals:np.ndarray=None):
   
        self.n_agents=n_agents
        self.traj_len = trajectory_length
#         self.domain_bounds = bounds
        self.domain_bounds = np.array([bounds[:,0].min() * np.ones(spatial_dim), 
                             bounds[:,1].max() * np.ones(spatial_dim)]).T
 
        self.spatial_dim = spatial_dim
#         assert goals is None or len(goals[0]) == self.spatial_dim, f'Goals: {goals}, expected {self.n_agents * self.spatial_dim} elements.'
        self.goals = list(goals) if not goals is None else None 
        self.sampler = qmc.Sobol(d=self.spatial_dim) 

#         print('dims: ', self.traj_len, self.domain_bounds.shape)
#         print('bounds: ', self.domain_bounds)
#         exit()


    def sample(self, num_points:int=10, method:str='sobol'):
        if not method == 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        ### end if

        num_samples = num_points * self.traj_len * self.n_agents
        u_sample_points = self.sampler.random(num_samples)
        sample_points = qmc.scale(u_sample_points,
                                  self.domain_bounds[:,0], 
                                  self.domain_bounds[:,1])
        
        if not self.goals is None:
            # Goal noise: 0.1 * U[-1,1]
            nx = 0.1
            sample_points = sample_points.reshape((num_points, 
                                                   self.traj_len,
                                                   self.spatial_dim,
                                                   self.n_agents)
                                                )
            for i in range(self.n_agents):
                noise = np.random.uniform(-1,1, size=(num_points,
                                                      self.spatial_dim)
                                    )
                # Select the right goal
                # Add the noise
                goal_index =(2*i+1) % len(self.goals)
                goal_i = np.array(self.goals[goal_index])
                sample_points[:,-1,:,i] = goal_i + nx * noise
            ### end for
        ### end if

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

