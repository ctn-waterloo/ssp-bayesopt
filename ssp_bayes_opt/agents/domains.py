import numpy as np
from scipy.stats import qmc


class Domain:
    """Abstract base class for function domains."""

    def sample(self, num_points: int = 10, method: str = 'sobol') -> np.ndarray:
        """Draw num_points samples from the domain.

        Parameters
        ----------
        num_points : int
        method : str
            Sampling strategy. Currently only 'sobol' is supported by subclasses.

        Returns
        -------
        np.ndarray, shape (num_points, domain_dim)
        """
        pass


class BoundedDomain(Domain):
    """Axis-aligned rectangular domain sampled with Sobol sequences."""

    def __init__(self, bounds: np.ndarray):
        self.bounds = bounds
        self.data_dim = self.bounds.shape[0]
        self.sampler = qmc.Sobol(d=self.data_dim)

    def sample(self, num_points: int = 10, method: str = 'sobol') -> np.ndarray:
        if method != 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        u_sample_points = self.sampler.random(num_points)
        sample_points = qmc.scale(u_sample_points,
                                  self.bounds[:, 0], self.bounds[:, 1])
        return sample_points


class MultiTrajectoryDomain(Domain):
    """Domain for multi-agent trajectory optimization.

    Samples flattened trajectories of shape (traj_len * spatial_dim * n_agents,).
    Optionally pins the last timestep of each agent's trajectory near a goal.
    """

    def __init__(self, n_agents: int, trajectory_length: int, spatial_dim: int,
                 bounds: np.ndarray, goals: np.ndarray = None):
        self.n_agents = n_agents
        self.traj_len = trajectory_length
        self.domain_bounds = np.array([
            bounds[:, 0].min() * np.ones(spatial_dim),
            bounds[:, 1].max() * np.ones(spatial_dim)
        ]).T
        self.spatial_dim = spatial_dim
        self.goals = list(goals) if goals is not None else None
        self.sampler = qmc.Sobol(d=self.spatial_dim)

    def sample(self, num_points: int = 10, method: str = 'sobol') -> np.ndarray:
        if method != 'sobol':
            raise NotImplementedError(f'{method} not implemented')

        num_samples = num_points * self.traj_len * self.n_agents
        u_sample_points = self.sampler.random(num_samples)
        sample_points = qmc.scale(u_sample_points,
                                  self.domain_bounds[:, 0],
                                  self.domain_bounds[:, 1])

        if self.goals is not None:
            nx = 0.1
            sample_points = sample_points.reshape(
                (num_points, self.traj_len, self.spatial_dim, self.n_agents))
            for i in range(self.n_agents):
                noise = np.random.uniform(-1, 1, size=(num_points, self.spatial_dim))
                goal_index = (2 * i + 1) % len(self.goals)
                goal_i = np.array(self.goals[goal_index])
                sample_points[:, -1, :, i] = goal_i + nx * noise

        return sample_points.reshape(
            num_points, self.traj_len * self.spatial_dim * self.n_agents)


class TrajectoryDomain(Domain):
    """Domain for single-agent trajectory optimization.

    Samples flattened trajectories of shape (traj_len * spatial_dim,).
    """

    def __init__(self, trajectory_length: int, spatial_dim: int, bounds: np.ndarray):
        self.traj_len = trajectory_length
        self.domain_bounds = np.array([
            bounds[:, 0].min() * np.ones(spatial_dim),
            bounds[:, 1].max() * np.ones(spatial_dim)
        ]).T
        self.spatial_dim = spatial_dim
        self.sampler = qmc.Sobol(d=self.spatial_dim)

    def sample(self, num_points: int = 10, method: str = 'sobol') -> np.ndarray:
        if method != 'sobol':
            raise NotImplementedError(f'{method} not implemented')
        u_sample_points = self.sampler.random(num_points * self.traj_len)
        sample_points = qmc.scale(u_sample_points,
                                  self.domain_bounds[:, 0],
                                  self.domain_bounds[:, 1])
        return sample_points.reshape(num_points, self.traj_len * self.spatial_dim)


class TargetDefinedDomain(Domain):
    """Domain whose sample() method is delegated to the target object itself.

    Used with NAS-Bench and MCBO benchmarks where the search space is discrete
    and the target object owns the sampling logic.
    """

    def __init__(self, target):
        self.target = target

    def sample(self, num_points: int = 10, **kwargs) -> np.ndarray:
        return self.target.sample(num_points, **kwargs)
