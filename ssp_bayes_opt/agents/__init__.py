from .ssp_agent import SSPAgent
from .ssp_traj_agent import SSPTrajectoryAgent
from .ssp_multi_agent import SSPMultiAgent
from .ssp_context_agent import SSPContinuousContextualAgent, SSPDiscreteContextualAgent
from .gp_agent import GPAgent

from .. import sspspace

from . import domains


def factory(data_dim, init_xs, init_ys, **kwargs):

    if agent_type=='ssp-hex':
        ssp_space = sspspace.HexagonalSSPSpace(data_dim, **kwargs)
        agt = SSPAgent(init_xs, init_ys,ssp_space) 
    elif agent_type=='ssp-rand':
        ssp_space = sspspace.RandomSSPSpace(data_dim, **kwargs)
        agt = SSPAgent(init_xs, init_ys,ssp_space) 
    elif agent_type=='gp':
        agt = GPAgent(init_xs, init_ys,**kwargs) 
    elif agent_type=='static-gp':
        agt = GPAgent(init_xs, init_ys, updating=False, **kwargs) 
    elif agent_type=='ssp-traj':
        agt = SSPTrajectoryAgent(init_points, self.target, **kwargs) 
        init_xs = agt.init_xs
        init_ys = agt.init_ys
    else:
        raise NotImplementedError()

    return agt, init_xs, init_ys
