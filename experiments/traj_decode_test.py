import numpy as np
from numpy.polynomial import legendre
def make_trajectory(ts, coeffs):
    return legendre.legval(ts, coeffs).T

import ssp_bayes_opt

x_dim = 2
bounds = 10*np.stack([-np.ones(x_dim), np.ones(x_dim)]).T

n_agents = 3
traj_len = 5

ssp_dim = 487
domain = ssp_bayes_opt.agents.domains.MultiTrajectoryDomain(n_agents, 
                                         traj_len, 
                                         x_dim,
                                         bounds,)
init_xs = domain.sample(10)
myagent = ssp_bayes_opt.agents.SSPMultiAgent(init_xs, np.zeros((init_xs.shape[0],1)),
                                             n_agents, x_dim=x_dim, traj_len=traj_len,
             ssp_dim=ssp_dim,
             domain_bounds=bounds,
             length_scale=[1]*n_agents,
             gamma_c=1.0,
             beta_ucb=np.log(2/1e-6),
             init_pos=None,)
ssp_dim = myagent.ssp_dim

def encodes(agt,x, agt_sps):
    '''
    Translates a trajectory x into an SSP representation.
    

    Parameters:
    -----------
    x : np.ndarray
        A (s, l, d) numpy array specifying s trajectories
        of length l.
    '''
    enc_x = np.atleast_2d(x)
    S = []
    
    enc_x = enc_x.reshape(-1,agt.n_agents,agt.traj_len,agt.x_dim)
    for i in range(agt.n_agents):
        Si = np.zeros((x.shape[0], agt.ssp_dim))
        for j in range(agt.traj_len):
            #print(enc_x.shape)
            #print(enc_x[:,i,j,:].shape)
            #print(self.ssp_x_spaces[i].encode(enc_x[:,i,j,:]).shape)
            Si += agt.ssp_x_spaces[i].encode(enc_x[:,i,j,:])
            # Si += agt.ssp_x_spaces[i].bind(agt.timestep_ssps[j,:], 
            #                        agt.ssp_x_spaces[i].encode(enc_x[:,i,j,:]))
        S.append(Si)
    return S

    
def decode(agt,ssp,agent_sps):
    decoded_traj = np.zeros((agt.n_agents, agt.traj_len, agt.x_dim))
    for i in range(agt.n_agents):
        sspi = agt.ssp_x_spaces[i].bind(agt.ssp_x_spaces[i].invert(agent_sps[i,:]), ssp)
        queries = agt.ssp_x_spaces[i].bind(agt.ssp_t_space.invert(agt.timestep_ssps) , sspi)
        decoded_traj[i,:,:] = agt.ssp_x_spaces[i].decode(queries, 
                                                          method=agt.decoder_method,
                                                          samples=agt.init_samples[i])
        # for j in range(self.traj_len):
        #     query = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , sspi)
        #     decoded_traj[i,j,:] = self.ssp_x_spaces[i].decode(query, method=self.decoder_method,samples=self.init_samples[i])
    return decoded_traj.reshape(-1)

    
def decodep(agt,ssp,agent_sps):
    decoded_traj = np.zeros((agt.n_agents, agt.traj_len, agt.x_dim))
    sspsr = []
    for i in range(agt.n_agents):
        sspi = agt.ssp_x_spaces[i].bind(agt.ssp_x_spaces[i].invert(agent_sps[i,:]), ssp)
        queries = agt.ssp_x_spaces[i].bind(agt.ssp_t_space.invert(agt.timestep_ssps) , sspi)
        sspsr.append(queries)
        # decoded_traj[i,:,:] = agt.ssp_x_spaces[i].decode(queries, 
        #                                                   method=agt.decoder_method,
        #                                                   samples=agt.init_samples[i])
        # for j in range(self.traj_len):
        #     query = self.ssp_x_spaces[i].bind(self.ssp_t_space.invert(self.timestep_ssps[j,:]) , sspi)
        #     decoded_traj[i,j,:] = self.ssp_x_spaces[i].decode(query, method=self.decoder_method,samples=self.init_samples[i])
    return sspsr

import matplotlib.pyplot as plt

domain_dim=2
ssp_space = ssp_bayes_opt.sspspace.HexagonalSSPSpace(domain_dim, 
                        n_rotates=4, 
                        n_scales=7, 
                        scale_min=1,
                        scale_max=3,
                        domain_bounds= np.tile([-1,1],(domain_dim,1)),
                        length_scale=0.15
                        )

import seaborn as sns
sns.set_theme(style="darkgrid")
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)


traj2 = 1.2*np.array([[-0.25,0.5],[-.35,.45],[-.45,.4],[-.6,.3],[-.5,.2],[-.4,.1]])
ssppts2 = np.sum(ssp_space.encode(traj2),axis=0)
levels = np.arange(0.5, 1.7, 0.1) 
ssp_space.similarity_plot(ssppts2,plot_type='contourf',cmap='Blues', levels=levels,alpha=0.5,ax=ax, zorder=5)


traj3 = 1.5*(np.array([[0.6,-.6],[0.55,-0.5],[0.5,-0.4],[0.4,-0.5],[0.34,-0.6]]) + np.array([[-0.2,0.2]]))
ssppts2 = np.sum(ssp_space.encode(traj3),axis=0)
levels = np.arange(0.4, 1.4, 0.1) 
ssp_space.similarity_plot(ssppts3,plot_type='contourf',cmap='Greens', levels=levels,alpha=0.5,ax=ax, zorder=5)

traj1 = 1.2*np.array([[0,0],[.1,0.06],[0.18,.21],[0.2,.35],[.25,.5]])
ssppts1 = np.sum(ssp_space.encode(traj1),axis=0)
#plt.contourf(xx, yy, f, levels, cmap=cmap, alpha=0.5)

levels = np.arange(0.5, 1.7, 0.1) 
#ssp_space.similarity_plot(ssppts1,plot_type='contourf',cmap='Greys', levels=levels,alpha=0.5,ax=ax)
ssp_space.similarity_plot(ssppts1,plot_type='contourf',cmap='Reds', levels=levels,alpha=0.5,ax=ax, zorder=5)

levels = np.arange(0.5, 2.5, 0.1) 
traj4 = np.array([[0.25,-.3],[.35,-.25],[.45,-.2],[.5,-.2],[.7,-.25],[.75,-.1]])
ssppts4 = np.sum(ssp_space.encode(traj4),axis=0)
ssp_space.similarity_plot(ssppts4,plot_type='contourf',cmap='Oranges', levels=levels,alpha=0.5,ax=ax, zorder=5)


ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])
# # Give plot a gray background like ggplot.
# ax.set_facecolor('#EBEBEB')
# # Remove border around plot.
# [ax.spines[side].set_visible(False) for side in ax.spines]
# # Style the grid.
# ax.xaxis.grid(True, zorder=0)
# ax.yaxis.grid(True, zorder=0)
# ax.grid(which='major', color='white', linewidth=1.2)
# ax.grid(which='minor', color='white', linewidth=0.6)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.plot(traj2[:,0],traj2[:,1],'o-',color='blue')
ax.plot(traj3[:,0],traj3[:,1],'s-',color='green')
ax.plot(traj1[:,0],traj1[:,1],'x-',color='red')
ax.plot(traj4[:,0],traj4[:,1],'x-',color='orange')
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])

ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])


from ssp_bayes_opt.sspspace import _get_sub_SSP, _proj_sub_SSP
def sample_grid_encoders(ssp_space, n):
    sample_pts = ssp_space.get_sample_points(n)
    N = ssp_space.num_grids
    sorts = np.random.randint(0,N,size=n)
    
    encoders = np.zeros((ssp_space.ssp_dim,n))
    for i in range(n):
        sub_mat = _get_sub_SSP(sorts[i],N,sublen=ssp_space.grid_basis_dim)
        proj_mat = _proj_sub_SSP(sorts[i],N,sublen=ssp_space.grid_basis_dim)
        sub_space = ssp_bayes_opt.sspspace.SSPSpace(ssp_space.domain_dim,2*ssp_space.grid_basis_dim + 1, axis_matrix= sub_mat @ ssp_space.axis_matrix)
        encoders[:,i] = N * proj_mat @ sub_space.encode(np.atleast_2d(sample_pts[i,:])).reshape(-1)
    return encoders

GC_vecs = sample_grid_encoders(ssp_space, 4)
for i in range(4):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111)
    ssp_space.similarity_plot(GC_vecs[:,i],plot_type='contourf',ax=ax)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
# idx = 0
# for idx in [0,1]:
#     plt.figure()
#     atraj = init_xs[idx,:].reshape(n_agents,traj_len,x_dim)
#     for i in range(1):
#         plt.plot(atraj[i,:,0],atraj[i,:,1])
#     decoded_traj1 = decode(myagent,encoded_traj1[idx,:], myagent.agent_sps)
#     adtraj = decoded_traj1.reshape(n_agents,traj_len,x_dim)
#     for i in range(1):
#         plt.plot(adtraj[i,:,0],adtraj[i,:,1],'--')
#     print('ortho sps: ' + str(np.mean(np.sum((atraj - adtraj)**2,axis=2))))
        
#     agent_sps =  ssp_bayes_opt.sspspace.RandomSSPSpace(n_agents, ssp_dim=ssp_dim, length_scale=1).axis_matrix.T
#     encoded_traj1 = encode(myagent,init_xs, agent_sps)
#     idx = 0
#     plt.figure()
#     atraj = init_xs[idx,:].reshape(n_agents,traj_len,x_dim)
#     for i in range(1):
#         plt.plot(atraj[i,:,0],atraj[i,:,1])
        
#     decoded_traj1 = decode(myagent,encoded_traj1[idx,:], agent_sps)
#     adtraj = decoded_traj1.reshape(n_agents,traj_len,x_dim)
#     for i in range(1):
#         plt.plot(adtraj[i,:,0],adtraj[i,:,1],'--')
    
#     print('randssp sps: ' + str(np.mean(np.sum((atraj - adtraj)**2,axis=2))))