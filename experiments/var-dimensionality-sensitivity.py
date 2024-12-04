#!/bin/bash
#dim = np.array([1,2,3,4,5,6,7,8,10,12,14,16,18,20])
#ns =np.ceil(np.sqrt((250 - 1)/(2*dim+2)))
#2*(dim+1)*(ns**2) + 1

import os
import numpy as np

func_name="styblinski-tang" # rastrigin , styblinski-tang  rosenbrock
# func_name="rastrigin" # rastrigin , styblinski-tang  rosenbrock
dims = np.arange(1,10,1)#,6,7,8,10,12,14,16,18])
n_scales = 4
ns =np.ceil((250 - 1)/(2*n_scales*(dims+1)))
ssp_dims = (2*(dims+1)*(ns*n_scales) + 1).astype(int)

ssp_dims = 2**np.array([5, 6, 7, 8, 9, 10])
# ssp_dims = np.array([1024])
# ssp_dim = 300

#         dest_dir="/home/ns2dumon/Documents/ssp-bayesopt/data/var-dim-sensitivity/" + str(dim) + "/"

num_trials = 10
# for i,dim in enumerate(dims):
for i,dim in enumerate([8,9]):
    # run the gp model
    dest_dir=f"/run/media/furlong/Data/bo-ssp/var-dim-sensitivity/{dim}/" 
    os.system("python run_agent.py --agent gp-matern" 
            + " --num-trials 30 --func " + func_name + str(dim) 
            + " --data-dir " + dest_dir 
            + " --len-scale -1 "
        )
    os.system("python run_agent.py --agent gp-sinc" 
            + f" --num-trials {num_trials} --func " + func_name + str(dim) 
            + " --data-dir " + dest_dir 
            + f" --len-scale {len_scale} "
            + f" --len-scale -1 "
        )
    # run the ssp models
    for ssp_dim in ssp_dims:
        ssp_dest_dir = dest_dir + f'{ssp_dim}/'
        for agt in ["ssp-rand"]: #"ssp-hex", "gp-sinc" 
            os.system("python run_agent.py --agent " + agt
                    + f" --num-trials {num_trials} --func " + func_name + str(dim) 
                    + " --data-dir " + ssp_dest_dir 
                    + f" --len-scale -1 --ssp-dim " + str(ssp_dim)
            )
        ### end for agt in ['ssp-rand']
    ### end for ssp_dim in ssp_dims
### end for i, dim in enumerate(dims)

        # if agt=='ssp-hex':
        #     os.system("python run_agent.py --agent " + agt
        #         + " --num-trials 10 --func " + func_name + str(dim) + " --data-dir " + dest_dir 
        #         + " --len-scale -1 --n_scales " + str(n_scales) + " --n_rotates " + str(ns[i]))
        # else:
        #     os.system("python run_agent.py --agent " + agt
        #         + " --num-trials 10 --func " + func_name + str(dim) + " --data-dir " + dest_dir 
        #         + " --len-scale -1 --ssp-dim " + str(ssp_dims[i]))


