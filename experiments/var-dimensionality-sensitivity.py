#!/bin/bash
#dim = np.array([1,2,3,4,5,6,7,8,10,12,14,16,18,20])
#ns =np.ceil(np.sqrt((250 - 1)/(2*dim+2)))
#2*(dim+1)*(ns**2) + 1

import os
import numpy as np

func_name="styblinski-tang" # rastrigin , styblinski-tang  rosenbrock
dims = np.arange(1,10,1)#,6,7,8,10,12,14,16,18])
n_scales = 4
ns =np.ceil((250 - 1)/(2*n_scales*(dims+1)))
ssp_dims = (2*(dims+1)*(ns*n_scales) + 1).astype(int)
ssp_dim = 300

for i,dim in enumerate(dims):
    dest_dir="/home/ns2dumon/Documents/ssp-bayesopt/data/var-dim-sensitivity/" + str(dim) + "/"
    for agt in ["ssp-rand" ,"gp-matern"]: #"ssp-hex", "gp-sinc" 
        os.system("python run_agent.py --agent " + agt
                + " --num-trials 10 --func " + func_name + str(dim) + " --data-dir " + dest_dir 
                + " --len-scale -1 --ssp-dim " + str(ssp_dim))
        # if agt=='ssp-hex':
        #     os.system("python run_agent.py --agent " + agt
        #         + " --num-trials 10 --func " + func_name + str(dim) + " --data-dir " + dest_dir 
        #         + " --len-scale -1 --n_scales " + str(n_scales) + " --n_rotates " + str(ns[i]))
        # else:
        #     os.system("python run_agent.py --agent " + agt
        #         + " --num-trials 10 --func " + func_name + str(dim) + " --data-dir " + dest_dir 
        #         + " --len-scale -1 --ssp-dim " + str(ssp_dims[i]))


