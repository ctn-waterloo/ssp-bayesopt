#!/bin/bash

set -e

# num_bins=10
# dest_dir="/home/ns2dumon/Documents/ssp-bayesopt/data/test-funcs"
# dest_dir="${HOME}/Public/data/ssp-bayesopt/test-funcs-variable-ls"
# dest_dir="${HOME}/Public/data/ssp-bayesopt/test-funcs-variable-2ls"
dest_dir="/run/media/furlong/Data/ssp-bayesopt/memory-test"


echo $dest_dir
for func in "branin-hoo" "himmelblau" "goldstein-price"
do
	for agt in "ssp-rand" "ssp-hex" "gp-sinc" "gp-matern" "disc-domain"
	do
# 		python run_agent.py --agent $agt --num-trials 50 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim 151
# 		for num_bins in 10 20 30 40 50 60 70 80 90 100 150 200 250 300 350 400
# 		for num_bins in 3 4 5 6 7 8 9 10
# 		do
		python run_agent.py --agent $agt --num-trials 50 --func $func --data-dir $dest_dir/ --len-scale -1 --ssp-dim 151 
# 	done
	done
done
