#!/bin/bash

dest_dir="/home/ns2dumon/Documents/ssp-bayesopt/data/test-funcs"
echo $dest_dir
for func in "branin-hoo" "himmelblau" "goldstein-price"
do
	for agt in "ssp-rand" "ssp-hex" "gp-sinc" "gp-matern"
	do
		python run_agent.py --agent $agt --num-trials 50 --func $func --data-dir $dest_dir --len-scale -1 --ssp-dim 151
	done
done
