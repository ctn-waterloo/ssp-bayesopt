#!/bin/bash

dest_dir="$HOME/Public/data/ssp-bayesopt/profiling"
echo $dest_dir

for func in "branin-hoo" "himmelblau" "goldstein-price"
do
	for agt in "ssp-rand" "ssp-hex"
	do
		python run_agent.py --agent $agt --num-trials 30 --func $func --data-dir $dest_dir --len-scale -1
	done
done
