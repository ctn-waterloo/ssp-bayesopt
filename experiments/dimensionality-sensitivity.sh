#!/bin/bash

func_name="branin-hoo"

for dim in 7 25 55 97 151 217 295
do
    dest_dir="/home/ns2dumon/Documents/ssp-bayesopt/data/dim-sensitivity/"$dim"/"
    echo $dest_dir
	for agt in "ssp-rand" "ssp-hex"
	do
		python run_agent.py --agent $agt --num-trials 10 --func $func_name --data-dir $dest_dir --len-scale -1 --ssp-dim $dim
	done
done
