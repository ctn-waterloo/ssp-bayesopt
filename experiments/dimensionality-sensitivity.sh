#!/bin/bash

dest_dir="tmp-data/dim-sensitivity"
func_name="branin-hoo"
echo $dest_dir

# for dim in 10 25 50 75 100
for dim in 40 151 200
do
	for agt in "ssp-rand" "ssp-hex"
	do
		python run_agent.py --agent $agt --num-trials 30 --func $func_name --data-dir $dest_dir --len-scale -1 --ssp-dim $dim
	done
done
