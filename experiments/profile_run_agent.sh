#!/bin/bash

dest_dir="tmp-data/profiling"
echo $dest_dir

for agt in "ssp-rand" "ssp-hex"
do
	python -m cProfile -o $dest_dir/profile-$agt.out run_agent.py --agent $agt --num-trials 1 --func branin-hoo --data-dir $dest_dir
done
