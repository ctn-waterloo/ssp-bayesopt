#!/bin/bash

dest_dir="tmp-data/`git branch --show-current`"
echo $dest_dir

for agt in "ssp-rand" "ssp-hex"
do
	python run_agent.py --agent $agt --num-trials 30 --func branin-hoo --data-dir $dest_dir
done
