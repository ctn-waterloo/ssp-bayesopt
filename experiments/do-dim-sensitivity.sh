#!/bin/bash

dest_dir="$HOME/Public/data/ssp-bayesopt/domain-dim-sensitivity"
agt="ssp-hex"
func="rosenbrock"
echo $dest_dir

# # for ssp_dim in 2048 1024 256 128 75 
# for ssp_dim in 1024 512 256 128 75 
# for ssp_dim in 75 #128 256 512 1024 
# do
# # 	for domain_dim in 2 3 5 10 13 15
# 	for domain_dim in 5 #15
# 	do
# 		python run_domain_dim_sensitivity.py --agent $agt --num-trials 30 --func $func --data-dir $dest_dir --func-dim $domain_dim --ssp-dim $ssp_dim
# 	done
# done

for domain_dim in 2 3 4 5 6 7 8 9 10
do
	python run_domain_dim_sensitivity.py --agent gp-matern --num-trials 30 --func $func --data-dir $dest_dir --func-dim $domain_dim  --len-scale 5
done
