#!/bin/bash

# sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
# ssp_dir=/run/media/furlong/Data/ssp-bayesopt/lenscale-fitting
ssp_dir=/run/media/furlong/Data/ssp-bayesopt/lenscale-optim
sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/static-sinc/
matern_gp_dir=/run/media/furlong/Data/ssp-bayesopt/bak.test-funcs

for func in himmelblau branin-hoo goldstein-price
# for func in branin-hoo 
do
	
	python compare-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --baseline "GP-Matern" $matern_gp_dir/$func/matern.static-gp | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Hex" $ssp_dir/$func/ssp-hex --baseline "GP-Matern" $matern_gp_dir/$func/matern.static-gp | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --baseline "GP-Sinc" $sinc_gp_dir/$func/static-gp | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Hex" $ssp_dir/$func/ssp-hex  --baseline "GP-Sinc" $sinc_gp_dir/$func/static-gp | tee -a $func-stats.txt

	python compare-algs.py --alg "GP-Sinc" $sinc_gp_dir/$func/static-gp --baseline "GP-Matern" $matern_gp_dir/$func/matern.static-gp | tee -a $func-stats.txt

done

