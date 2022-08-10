#!/bin/bash

ssp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
matern_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs

for func in himmelblau branin-hoo goldstein-price
do
	
	python compare-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --baseline "GP-Matern" $matern_gp_dir/$func/gp-matern | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Hex" $ssp_dir/$func/ssp-hex --baseline "GP-Matern" $matern_gp_dir/$func/gp-matern | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --baseline "GP-Sinc" $sinc_gp_dir/$func/gp-sinc | tee -a $func-stats.txt

	python compare-algs.py --alg "SSP-Hex" $ssp_dir/$func/ssp-hex  --baseline "GP-Sinc" $sinc_gp_dir/$func/gp-sinc | tee -a $func-stats.txt

	python compare-algs.py --alg "GP-Sinc" $sinc_gp_dir/$func/gp-sinc --baseline "GP-Matern" $matern_gp_dir/$func/gp-matern | tee -a $func-stats.txt

done

