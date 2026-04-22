#!/bin/bash

ssp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
matern_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs

for func in himmelblau branin-hoo goldstein-price
# for func in branin-hoo 
do
	python plot-times.py --alg "SSP-BO-Rand" $ssp_dir/$func/ssp-rand --alg "SSP-BO-Hex" $ssp_dir/$func/ssp-hex --alg "GP-MI-Sinc" $sinc_gp_dir/$func/gp-sinc --alg "GP-MI-Matern" $matern_gp_dir/$func/gp-matern --no_legend --save 
done

