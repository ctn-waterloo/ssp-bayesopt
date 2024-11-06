#!/bin/bash

ssp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
matern_gp_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs

for func in himmelblau branin-hoo goldstein-price
do
# 	python plot-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --alg "SSP-Hex" $ssp_dir/$func/ssp-hex --alg "GP-Sinc" $sinc_gp_dir/$func/gp-sinc --alg "GP-Matern" $matern_gp_dir/$func/gp-matern #--save
	python plot-algs.py --alg "SSP-BO-Rand" $ssp_dir/$func/ssp-rand --alg "SSP-BO-Hex" $ssp_dir/$func/ssp-hex --alg "GP-MI-Sinc" $sinc_gp_dir/$func/gp-sinc --alg "GP-MI-Matern" $matern_gp_dir/$func/gp-matern --save --no_legend
	python plot-algs.py --alg "SSP-BO-Rand" $ssp_dir/$func/ssp-rand --alg "SSP-BO-Hex" $ssp_dir/$func/ssp-hex --alg "GP-MI-Sinc" $sinc_gp_dir/$func/gp-sinc --alg "GP-MI-Matern" $matern_gp_dir/$func/gp-matern --save
done

