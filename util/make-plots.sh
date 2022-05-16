#!/bin/bash

# data_dir=/run/media/furlong/Data/ssp-bayesopt/test-funcs
# ssp_dir=/run/media/furlong/Data/ssp-bayesopt/lenscale-fitting
ssp_dir=/run/media/furlong/Data/ssp-bayesopt/lenscale-optim
sinc_gp_dir=/run/media/furlong/Data/ssp-bayesopt/static-sinc
matern_gp_dir=/run/media/furlong/Data/ssp-bayesopt/bak.test-funcs

for func in himmelblau branin-hoo goldstein-price
do
	python plot-algs.py --alg "SSP-Rand" $ssp_dir/$func/ssp-rand --alg "SSP-Hex" $ssp_dir/$func/ssp-hex --alg "GP-Sinc" $sinc_gp_dir/$func/static-gp --alg "GP-Matern" $matern_gp_dir/$func/matern.static-gp --save
done

